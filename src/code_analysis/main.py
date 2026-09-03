"""Command-line entry point for code-analysis."""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from code_analysis.schemas import AnalysisResult, Finding  # noqa: F401
from code_analysis.tool_schemas import (
    APPLY_PATCH_TOOL,
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    RUN_DART_ANALYZE_TOOL,
    RUN_DART_FORMAT_TOOL,
    RUN_FLUTTER_TESTS_TOOL,
    SEARCH_CODE_TOOL,
    FileAccessPolicy,
    FileState,
    PatchRequest,
    PatchResult,
    ToolPermission,
    apply_patch,
    get_tool_permissions,
    list_files,
    read_file,
    run_dart_analyze,
    run_dart_format,
    run_flutter_tests,
    search_code,
)
from code_analysis.utils import (
    analyzer_issue_count as _analyzer_issue_count,
)
from code_analysis.utils import (
    analyzer_output as _analyzer_output,
)
from code_analysis.utils import (
    analyzer_output_diff as _analyzer_output_diff,
)
from code_analysis.utils import (
    format_analysis_metrics,  # noqa: F401
    format_analysis_result,  # noqa: F401
    format_analysis_result_object,
    format_changed_files,
    format_output_items,  # noqa: F401
    format_response,  # noqa: F401
    format_run_metrics,
    format_token_usage,  # noqa: F401
    max_output_tokens,  # noqa: F401
    number_source_lines,
    temperature,  # noqa: F401
)

MODEL = "gpt-5.6-luna"
AGENT_INSTRUCTIONS = """You are a software repository exploration, read, write agent.

Your job is to answer questions about the supplied repository using the available tools.

Rules:

- Inspect repository code before making claims.
- Do not guess about implementations you have not inspected.
- Use search_code when locating symbols or concepts.
- Use list_files when you need to understand directory structure.
- Use read_file when you need implementation context.
- Prefer targeted exploration over reading large numbers of files.
- Reference file paths and line numbers in your final answer.
- Before modifying a file, inspect the relevant current contents.
- Do not guess the current contents of a file.
- Use apply_patch only for small, targeted edits.
- Do not overwrite entire files unless explicitly required.
- You have a limited tool-call budget.
- Stop exploring when you have enough evidence to answer.

For edit requests involving a specific literal, identifier, symbol,
or exact text:

- Treat the user's specified target as authoritative.
- You may inspect nearby files/context to locate the target.
- Do not replace a different or similar-looking value.
- Do not infer that another value is a typo or intended target.
- If the exact target cannot be found after reasonable inspection,
  make no changes and report that the target was not found.

Only run verification tools after a change has actually been made,
unless the user explicitly asks to run them independently.

When a code modification causes static analysis or tests to fail:

- Inspect the failure output.
- Determine whether the failure is related to your change.
- If it is related, inspect the relevant code before modifying it again.
- Apply the smallest reasonable correction.
- Run verification again.
- Do not repeatedly make speculative edits.
- Stop if you cannot determine a grounded fix.
- Make no more than three repair attempts in one agent run.
- Run dart analyze before applying a patch to establish a baseline, and again after a
  successful patch to verify it.

"""
MAX_VALIDATION_RETRIES = 2
MAX_TOOL_CALL_ROUNDS = 15
MAX_REPAIR_ATTEMPTS = 3


class AnalysisError(Exception):
    """Raised when the model request cannot be completed."""


@dataclass
class AgentRunState:
    """All workflow state the harness retains throughout one agent run."""

    changed_file_paths: set[str] = field(default_factory=set)
    verification_dirty: bool = False
    repair_attempts: int = 0
    read_file_paths: set[str] = field(default_factory=set)
    analyzer_baseline_output: str | None = None
    analyzer_baseline_issue_count: int | None = None
    analyzer_issues_found: bool = False
    verification_failed: bool = False
    model_calls: int = 0
    tool_calls: int = 0
    searched_file_paths: set[str] = field(default_factory=set)
    patch_attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class AgentCompletionCheck:
    """The harness's assessment of whether an agent run completed safely."""

    passed: bool
    failures: tuple[str, ...]


class RunMetrics(BaseModel):
    """Aggregate measurements from one complete agent run."""

    model_calls: int
    tool_calls: int
    files_searched: int
    files_read: int
    files_changed: int
    patch_attempts: int
    repair_attempts: int
    input_tokens: int
    output_tokens: int
    runtime_seconds: float
    completion_passed: bool


@dataclass(frozen=True)
class PermissionPolicy:
    """The permissions this app run grants to model-requested tools."""

    allowed_permissions: frozenset[ToolPermission]

    def allows(self, tool_name: str) -> bool:
        """Return whether every permission required by a known tool is allowed."""
        required_permissions = get_tool_permissions(tool_name)
        return bool(required_permissions) and required_permissions <= self.allowed_permissions


DEFAULT_PERMISSION_POLICY = PermissionPolicy(
    frozenset({ToolPermission.READ, ToolPermission.WRITE, ToolPermission.EXECUTE})
)
ALL_TOOLS = [
    READ_FILE_TOOL,
    LIST_FILES_TOOL,
    SEARCH_CODE_TOOL,
    RUN_FLUTTER_TESTS_TOOL,
    RUN_DART_ANALYZE_TOOL,
    RUN_DART_FORMAT_TOOL,
    APPLY_PATCH_TOOL,
]


def permission_policy_from_cli_values(
    values: list[str] | None,
) -> PermissionPolicy:
    """Build a policy from repeatable CLI values, defaulting to read-only access."""
    if values is None:
        return DEFAULT_PERMISSION_POLICY

    return PermissionPolicy(frozenset(ToolPermission(value) for value in values))


def format_permission_policy(permission_policy: PermissionPolicy) -> str:
    """Format the active permissions for the terminal."""
    permissions = ", ".join(sorted(permission_policy.allowed_permissions)) or "none"
    return f"Allowed permissions: {permissions}"


def root_path(value: str) -> Path:
    """Parse an existing project root supplied by the CLI user."""
    path = Path(value).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"root path is not a directory: {value}")
    return path


def instructions_for_file_access_policy(file_access_policy: FileAccessPolicy | None) -> str:
    """Tell the model the app-controlled root used by its filesystem tools."""
    if file_access_policy is None:
        return AGENT_INSTRUCTIONS

    root = file_access_policy.root_path.resolve()
    return (
        f"{AGENT_INSTRUCTIONS} The approved project root is {root}. "
        "Use only paths relative to this root when calling filesystem tools."
    )


def tools_allowed_by(permission_policy: PermissionPolicy) -> list[dict[str, object]]:
    """Return only tool schemas that the active policy allows the app to execute."""
    return [tool for tool in ALL_TOOLS if permission_policy.allows(tool["name"])]


def read_text_file(file_path: str) -> str:
    """Read a UTF-8 text file for use as model input."""
    return Path(file_path).read_text(encoding="utf-8")


def read_text_file_with_retry(file_path: str) -> str:
    """Read a file, prompting for a replacement path when it cannot be read."""
    while True:
        try:
            return read_text_file(file_path)
        except OSError as error:
            print(f"Could not read {file_path}: {error.strerror or error}")
            file_path = input("Enter a valid file path: ").strip()


def validate_analysis_result_source_lines(
    result: AnalysisResult,
    numbered_source: str,
) -> None:
    """Ensure each finding references lines included in the model input."""
    if not numbered_source:
        return

    source_line_count = len(numbered_source.splitlines())

    for finding_number, finding in enumerate(result.findings, start=1):
        if finding.end_line > source_line_count:
            raise AnalysisError(
                f"Finding {finding_number} references lines {finding.start_line}-"
                f"{finding.end_line}, but the input contains only {source_line_count} lines"
            )


def execute_file_tool_calls(
    response: object,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    file_access_policy: FileAccessPolicy | None = None,
    run_state: AgentRunState | None = None,
) -> list[dict[str, str]]:
    """Execute requested file tools and format their results for the API."""
    tool_outputs = []
    run_state = run_state if run_state is not None else AgentRunState()

    for output_item in getattr(response, "output", []):
        if getattr(output_item, "type", None) != "function_call":
            continue
        run_state.tool_calls += 1

        try:
            arguments = json.loads(output_item.arguments)
            if not permission_policy.allows(output_item.name):
                required_permissions = ", ".join(sorted(get_tool_permissions(output_item.name)))
                raise PermissionError(
                    f"Tool '{output_item.name}' is not allowed. Required permissions: "
                    f"{required_permissions or 'unknown'}."
                )

            if file_access_policy is None:
                raise ValueError("A file access policy is required to execute tools.")

            if output_item.name == READ_FILE_TOOL["name"]:
                if not isinstance(arguments, dict) or set(arguments) != {"file_path"}:
                    raise ValueError("tool arguments must contain only file_path")

                file_path = arguments["file_path"]
                print(f"Tool call: read_file({file_path})")
                source_text = read_file(file_path, file_access_policy)
                run_state.read_file_paths.add(file_path)
                file_state = FileState(
                    file_path=file_path,
                    line_count=len(source_text.splitlines()),
                    numbered_content=number_source_lines(source_text),
                )
                output = json.dumps(asdict(file_state))
                print(f"Tool result: read {file_path} ({file_state.line_count} lines)")
            elif output_item.name == LIST_FILES_TOOL["name"]:
                if not isinstance(arguments, dict) or set(arguments) != {"directory_path"}:
                    raise ValueError("tool arguments must contain only directory_path")

                directory_path = arguments["directory_path"]
                print(f"Tool call: list_files({directory_path})")
                output = list_files(directory_path, file_access_policy)
                print(f"Tool result: listed {len(output.splitlines())} files")
            elif output_item.name == SEARCH_CODE_TOOL["name"]:
                if not isinstance(arguments, dict) or set(arguments) != {"query"}:
                    raise ValueError("tool arguments must contain only query")

                query = arguments["query"]
                print(f"Tool call: search_code({query})")
                search_result = search_code(query, file_access_policy)
                run_state.searched_file_paths.update(match.path for match in search_result.matches)
                output = json.dumps(asdict(search_result))
                print(
                    "Tool result: found "
                    f"{search_result.total_matches} matches "
                    f"(returned {len(search_result.matches)})"
                )
                # print(f"Tool output:\n{json.dumps(asdict(search_result), indent=2)}")
            elif output_item.name == RUN_FLUTTER_TESTS_TOOL["name"]:
                if not isinstance(arguments, dict) or arguments:
                    raise ValueError("run_flutter_tests does not accept arguments")

                print("Tool call: run_flutter_tests()")
                test_result = run_flutter_tests(file_access_policy.root_path)
                run_state.verification_failed = not test_result.success
                output = json.dumps(asdict(test_result))
                print(f"Tool result: flutter test exited with code {test_result.exit_code}")
                print(f"Tool output:\n{json.dumps(asdict(test_result), indent=2)}")
            elif output_item.name == RUN_DART_ANALYZE_TOOL["name"]:
                if not isinstance(arguments, dict) or arguments:
                    raise ValueError("run_dart_analyze does not accept arguments")

                print("Tool call: run_dart_analyze()")
                analyze_result = run_dart_analyze(file_access_policy.root_path)
                run_state.verification_failed = not analyze_result.success
                analyzer_output = _analyzer_output(analyze_result)
                issue_count = _analyzer_issue_count(analyzer_output)
                run_state.analyzer_issues_found = (
                    issue_count is not None and issue_count > 0
                ) or bool(analyze_result.issues)
                baseline_diff = ""
                new_issues_introduced = False
                if run_state.verification_dirty:
                    baseline_diff = _analyzer_output_diff(
                        run_state.analyzer_baseline_output or "",
                        analyzer_output,
                    )
                    new_issues_introduced = (
                        issue_count is not None
                        and run_state.analyzer_baseline_issue_count is not None
                        and issue_count > run_state.analyzer_baseline_issue_count
                    )
                    run_state.verification_dirty = not analyze_result.success
                run_state.analyzer_baseline_output = analyzer_output
                run_state.analyzer_baseline_issue_count = issue_count
                output = json.dumps(
                    {
                        "analysis": asdict(analyze_result),
                        "issue_count": issue_count,
                        "baseline_diff": baseline_diff or None,
                        "new_issues_introduced": new_issues_introduced,
                    }
                )
                if analyze_result.timed_out:
                    print("Tool result: dart analyze timed out")
                else:
                    print(f"Tool result: dart analyze exited with code {analyze_result.exit_code}")
                if issue_count is None:
                    print("Tool result: dart analyze issue count unavailable")
                else:
                    print(f"Tool result: dart analyze reported {issue_count} issues")
                if new_issues_introduced:
                    print("Tool result: new analyzer issues introduced")
            elif output_item.name == RUN_DART_FORMAT_TOOL["name"]:
                if not isinstance(arguments, dict) or arguments:
                    raise ValueError("run_dart_format does not accept arguments")

                print("Tool call: run_dart_format()")
                format_result = run_dart_format(file_access_policy.root_path)
                output = json.dumps(asdict(format_result))
                print(f"Tool result: dart format exited with code {format_result.exit_code}")
                print(f"Tool output:\n{json.dumps(asdict(format_result), indent=2)}")
            elif output_item.name == APPLY_PATCH_TOOL["name"]:
                if not isinstance(arguments, dict) or set(arguments) != {
                    "path",
                    "old_text",
                    "new_text",
                }:
                    raise ValueError(
                        "tool arguments must contain only path, old_text, and new_text"
                    )

                patch_request = PatchRequest(**arguments)
                print(f"Tool call: apply_patch({patch_request.path})")
                run_state.patch_attempts += 1
                if run_state.analyzer_baseline_output is None:
                    patch_result = PatchResult(
                        success=False,
                        path=patch_request.path,
                        replacements=0,
                        message="Run dart analyze before applying a patch to establish a baseline.",
                    )
                elif run_state.analyzer_issues_found:
                    if run_state.repair_attempts >= MAX_REPAIR_ATTEMPTS:
                        patch_result = PatchResult(
                            success=False,
                            path=patch_request.path,
                            replacements=0,
                            message=(
                                f"Repair attempt limit reached ({MAX_REPAIR_ATTEMPTS} attempts)."
                            ),
                        )
                    else:
                        run_state.repair_attempts += 1
                        patch_result = apply_patch(
                            patch_request,
                            file_access_policy,
                            run_state.read_file_paths,
                        )
                else:
                    patch_result = apply_patch(
                        patch_request,
                        file_access_policy,
                        run_state.read_file_paths,
                    )
                if patch_result.success:
                    run_state.changed_file_paths.add(patch_result.path)
                    run_state.verification_dirty = True
                output = json.dumps(asdict(patch_result))
                print(f"Tool result: {output}")
            else:
                raise ValueError(f"unsupported tool: {output_item.name}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            output = f"Could not execute file tool: {error}"
            print(f"Tool error: {output}")

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": output_item.call_id,
                "output": output,
            }
        )

    return tool_outputs


def request_analysis_with_tools(
    text: str,
    instructions: str,
    client: OpenAI,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    file_access_policy: FileAccessPolicy | None = None,
    run_state: AgentRunState | None = None,
) -> object:
    """Request an analysis and service file-read calls until it is complete."""
    request_options = {
        "model": MODEL,
        "instructions": instructions_for_file_access_policy(file_access_policy),
        "text_format": AnalysisResult,
        "tools": tools_allowed_by(permission_policy),
        "tool_choice": "auto",
    }
    request_options["input"] = number_source_lines(text) if text else instructions

    run_state = run_state if run_state is not None else AgentRunState()
    response = client.responses.parse(**request_options)
    _record_model_response(response, run_state)

    for tool_round in range(MAX_TOOL_CALL_ROUNDS):
        tool_outputs = execute_file_tool_calls(
            response,
            permission_policy,
            file_access_policy,
            run_state,
        )
        if not tool_outputs:
            completion = check_agent_completion(run_state)
            if completion.passed:
                return response
            rejection_message = completion_rejection_message(completion)
            print(rejection_message["content"])
            tool_outputs = [rejection_message]

        response = client.responses.parse(
            model=MODEL,
            instructions=instructions_for_file_access_policy(file_access_policy),
            input=tool_outputs,
            previous_response_id=response.id,
            text_format=AnalysisResult,
            tools=tools_allowed_by(permission_policy),
            tool_choice="auto",
        )
        _record_model_response(response, run_state)

    raise AnalysisError(f"Model requested more than {MAX_TOOL_CALL_ROUNDS} rounds of tooling.")


def _record_model_response(response: object, run_state: AgentRunState) -> None:
    """Accumulate one completed Responses API call in the run metrics."""
    run_state.model_calls += 1
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    run_state.input_tokens += getattr(usage, "input_tokens", 0) or 0
    run_state.output_tokens += getattr(usage, "output_tokens", 0) or 0


def analyze_text(
    text: str,
    instructions: str,
    client: OpenAI,
    simulate_validation_error_once: bool = False,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    file_access_policy: FileAccessPolicy | None = None,
    run_state: AgentRunState | None = None,
) -> object:
    """Send text for analysis and return a response containing an AnalysisResult."""
    numbered_source = number_source_lines(text)

    run_state = run_state if run_state is not None else AgentRunState()
    for attempt in range(MAX_VALIDATION_RETRIES + 1):
        try:
            if simulate_validation_error_once and attempt == 0:
                AnalysisResult.model_validate({})

            response = request_analysis_with_tools(
                text,
                instructions,
                client,
                permission_policy,
                file_access_policy,
                run_state,
            )
        except ValidationError as error:
            validation_error = error
        except Exception as error:
            raise AnalysisError(f"OpenAI request failed: {error}") from error
        else:
            if response.output_parsed is not None:
                break
            validation_error = ValueError("response did not contain a parsed analysis result")

        if attempt == MAX_VALIDATION_RETRIES:
            raise AnalysisError(
                "OpenAI returned an invalid analysis after "
                f"{attempt + 1} attempts: {validation_error}"
            ) from validation_error

        print(f"Invalid analysis response. Retrying ({attempt + 1}/{MAX_VALIDATION_RETRIES})...")

    validate_analysis_result_source_lines(response.output_parsed, numbered_source)

    return response


def build_run_metrics(
    run_state: AgentRunState,
    runtime_seconds: float,
    completion: AgentCompletionCheck,
) -> RunMetrics:
    """Build the final metrics object from locally tracked harness state."""
    return RunMetrics(
        model_calls=run_state.model_calls,
        tool_calls=run_state.tool_calls,
        files_searched=len(run_state.searched_file_paths),
        files_read=len(run_state.read_file_paths),
        files_changed=len(run_state.changed_file_paths),
        patch_attempts=run_state.patch_attempts,
        repair_attempts=run_state.repair_attempts,
        input_tokens=run_state.input_tokens,
        output_tokens=run_state.output_tokens,
        runtime_seconds=runtime_seconds,
        completion_passed=completion.passed,
    )


def check_agent_completion(run_state: AgentRunState) -> AgentCompletionCheck:
    """Return completion failures derived from the final local agent-run state."""
    failures = []
    if run_state.verification_dirty:
        failures.append(
            "A file was changed but has not been verified. Run the appropriate verification tool."
        )
    if run_state.verification_failed:
        failures.append(
            "The most recent verification failed. Inspect its output before making a "
            "grounded repair."
        )
    if run_state.repair_attempts >= MAX_REPAIR_ATTEMPTS and (
        run_state.verification_dirty or run_state.verification_failed
    ):
        failures.append(
            f"The repair limit of {MAX_REPAIR_ATTEMPTS} attempts was reached. Do not make "
            "another repair; report the unresolved verification failure."
        )

    return AgentCompletionCheck(passed=not failures, failures=tuple(failures))


def completion_rejection_message(completion: AgentCompletionCheck) -> dict[str, str]:
    """Format a harness-controlled continuation for an incomplete agent run."""
    return {
        "role": "system",
        "content": (
            "Completion rejected by the harness:\n"
            + "\n".join(completion.failures)
            + "\nContinue working on the task."
        ),
    }


def main() -> None:
    """Analyze a text file supplied from the command line."""
    parser = argparse.ArgumentParser(description="Analyze source files with the available tools.")
    parser.add_argument("instructions", help="Instructions for the analysis")
    parser.add_argument(
        "--root-path",
        required=True,
        type=root_path,
        help="Project root that bounds all filesystem tools for this run",
    )
    parser.add_argument(
        "--simulate-validation-error-once",
        action="store_true",
        help="Testing only: force one validation error before calling the API",
    )
    parser.add_argument(
        "--allow-permission",
        action="append",
        choices=[permission.value for permission in ToolPermission],
        metavar="PERMISSION",
        help=(
            "Grant a tool permission; repeat for multiple permissions. "
            "When omitted, only read is allowed."
        ),
    )
    args = parser.parse_args()
    permission_policy = permission_policy_from_cli_values(args.allow_permission)
    file_access_policy = FileAccessPolicy(root_path=args.root_path)

    try:
        client = OpenAI()
    except Exception as error:
        parser.error(f"could not initialize OpenAI client: {error}")

    analysis_started_at = time.perf_counter()
    run_state = AgentRunState()
    print(format_permission_policy(permission_policy))

    try:
        response = analyze_text(
            "",
            args.instructions,
            client,
            simulate_validation_error_once=args.simulate_validation_error_once,
            permission_policy=permission_policy,
            file_access_policy=file_access_policy,
            run_state=run_state,
        )
    except AnalysisError as error:
        parser.error(str(error))

    analysis_elapsed_seconds = time.perf_counter() - analysis_started_at

    completion = check_agent_completion(run_state)
    metrics = build_run_metrics(run_state, analysis_elapsed_seconds, completion)
    print(format_analysis_result_object(response.output_parsed))
    print(format_run_metrics(metrics))
    print(format_changed_files(run_state.changed_file_paths))
    print(f"Completion Check: {json.dumps(asdict(completion))}")
    # print(format_token_usage(response))


if __name__ == "__main__":
    main()
