"""Command-line entry point for code-analysis."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from code_analysis.schemas import AnalysisResult, Finding  # noqa: F401
from code_analysis.tool_schemas import (
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    RUN_DART_ANALYZE_TOOL,
    RUN_DART_FORMAT_TOOL,
    RUN_FLUTTER_TESTS_TOOL,
    SEARCH_CODE_TOOL,
    FileState,
    ToolPermission,
    get_tool_permissions,
    list_files,
    read_file,
    run_dart_analyze,
    run_dart_format,
    run_flutter_tests,
    search_code,
)

MODEL = "gpt-5.6-luna"
AGENT_INSTRUCTIONS = "You are a code-analysis agent. Do not make claims about implementation unless those claims are supported by files you have inspected using the available tools. Return a structured analysis result. "
MAX_VALIDATION_RETRIES = 2
MAX_TOOL_CALL_ROUNDS = 15
INPUT_TOKEN_COST_PER_MILLION = 0.20
OUTPUT_TOKEN_COST_PER_MILLION = 1.20


class AnalysisError(Exception):
    """Raised when the model request cannot be completed."""


@dataclass(frozen=True)
class PermissionPolicy:
    """The permissions this app run grants to model-requested tools."""

    allowed_permissions: frozenset[ToolPermission]

    def allows(self, tool_name: str) -> bool:
        """Return whether every permission required by a known tool is allowed."""
        required_permissions = get_tool_permissions(tool_name)
        return bool(required_permissions) and required_permissions <= self.allowed_permissions


DEFAULT_PERMISSION_POLICY = PermissionPolicy(frozenset({ToolPermission.READ}))
ALL_TOOLS = [
    READ_FILE_TOOL,
    LIST_FILES_TOOL,
    SEARCH_CODE_TOOL,
    RUN_FLUTTER_TESTS_TOOL,
    RUN_DART_ANALYZE_TOOL,
    RUN_DART_FORMAT_TOOL,
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


# def temperature(value: str) -> float:
#     """Parse a temperature value accepted by the Responses API."""
#     parsed_value = float(value)
#     if not 0 <= parsed_value <= 2:
#         raise argparse.ArgumentTypeError("temperature must be between 0 and 2")
#     return parsed_value


# def max_output_tokens(value: str) -> int:
#     """Parse an output-token limit accepted by the selected model."""
#     parsed_value = int(value)
#     if not 1 <= parsed_value <= 128_000:
#         raise argparse.ArgumentTypeError("max output tokens must be between 1 and 128000")
#     return parsed_value


def number_source_lines(text: str) -> str:
    """Add one-based line numbers so the model can reference source locations."""
    return "\n".join(
        f"{line_number}: {line}" for line_number, line in enumerate(text.splitlines(), start=1)
    )


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
) -> list[dict[str, str]]:
    """Execute requested file tools and format their results for the API."""
    tool_outputs = []

    for output_item in getattr(response, "output", []):
        if getattr(output_item, "type", None) != "function_call":
            continue

        try:
            arguments = json.loads(output_item.arguments)
            if not permission_policy.allows(output_item.name):
                required_permissions = ", ".join(sorted(get_tool_permissions(output_item.name)))
                raise PermissionError(
                    f"Tool '{output_item.name}' is not allowed. Required permissions: "
                    f"{required_permissions or 'unknown'}."
                )

            if output_item.name == READ_FILE_TOOL["name"]:
                if not isinstance(arguments, dict) or set(arguments) != {"file_path"}:
                    raise ValueError("tool arguments must contain only file_path")

                file_path = arguments["file_path"]
                print(f"Tool call: read_file({file_path})")
                source_text = read_file(file_path)
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
                output = list_files(directory_path)
                print(f"Tool result: listed {len(output.splitlines())} files")
                print(f"Tool output:\n{output or '(no files found)'}")
            elif output_item.name == SEARCH_CODE_TOOL["name"]:
                if not isinstance(arguments, dict) or set(arguments) != {"query"}:
                    raise ValueError("tool arguments must contain only query")

                query = arguments["query"]
                print(f"Tool call: search_code({query})")
                search_result = search_code(query)
                output = json.dumps(asdict(search_result))
                print(
                    "Tool result: found "
                    f"{search_result.total_matches} matches "
                    f"(returned {len(search_result.matches)})"
                )
                print(f"Tool output:\n{json.dumps(asdict(search_result), indent=2)}")
            elif output_item.name == RUN_FLUTTER_TESTS_TOOL["name"]:
                if not isinstance(arguments, dict) or arguments:
                    raise ValueError("run_flutter_tests does not accept arguments")

                print("Tool call: run_flutter_tests()")
                test_result = run_flutter_tests()
                output = json.dumps(asdict(test_result))
                print(f"Tool result: flutter test exited with code {test_result.exit_code}")
                print(f"Tool output:\n{json.dumps(asdict(test_result), indent=2)}")
            elif output_item.name == RUN_DART_ANALYZE_TOOL["name"]:
                if not isinstance(arguments, dict) or arguments:
                    raise ValueError("run_dart_analyze does not accept arguments")

                print("Tool call: run_dart_analyze()")
                analyze_result = run_dart_analyze()
                output = json.dumps(asdict(analyze_result))
                print(f"Tool result: dart analyze exited with code {analyze_result.exit_code}")
                print(f"Tool output:\n{json.dumps(asdict(analyze_result), indent=2)}")
            elif output_item.name == RUN_DART_FORMAT_TOOL["name"]:
                if not isinstance(arguments, dict) or arguments:
                    raise ValueError("run_dart_format does not accept arguments")

                print("Tool call: run_dart_format()")
                format_result = run_dart_format()
                output = json.dumps(asdict(format_result))
                print(f"Tool result: dart format exited with code {format_result.exit_code}")
                print(f"Tool output:\n{json.dumps(asdict(format_result), indent=2)}")
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
) -> object:
    """Request an analysis and service file-read calls until it is complete."""
    request_options = {
        "model": MODEL,
        "instructions": AGENT_INSTRUCTIONS,
        "text_format": AnalysisResult,
        "tools": tools_allowed_by(permission_policy),
        "tool_choice": "auto",
    }
    request_options["input"] = number_source_lines(text) if text else instructions

    response = client.responses.parse(**request_options)

    for tool_round in range(MAX_TOOL_CALL_ROUNDS):
        tool_outputs = execute_file_tool_calls(response, permission_policy)
        if not tool_outputs:
            return response

        response = client.responses.parse(
            model=MODEL,
            instructions=AGENT_INSTRUCTIONS,
            input=tool_outputs,
            previous_response_id=response.id,
            text_format=AnalysisResult,
            tools=tools_allowed_by(permission_policy),
            tool_choice="auto",
        )

    raise AnalysisError(f"Model requested more than {MAX_TOOL_CALL_ROUNDS} rounds of tooling.")


def analyze_text(
    text: str,
    instructions: str,
    client: OpenAI,
    simulate_validation_error_once: bool = False,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
) -> object:
    """Send text for analysis and return a response containing an AnalysisResult."""
    numbered_source = number_source_lines(text)

    for attempt in range(MAX_VALIDATION_RETRIES + 1):
        try:
            if simulate_validation_error_once and attempt == 0:
                AnalysisResult.model_validate({})

            response = request_analysis_with_tools(text, instructions, client, permission_policy)
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


def format_response(response: object) -> str:
    """Serialize an SDK response as formatted JSON for terminal output."""
    return json.dumps(response.model_dump(), indent=2)


def format_output_items(response: object) -> str:
    """Serialize the model's typed output items as formatted JSON."""
    return json.dumps([item.model_dump() for item in response.output], indent=2)


def format_analysis_result(response: object) -> str:
    """Serialize the parsed, application-specific analysis result."""
    return response.output_parsed.model_dump_json(indent=2)


def format_analysis_result_object(result: AnalysisResult) -> str:
    """Format an AnalysisResult object as a readable terminal report."""
    lines = ["Analysis Result", f"Summary: {result.summary}", "", "Findings:"]

    if not result.findings:
        lines.append("  No issues found.")

    for number, finding in enumerate(result.findings, start=1):
        location = f"lines {finding.start_line}-{finding.end_line}"
        if finding.start_character is not None and finding.end_character is not None:
            location += f", characters {finding.start_character}-{finding.end_character}"

        lines.extend(
            [
                f"  {number}. [{finding.severity.upper()}] {location}",
                f"     Problem: {finding.problem}",
                f"     Solution: {finding.solution}",
            ]
        )

    return "\n".join(lines)


def format_token_usage(response: object) -> str:
    """Format the response's input and output token counts."""
    if response.usage is None:
        return "Token usage: unavailable"

    return (
        "Token usage:\n"
        f"  Input tokens: {response.usage.input_tokens}\n"
        f"  Output tokens: {response.usage.output_tokens}"
    )


def format_analysis_metrics(response: object, elapsed_seconds: float) -> str:
    """Format token usage, estimated token cost, and analysis duration."""
    lines = ["Analysis Metrics"]

    if response.usage is None:
        lines.append("  Token usage: unavailable")
        lines.append("  Estimated token cost: unavailable")
    else:
        input_cost = response.usage.input_tokens * INPUT_TOKEN_COST_PER_MILLION / 1_000_000
        output_cost = response.usage.output_tokens * OUTPUT_TOKEN_COST_PER_MILLION / 1_000_000
        estimated_cost = input_cost + output_cost

        lines.extend(
            [
                f"  Input tokens: {response.usage.input_tokens}",
                f"  Output tokens: {response.usage.output_tokens}",
                f"  Estimated token cost: ${estimated_cost:.6f}",
            ]
        )

    lines.append(f"  Analysis time: {elapsed_seconds:.2f} seconds")
    return "\n".join(lines)


def main() -> None:
    """Analyze a text file supplied from the command line."""
    parser = argparse.ArgumentParser(description="Analyze source files with the available tools.")
    parser.add_argument("instructions", help="Instructions for the analysis")
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

    try:
        client = OpenAI()
    except Exception as error:
        parser.error(f"could not initialize OpenAI client: {error}")

    analysis_started_at = time.perf_counter()
    print(format_permission_policy(permission_policy))

    try:
        response = analyze_text(
            "",
            args.instructions,
            client,
            simulate_validation_error_once=args.simulate_validation_error_once,
            permission_policy=permission_policy,
        )
    except AnalysisError as error:
        parser.error(str(error))

    analysis_elapsed_seconds = time.perf_counter() - analysis_started_at

    print(format_analysis_result_object(response.output_parsed))
    print(format_analysis_metrics(response, analysis_elapsed_seconds))
    # print(format_token_usage(response))


if __name__ == "__main__":
    main()
