import argparse
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from code_analysis import main, tool_schemas
from code_analysis.schemas import AnalysisResult, Finding
from code_analysis.tool_schemas import (
    APPLY_PATCH_TOOL,
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    READ_SOURCE_LINE_TOOL,
    RUN_DART_ANALYZE_TOOL,
    RUN_DART_FORMAT_TOOL,
    RUN_FLUTTER_TESTS_TOOL,
    SEARCH_CODE_MAX_RESULTS,
    SEARCH_CODE_TOOL,
    DartAnalyzeIssue,
    DartAnalyzeResult,
    DartFormatResult,
    FileAccessPolicy,
    FileState,
    FlutterTestResult,
    PatchRequest,
    PatchResult,
    SearchCodeResult,
    SearchResult,
    ToolPermission,
    apply_patch,
    get_tool_permissions,
    list_files,
    read_file,
    read_source_line,
    run_dart_analyze,
    run_dart_format,
    run_flutter_tests,
    search_code,
)


def file_access_policy_for(root_path: Path) -> FileAccessPolicy:
    """Create a default file policy limited to temporary test directories."""
    return FileAccessPolicy(root_path=root_path)


def test_main_is_available() -> None:
    assert callable(main.main)


def test_read_source_line_tool_schema_requires_a_positive_line_number() -> None:
    tool = READ_SOURCE_LINE_TOOL

    assert tool["type"] == "function"
    assert tool["name"] == "read_source_line"
    assert tool["strict"] is True
    assert tool["parameters"] == {
        "type": "object",
        "properties": {
            "line_number": {
                "type": "integer",
                "description": "The one-based source line number to read.",
                "minimum": 1,
            }
        },
        "required": ["line_number"],
        "additionalProperties": False,
    }


def test_read_file_tool_schema_requires_a_file_path() -> None:
    assert READ_FILE_TOOL["name"] == "read_file"
    assert READ_FILE_TOOL["strict"] is True
    assert READ_FILE_TOOL["parameters"]["required"] == ["file_path"]
    assert READ_FILE_TOOL["parameters"]["properties"]["file_path"]["type"] == "string"


def test_list_files_tool_schema_requires_a_directory_path() -> None:
    assert LIST_FILES_TOOL["type"] == "function"
    assert LIST_FILES_TOOL["name"] == "list_files"
    assert LIST_FILES_TOOL["strict"] is True
    assert LIST_FILES_TOOL["parameters"]["required"] == ["directory_path"]
    assert LIST_FILES_TOOL["parameters"]["properties"]["directory_path"]["type"] == "string"


def test_search_code_tool_schema_requires_a_query() -> None:
    assert SEARCH_CODE_TOOL["type"] == "function"
    assert SEARCH_CODE_TOOL["name"] == "search_code"
    assert SEARCH_CODE_TOOL["strict"] is True
    assert SEARCH_CODE_TOOL["parameters"]["required"] == ["query"]
    assert SEARCH_CODE_TOOL["parameters"]["properties"]["query"]["type"] == "string"


def test_run_flutter_tests_tool_schema_has_no_arguments() -> None:
    assert RUN_FLUTTER_TESTS_TOOL["type"] == "function"
    assert RUN_FLUTTER_TESTS_TOOL["name"] == "run_flutter_tests"
    assert RUN_FLUTTER_TESTS_TOOL["strict"] is True
    assert RUN_FLUTTER_TESTS_TOOL["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_run_dart_analyze_tool_schema_has_no_arguments() -> None:
    assert RUN_DART_ANALYZE_TOOL["type"] == "function"
    assert RUN_DART_ANALYZE_TOOL["name"] == "run_dart_analyze"
    assert RUN_DART_ANALYZE_TOOL["strict"] is True
    assert RUN_DART_ANALYZE_TOOL["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_run_dart_format_tool_schema_has_no_arguments() -> None:
    assert RUN_DART_FORMAT_TOOL["type"] == "function"
    assert RUN_DART_FORMAT_TOOL["name"] == "run_dart_format"
    assert RUN_DART_FORMAT_TOOL["strict"] is True
    assert RUN_DART_FORMAT_TOOL["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_apply_patch_tool_schema_requires_a_structured_patch_request() -> None:
    assert APPLY_PATCH_TOOL["type"] == "function"
    assert APPLY_PATCH_TOOL["name"] == "apply_patch"
    assert APPLY_PATCH_TOOL["strict"] is True
    assert APPLY_PATCH_TOOL["parameters"]["required"] == ["path", "old_text", "new_text"]


def test_tool_permissions_describe_the_current_read_only_tools() -> None:
    assert set(ToolPermission) == {
        ToolPermission.READ,
        ToolPermission.WRITE,
        ToolPermission.EXTERNAL,
        ToolPermission.EXECUTE,
    }
    assert get_tool_permissions(READ_FILE_TOOL["name"]) == frozenset({ToolPermission.READ})
    assert get_tool_permissions(LIST_FILES_TOOL["name"]) == frozenset({ToolPermission.READ})
    assert get_tool_permissions(SEARCH_CODE_TOOL["name"]) == frozenset({ToolPermission.READ})
    assert get_tool_permissions(RUN_FLUTTER_TESTS_TOOL["name"]) == frozenset()
    assert get_tool_permissions(RUN_DART_ANALYZE_TOOL["name"]) == frozenset(
        {ToolPermission.EXECUTE}
    )
    assert get_tool_permissions(RUN_DART_FORMAT_TOOL["name"]) == frozenset()
    assert get_tool_permissions(APPLY_PATCH_TOOL["name"]) == frozenset({ToolPermission.WRITE})
    assert get_tool_permissions("unknown_tool") == frozenset()


def test_permission_policy_defaults_to_read_only_and_honors_cli_values() -> None:
    assert main.permission_policy_from_cli_values(None) == main.DEFAULT_PERMISSION_POLICY
    assert main.DEFAULT_PERMISSION_POLICY.allows(READ_FILE_TOOL["name"])
    assert (
        main.format_permission_policy(main.DEFAULT_PERMISSION_POLICY)
        == "Allowed permissions: execute, read, write"
    )

    explicit_policy = main.permission_policy_from_cli_values(["write", "external"])

    assert explicit_policy.allowed_permissions == frozenset(
        {ToolPermission.WRITE, ToolPermission.EXTERNAL}
    )
    assert explicit_policy.allows(READ_FILE_TOOL["name"]) is False


def test_agent_run_state_starts_with_isolated_empty_workflow_state() -> None:
    first_state = main.AgentRunState()
    second_state = main.AgentRunState()

    first_state.changed_file_paths.add("lib/example.dart")
    first_state.read_file_paths.add("lib/example.dart")

    assert first_state.verification_dirty is False
    assert first_state.repair_attempts == 0
    assert first_state.analyzer_baseline_output is None
    assert first_state.analyzer_baseline_issue_count is None
    assert first_state.verification_failed is False
    assert second_state.changed_file_paths == set()
    assert second_state.read_file_paths == set()


def test_completion_check_reports_incomplete_agent_run_state() -> None:
    assert main.check_agent_completion(
        main.AgentRunState(changed_file_paths=set())
    ) == main.AgentCompletionCheck(
        passed=True,
        failures=(),
    )


def test_completion_rejection_message_instructs_the_model_to_continue() -> None:
    assert main.completion_rejection_message(
        main.AgentCompletionCheck(
            passed=False,
            failures=(
                "A file was changed but has not been verified. Run the appropriate "
                "verification tool.",
                "The most recent verification failed. Inspect its output before making a "
                "grounded repair.",
            ),
        )
    ) == {
        "role": "system",
        "content": (
            "Completion rejected by the harness:\n"
            "A file was changed but has not been verified. Run the appropriate "
            "verification tool.\n"
            "The most recent verification failed. Inspect its output before making a "
            "grounded repair.\n"
            "Continue working on the task."
        ),
    }


def test_incomplete_run_continues_with_a_harness_completion_rejection(capsys) -> None:
    run_state = main.AgentRunState(verification_dirty=True)
    initial_response = type("Response", (), {"id": "response_initial", "output": []})()
    completed_response = type("Response", (), {"id": "response_completed", "output": []})()
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return initial_response

            run_state.verification_dirty = False
            return completed_response

    client = type("Client", (), {"responses": FakeResponses()})()

    assert (
        main.request_analysis_with_tools(
            "",
            "Complete the task.",
            client,
            run_state=run_state,
        )
        is completed_response
    )
    assert calls[1]["input"] == [
        {
            "role": "system",
            "content": (
                "Completion rejected by the harness:\n"
                "A file was changed but has not been verified. Run the appropriate "
                "verification tool.\n"
                "Continue working on the task."
            ),
        }
    ]
    assert calls[1]["previous_response_id"] == "response_initial"
    assert capsys.readouterr().out == (
        "Completion rejected by the harness:\n"
        "A file was changed but has not been verified. Run the appropriate verification tool.\n"
        "Continue working on the task.\n"
    )
    assert main.check_agent_completion(
        main.AgentRunState(
            verification_dirty=True,
            verification_failed=True,
            repair_attempts=main.MAX_REPAIR_ATTEMPTS,
        )
    ) == main.AgentCompletionCheck(
        passed=False,
        failures=(
            "A file was changed but has not been verified. Run the appropriate verification tool.",
            "The most recent verification failed. Inspect its output before making a "
            "grounded repair.",
            "The repair limit of 3 attempts was reached. Do not make another repair; report "
            "the unresolved verification failure.",
        ),
    )


def test_root_path_requires_an_existing_directory(tmp_path) -> None:
    assert main.root_path(str(tmp_path)) == tmp_path.resolve()

    with pytest.raises(argparse.ArgumentTypeError, match="not a directory"):
        main.root_path(str(tmp_path / "missing"))


def test_file_access_instructions_identify_the_project_root(tmp_path) -> None:
    instructions = main.instructions_for_file_access_policy(file_access_policy_for(tmp_path))

    assert str(tmp_path.resolve()) in instructions
    assert "relative to this root" in instructions


def test_permission_policy_rejects_a_tool_without_its_required_permission() -> None:
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"file_path": "widget.dart"}',
            "call_id": "call_123",
        },
    )()
    response = type("Response", (), {"output": [tool_call]})()
    no_permissions = main.PermissionPolicy(frozenset())

    assert main.execute_file_tool_calls(response, no_permissions) == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": (
                "Could not execute file tool: Tool 'read_file' is not allowed. "
                "Required permissions: read."
            ),
        }
    ]


def test_execute_permission_exposes_the_execute_tools() -> None:
    execute_policy = main.PermissionPolicy(frozenset({ToolPermission.EXECUTE}))

    assert main.tools_allowed_by(execute_policy) == [RUN_DART_ANALYZE_TOOL]


def test_write_permission_exposes_the_write_tools() -> None:
    write_policy = main.PermissionPolicy(frozenset({ToolPermission.WRITE}))

    assert main.tools_allowed_by(write_policy) == [APPLY_PATCH_TOOL]


def test_apply_patch_replaces_one_unique_match(tmp_path, capsys) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "example.dart").write_text("before", encoding="utf-8")
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "apply_patch",
            "arguments": (
                '{"path": "lib/example.dart", "old_text": "before", "new_text": "after"}'
            ),
            "call_id": "call_patch",
        },
    )()
    response = type("Response", (), {"output": [tool_call]})()
    write_policy = main.PermissionPolicy(frozenset({ToolPermission.READ, ToolPermission.WRITE}))

    assert main.execute_file_tool_calls(
        response,
        write_policy,
        file_access_policy_for(tmp_path),
        main.AgentRunState(
            read_file_paths={"lib/example.dart"},
            analyzer_baseline_output="baseline",
        ),
    ) == [
        {
            "type": "function_call_output",
            "call_id": "call_patch",
            "output": (
                '{"success": true, "path": "lib/example.dart", "replacements": 1, '
                '"message": "Patch applied successfully."}'
            ),
        }
    ]
    assert capsys.readouterr().out == (
        "Tool call: apply_patch(lib/example.dart)\n"
        'Tool result: {"success": true, "path": "lib/example.dart", "replacements": 1, '
        '"message": "Patch applied successfully."}\n'
    )
    assert (tmp_path / "lib" / "example.dart").read_text(encoding="utf-8") == "after"


def test_apply_patch_rejects_ambiguous_old_text_matches(tmp_path, capsys) -> None:
    (tmp_path / "lib").mkdir()
    source_file = tmp_path / "lib" / "example.dart"
    source_file.write_text("before\nbefore\n", encoding="utf-8")
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "apply_patch",
            "arguments": (
                '{"path": "lib/example.dart", "old_text": "before", "new_text": "after"}'
            ),
            "call_id": "call_patch",
        },
    )()
    response = type("Response", (), {"output": [tool_call]})()
    write_policy = main.PermissionPolicy(frozenset({ToolPermission.READ, ToolPermission.WRITE}))

    assert main.execute_file_tool_calls(
        response,
        write_policy,
        file_access_policy_for(tmp_path),
        main.AgentRunState(
            read_file_paths={"lib/example.dart"},
            analyzer_baseline_output="baseline",
        ),
    ) == [
        {
            "type": "function_call_output",
            "call_id": "call_patch",
            "output": (
                '{"success": false, "path": "lib/example.dart", "replacements": 0, '
                '"message": "old_text matched more than once in the requested file; '
                'found 2 matches."}'
            ),
        }
    ]
    assert source_file.read_text(encoding="utf-8") == "before\nbefore\n"
    assert capsys.readouterr().out == (
        "Tool call: apply_patch(lib/example.dart)\n"
        'Tool result: {"success": false, "path": "lib/example.dart", "replacements": 0, '
        '"message": "old_text matched more than once in the requested file; '
        'found 2 matches."}\n'
    )


def test_patch_request_represents_one_text_replacement() -> None:
    request = PatchRequest(path="lib/example.dart", old_text="before", new_text="after")

    assert request.path == "lib/example.dart"


def test_apply_patch_returns_not_found_outcome(tmp_path) -> None:
    (tmp_path / "example.dart").write_text("before", encoding="utf-8")

    assert apply_patch(
        PatchRequest(path="example.dart", old_text="missing", new_text="after"),
        file_access_policy_for(tmp_path),
        {"example.dart"},
    ) == PatchResult(
        success=False,
        path="example.dart",
        replacements=0,
        message="old_text was not found in the requested file.",
    )


def test_apply_patch_returns_the_replacement_count(tmp_path) -> None:
    (tmp_path / "example.dart").write_text("before", encoding="utf-8")

    assert apply_patch(
        PatchRequest(path="example.dart", old_text="before", new_text="after"),
        file_access_policy_for(tmp_path),
        {"example.dart"},
    ) == PatchResult(
        success=True,
        path="example.dart",
        replacements=1,
        message="Patch applied successfully.",
    )


def test_apply_patch_requires_the_file_to_be_read_first(tmp_path) -> None:
    source_file = tmp_path / "example.dart"
    source_file.write_text("before", encoding="utf-8")

    assert apply_patch(
        PatchRequest(path="example.dart", old_text="before", new_text="after"),
        file_access_policy_for(tmp_path),
        set(),
    ) == PatchResult(
        success=False,
        path="example.dart",
        replacements=0,
        message="The file must be read with read_file before applying a patch.",
    )
    assert source_file.read_text(encoding="utf-8") == "before"


def test_read_file_allows_a_later_patch_for_the_same_file(tmp_path) -> None:
    source_file = tmp_path / "example.dart"
    source_file.write_text("before", encoding="utf-8")
    read_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"file_path": "example.dart"}',
            "call_id": "call_read",
        },
    )()
    patch_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "apply_patch",
            "arguments": ('{"path": "example.dart", "old_text": "before", "new_text": "after"}'),
            "call_id": "call_patch",
        },
    )()
    response = type("Response", (), {"output": [read_call, patch_call]})()
    read_write_policy = main.PermissionPolicy(
        frozenset({ToolPermission.READ, ToolPermission.WRITE})
    )

    run_state = main.AgentRunState(analyzer_baseline_output="baseline")
    tool_outputs = main.execute_file_tool_calls(
        response,
        read_write_policy,
        file_access_policy_for(tmp_path),
        run_state,
    )

    assert tool_outputs[1] == {
        "type": "function_call_output",
        "call_id": "call_patch",
        "output": (
            '{"success": true, "path": "example.dart", "replacements": 1, '
            '"message": "Patch applied successfully."}'
        ),
    }
    assert source_file.read_text(encoding="utf-8") == "after"
    assert run_state.changed_file_paths == {"example.dart"}
    assert run_state.verification_dirty is True


def test_apply_patch_requires_a_dart_analyze_baseline(tmp_path) -> None:
    source_file = tmp_path / "example.dart"
    source_file.write_text("before", encoding="utf-8")
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "apply_patch",
            "arguments": ('{"path": "example.dart", "old_text": "before", "new_text": "after"}'),
            "call_id": "call_patch",
        },
    )()
    response = type("Response", (), {"output": [tool_call]})()
    read_write_policy = main.PermissionPolicy(
        frozenset({ToolPermission.READ, ToolPermission.WRITE})
    )

    assert main.execute_file_tool_calls(
        response,
        read_write_policy,
        file_access_policy_for(tmp_path),
        main.AgentRunState(read_file_paths={"example.dart"}),
    )[0]["output"] == (
        '{"success": false, "path": "example.dart", "replacements": 0, '
        '"message": "Run dart analyze before applying a patch to establish a baseline."}'
    )
    assert source_file.read_text(encoding="utf-8") == "before"


def test_analyzer_output_diff_identifies_new_output() -> None:
    assert main._analyzer_output_diff(
        "stdout:\nexisting issue\nstderr:\n",
        "stdout:\nexisting issue\nnew issue\nstderr:\n",
    ) == (
        "--- baseline\n+++ current\n@@ -1,3 +1,4 @@\n"
        " stdout:\n existing issue\n+new issue\n stderr:"
    )


def test_analyzer_issue_count_reads_dart_analyzer_summaries() -> None:
    assert main._analyzer_issue_count("No issues found!") == 0
    assert main._analyzer_issue_count("Analyzed 12 files, 1 issue found.") == 1
    assert main._analyzer_issue_count("3 issues found.") == 3
    assert main._analyzer_issue_count("analyzer failed before producing a summary") is None


def test_repair_attempts_are_limited_after_a_verification_failure(tmp_path) -> None:
    source_file = tmp_path / "example.dart"
    source_file.write_text("before", encoding="utf-8")
    read_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"file_path": "example.dart"}',
            "call_id": "call_read",
        },
    )()
    patch_calls = [
        type(
            "ToolCall",
            (),
            {
                "type": "function_call",
                "name": "apply_patch",
                "arguments": (
                    '{"path": "example.dart", "old_text": "before", "new_text": "after"}'
                ),
                "call_id": f"call_patch_{attempt}",
            },
        )()
        for attempt in range(4)
    ]
    response = type("Response", (), {"output": [read_call, *patch_calls]})()
    permission_policy = main.PermissionPolicy(
        frozenset({ToolPermission.READ, ToolPermission.WRITE})
    )
    run_state = main.AgentRunState(
        analyzer_baseline_output="baseline",
        analyzer_issues_found=True,
    )

    tool_outputs = main.execute_file_tool_calls(
        response,
        permission_policy,
        file_access_policy_for(tmp_path),
        run_state,
    )

    assert run_state.repair_attempts == main.MAX_REPAIR_ATTEMPTS
    assert tool_outputs[-1]["output"] == (
        '{"success": false, "path": "example.dart", "replacements": 0, '
        '"message": "Repair attempt limit reached (3 attempts)."}'
    )


def test_run_flutter_tests_uses_a_fixed_flutter_command(monkeypatch, tmp_path) -> None:
    (tmp_path / "pubspec.yaml").write_text("name: test_project", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type(
            "CompletedProcess", (), {"returncode": 0, "stdout": "All passed", "stderr": ""}
        )()

    monkeypatch.setattr(tool_schemas.subprocess, "run", fake_run)

    assert run_flutter_tests(tmp_path) == FlutterTestResult(
        exit_code=0,
        stdout="All passed",
        stderr="",
        success=True,
        output_truncated=False,
    )
    assert calls == [
        (
            ["flutter", "test"],
            {
                "cwd": tmp_path.resolve(),
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 300,
            },
        )
    ]


def test_run_dart_analyze_uses_a_fixed_dart_command(monkeypatch, tmp_path) -> None:
    (tmp_path / "pubspec.yaml").write_text("name: test_project", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type(
            "CompletedProcess", (), {"returncode": 1, "stdout": "Issue found", "stderr": ""}
        )()

    monkeypatch.setattr(tool_schemas.subprocess, "run", fake_run)

    assert run_dart_analyze(tmp_path) == DartAnalyzeResult(
        exit_code=1,
        stdout="Issue found",
        stderr="",
        timed_out=False,
        success=False,
        output_truncated=False,
        issues=(),
    )
    assert calls == [
        (
            ["dart", "analyze"],
            {
                "cwd": tmp_path.resolve(),
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 300,
            },
        )
    ]


def test_run_dart_analyze_returns_a_timeout_result(monkeypatch, tmp_path) -> None:
    (tmp_path / "pubspec.yaml").write_text("name: test_project", encoding="utf-8")

    def fake_run(command, **kwargs):
        raise tool_schemas.subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output=b"partial output",
            stderr=b"partial error",
        )

    monkeypatch.setattr(tool_schemas.subprocess, "run", fake_run)

    assert run_dart_analyze(tmp_path) == DartAnalyzeResult(
        exit_code=None,
        stdout="partial output",
        stderr="partial error",
        timed_out=True,
        success=False,
        output_truncated=False,
        issues=(),
    )


def test_run_flutter_tests_truncates_large_output(monkeypatch, tmp_path) -> None:
    (tmp_path / "pubspec.yaml").write_text("name: test_project", encoding="utf-8")
    stdout = "a" * 10_000 + "removed" + "b" * 10_000
    stderr = "c" * 10_000 + "removed" + "d" * 10_000

    def fake_run(command, **kwargs):
        return type(
            "CompletedProcess",
            (),
            {"returncode": 0, "stdout": stdout, "stderr": stderr},
        )()

    monkeypatch.setattr(tool_schemas.subprocess, "run", fake_run)

    result = run_flutter_tests(tmp_path)

    assert result.stdout == "a" * 10_000 + "b" * 10_000
    assert result.stderr == "c" * 10_000 + "d" * 10_000
    assert result.output_truncated is True


def test_run_dart_analyze_truncates_large_output(monkeypatch, tmp_path) -> None:
    (tmp_path / "pubspec.yaml").write_text("name: test_project", encoding="utf-8")
    stdout = "a" * 10_000 + "removed" + "b" * 10_000
    stderr = "c" * 10_000 + "removed" + "d" * 10_000

    def fake_run(command, **kwargs):
        return type(
            "CompletedProcess",
            (),
            {"returncode": 0, "stdout": stdout, "stderr": stderr},
        )()

    monkeypatch.setattr(tool_schemas.subprocess, "run", fake_run)

    result = run_dart_analyze(tmp_path)

    assert result.stdout == "a" * 10_000 + "b" * 10_000
    assert result.stderr == "c" * 10_000 + "d" * 10_000
    assert result.output_truncated is True


def test_run_dart_analyze_parses_structured_issues(monkeypatch, tmp_path) -> None:
    (tmp_path / "pubspec.yaml").write_text("name: test_project", encoding="utf-8")

    def fake_run(command, **kwargs):
        return type(
            "CompletedProcess",
            (),
            {
                "returncode": 1,
                "stdout": (
                    "error - lib/example.dart:12:4 - Missing semicolon - expected_token\n"
                    "warning - Unused import - lib/unused.dart:3:1 - unused_import\n"
                ),
                "stderr": "",
            },
        )()

    monkeypatch.setattr(tool_schemas.subprocess, "run", fake_run)

    assert run_dart_analyze(tmp_path).issues == (
        DartAnalyzeIssue(
            path="lib/example.dart",
            line_number=12,
            severity="error",
            message="Missing semicolon",
        ),
        DartAnalyzeIssue(
            path="lib/unused.dart",
            line_number=3,
            severity="warning",
            message="Unused import",
        ),
    )


def test_run_dart_format_uses_a_fixed_dart_command(monkeypatch, tmp_path) -> None:
    (tmp_path / "pubspec.yaml").write_text("name: test_project", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type(
            "CompletedProcess", (), {"returncode": 0, "stdout": "Formatted", "stderr": ""}
        )()

    monkeypatch.setattr(tool_schemas.subprocess, "run", fake_run)

    assert run_dart_format(tmp_path) == DartFormatResult(
        exit_code=0,
        stdout="Formatted",
        stderr="",
    )
    assert calls == [
        (
            ["dart", "format", "."],
            {
                "cwd": tmp_path.resolve(),
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 300,
            },
        )
    ]


def test_search_result_represents_a_code_match() -> None:
    result = SearchResult(
        path="widgets/button.dart",
        line_number=42,
        text="class Button extends StatelessWidget {",
    )

    assert result.path == "widgets/button.dart"
    assert result.line_number == 42
    assert result.text == "class Button extends StatelessWidget {"


def test_search_code_result_contains_matches_and_metadata() -> None:
    match = SearchResult(path="button.dart", line_number=1, text="class Button {}")
    result = SearchCodeResult(matches=(match,), total_matches=4, truncated=False)

    assert result.matches == (match,)
    assert result.total_matches == 4
    assert result.truncated is False


def test_file_state_contains_file_metadata_and_numbered_content() -> None:
    state = FileState(
        file_path="widgets/button.dart",
        line_count=2,
        numbered_content="1: class Button {}\n2: ",
    )

    assert state.file_path == "widgets/button.dart"
    assert state.line_count == 2
    assert state.numbered_content == "1: class Button {}\n2: "


def test_file_access_policy_uses_one_project_root(tmp_path) -> None:
    policy = file_access_policy_for(tmp_path)

    assert policy.root_path == tmp_path
    assert policy.max_file_size_bytes == 10 * 1024 * 1024


def test_read_source_line_returns_the_requested_one_based_line() -> None:
    assert read_source_line("first\nsecond\nthird", 2) == "2: second"


@pytest.mark.parametrize("line_number", [0, 4])
def test_read_source_line_rejects_out_of_range_line_numbers(line_number: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        read_source_line("first\nsecond\nthird", line_number)


def test_read_source_line_rejects_a_non_integer_line_number() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        read_source_line("first", True)


def test_read_file_returns_text_from_the_approved_directory(tmp_path) -> None:
    source_file = tmp_path / "example.dart"
    source_file.write_text("print('hello')\n", encoding="utf-8")

    assert read_file("example.dart", file_access_policy_for(tmp_path)) == "print('hello')\n"


def test_read_file_rejects_paths_outside_the_approved_directory(tmp_path) -> None:
    with pytest.raises(PermissionError, match="Access is not available outside the project root"):
        read_file("../outside.txt", file_access_policy_for(tmp_path))


def test_read_file_allows_non_dart_text_files(tmp_path) -> None:
    (tmp_path / "settings.json").write_text("{}", encoding="utf-8")

    assert read_file("settings.json", file_access_policy_for(tmp_path)) == "{}"


def test_read_file_rejects_files_larger_than_ten_mebibytes(tmp_path) -> None:
    source_file = tmp_path / "large_source.dart"
    source_file.touch()
    with source_file.open("wb") as file:
        file.truncate(10 * 1024 * 1024 + 1)

    with pytest.raises(ValueError, match="too large"):
        read_file("large_source.dart", file_access_policy_for(tmp_path))


def test_file_access_policy_can_customize_file_size_and_protected_names(tmp_path) -> None:
    (tmp_path / "large.dart").write_text("12345", encoding="utf-8")
    (tmp_path / "internal.txt").write_text("secret", encoding="utf-8")
    custom_policy = FileAccessPolicy(
        root_path=tmp_path,
        max_file_size_bytes=4,
        protected_file_names=frozenset({"internal.txt"}),
    )

    with pytest.raises(ValueError, match="too large"):
        read_file("large.dart", custom_policy)
    with pytest.raises(ValueError, match="protected"):
        read_file("internal.txt", custom_policy)
    assert list_files("", custom_policy) == "large.dart"


@pytest.mark.parametrize(
    "file_name",
    [".DS_Store", ".env", ".env.local", "private.pem", "id_rsa"],
)
def test_read_file_rejects_common_protected_files(tmp_path, file_name: str) -> None:
    (tmp_path / file_name).write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="protected"):
        read_file(file_name, file_access_policy_for(tmp_path))


def test_file_tools_require_paths_relative_to_the_project_root(tmp_path) -> None:
    source_file = tmp_path / "source.dart"
    source_file.write_text("match", encoding="utf-8")

    with pytest.raises(PermissionError, match="must be relative"):
        read_file(str(source_file), file_access_policy_for(tmp_path))


def test_search_code_returns_a_result_for_each_matching_line(tmp_path) -> None:
    (tmp_path / "example.dart").write_text(
        "first line\nfind this\n",
        encoding="utf-8",
    )
    widgets_directory = tmp_path / "widgets"
    widgets_directory.mkdir()
    (widgets_directory / "button.dart").write_text("find this too\n", encoding="utf-8")
    (tmp_path / "included.swift").write_text("find this\n", encoding="utf-8")
    (tmp_path / "excluded.txt").write_text("find this\n", encoding="utf-8")

    assert search_code("find this", file_access_policy_for(tmp_path)) == SearchCodeResult(
        matches=(
            SearchResult(path="example.dart", line_number=2, text="find this"),
            SearchResult(path="included.swift", line_number=1, text="find this"),
            SearchResult(path="widgets/button.dart", line_number=1, text="find this too"),
        ),
        total_matches=3,
        truncated=False,
    )


def test_search_code_rejects_an_empty_query(tmp_path) -> None:
    (tmp_path / "example.dart").write_text("anything", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        search_code("", file_access_policy_for(tmp_path))


def test_search_code_matches_case_insensitively(tmp_path) -> None:
    (tmp_path / "example.swift").write_text("func ImportantFunction() {}\n", encoding="utf-8")

    assert search_code("importantfunction", file_access_policy_for(tmp_path)) == SearchCodeResult(
        matches=(
            SearchResult(
                path="example.swift",
                line_number=1,
                text="func ImportantFunction() {}",
            ),
        ),
        total_matches=1,
        truncated=False,
    )


def test_file_tools_skip_ignored_directories(tmp_path) -> None:
    (tmp_path / "source.dart").write_text("target", encoding="utf-8")
    ignored_directory = tmp_path / ".dart_tool"
    ignored_directory.mkdir()
    (ignored_directory / "generated.dart").write_text("target", encoding="utf-8")

    assert list_files("", file_access_policy_for(tmp_path)) == "source.dart"
    assert search_code("target", file_access_policy_for(tmp_path)) == SearchCodeResult(
        matches=(SearchResult(path="source.dart", line_number=1, text="target"),),
        total_matches=1,
        truncated=False,
    )
    with pytest.raises(ValueError, match="ignored directory"):
        read_file(".dart_tool/generated.dart", file_access_policy_for(tmp_path))


def test_search_code_stops_after_thirty_matches(tmp_path) -> None:
    (tmp_path / "many_matches.dart").write_text(
        "\n".join("match" for _ in range(31)),
        encoding="utf-8",
    )

    result = search_code("match", file_access_policy_for(tmp_path))

    assert len(result.matches) == SEARCH_CODE_MAX_RESULTS
    assert result.matches[0].line_number == 1
    assert result.matches[-1].line_number == SEARCH_CODE_MAX_RESULTS
    assert result.total_matches == 31
    assert result.truncated is True


def test_list_files_returns_sorted_paths_relative_to_the_approved_directory(tmp_path) -> None:
    (tmp_path / "zebra.py").write_text("", encoding="utf-8")
    subdirectory = tmp_path / "widgets"
    subdirectory.mkdir()
    (subdirectory / "button.py").write_text("", encoding="utf-8")
    (subdirectory / "input.py").write_text("", encoding="utf-8")

    assert (
        list_files("widgets", file_access_policy_for(tmp_path))
        == "widgets/button.py\nwidgets/input.py"
    )
    assert list_files("", file_access_policy_for(tmp_path)) == "widgets/\nzebra.py"


def test_list_files_hides_protected_files_and_directories(tmp_path) -> None:
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    (tmp_path / "private.pem").write_text("secret", encoding="utf-8")
    (tmp_path / "source.py").write_text("print('safe')", encoding="utf-8")
    (tmp_path / ".git").mkdir()

    assert list_files("", file_access_policy_for(tmp_path)) == "source.py"


def test_list_files_rejects_paths_outside_the_approved_directory(tmp_path) -> None:
    with pytest.raises(PermissionError, match="Access is not available outside the project root"):
        list_files("../", file_access_policy_for(tmp_path))


def test_main_uses_the_schemas_module_models() -> None:
    assert main.AnalysisResult is AnalysisResult
    assert main.Finding is Finding


def test_analysis_result_accepts_valid_findings() -> None:
    result = main.AnalysisResult(
        summary="One issue found.",
        findings=[
            main.Finding(
                start_line=2,
                end_line=3,
                problem="The input is not validated.",
                solution="Validate the input before using it.",
                severity="medium",
                start_character=10,
                end_character=20,
            )
        ],
    )

    assert result.findings[0].severity == "medium"


def test_finding_rejects_unknown_severity_and_backward_ranges() -> None:
    with pytest.raises(ValidationError):
        main.Finding(
            start_line=3,
            end_line=2,
            problem="A problem.",
            solution="A solution.",
            severity="critical",
        )


def test_analysis_result_rejects_a_response_with_missing_finding_fields() -> None:
    incomplete_response = {
        "summary": "One issue found.",
        "findings": [
            {
                "start_line": 2,
                "end_line": 2,
                "problem": "The input is not validated.",
                "severity": "medium",
            }
        ],
    }

    with pytest.raises(ValidationError) as error:
        main.AnalysisResult.model_validate(incomplete_response)

    assert "solution" in str(error.value)


def test_analysis_result_rejects_a_response_with_wrong_field_types() -> None:
    wrong_type_response = {
        "summary": "One issue found.",
        "findings": [
            {
                "start_line": "second",
                "end_line": 2,
                "problem": "The input is not validated.",
                "solution": "Validate the input before using it.",
                "severity": "medium",
            }
        ],
    }

    with pytest.raises(ValidationError) as error:
        main.AnalysisResult.model_validate(wrong_type_response)

    assert "start_line" in str(error.value)


def test_analysis_result_rejects_a_response_with_an_unexpected_severity() -> None:
    unexpected_enum_response = {
        "summary": "One issue found.",
        "findings": [
            {
                "start_line": 2,
                "end_line": 2,
                "problem": "The input is not validated.",
                "solution": "Validate the input before using it.",
                "severity": "critical",
            }
        ],
    }

    with pytest.raises(ValidationError) as error:
        main.AnalysisResult.model_validate(unexpected_enum_response)

    assert "severity" in str(error.value)


def test_analysis_result_rejects_an_empty_response_object() -> None:
    with pytest.raises(ValidationError) as error:
        main.AnalysisResult.model_validate({})

    error_message = str(error.value)
    assert "summary" in error_message
    assert "findings" in error_message


def test_analysis_result_rejects_an_empty_response_string() -> None:
    with pytest.raises(ValidationError):
        main.AnalysisResult.model_validate("")


def test_read_text_file_returns_file_contents(tmp_path) -> None:
    source_file = tmp_path / "example.py"
    source_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    assert main.read_text_file(str(source_file)) == "def add(a, b):\n    return a + b\n"


def test_read_text_file_with_retry_prompts_after_an_invalid_path(monkeypatch, capsys) -> None:
    attempted_paths = []

    def fake_read_text_file(file_path: str) -> str:
        attempted_paths.append(file_path)
        if file_path == "missing.py":
            raise FileNotFoundError(2, "No such file or directory")
        return "file contents"

    monkeypatch.setattr(main, "read_text_file", fake_read_text_file)
    monkeypatch.setattr("builtins.input", lambda prompt: "valid.py")

    assert main.read_text_file_with_retry("missing.py") == "file contents"
    assert attempted_paths == ["missing.py", "valid.py"]
    assert "Could not read missing.py: No such file or directory" in capsys.readouterr().out


def test_analyze_text_uses_luna_model() -> None:
    calls = []
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    response = type("Response", (), {"output_parsed": parsed_result})()

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return response

    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("first line\nsecond line", "Be concise.", client) is response
    assert calls == [
        {
            "model": "gpt-5.6-luna",
            "instructions": main.AGENT_INSTRUCTIONS,
            "input": "1: first line\n2: second line",
            "text_format": main.AnalysisResult,
            "tools": [
                main.READ_FILE_TOOL,
                main.LIST_FILES_TOOL,
                main.SEARCH_CODE_TOOL,
                main.RUN_DART_ANALYZE_TOOL,
                main.APPLY_PATCH_TOOL,
            ],
            "tool_choice": "auto",
        }
    ]


def test_analyze_text_executes_a_requested_file_read_and_returns_its_output(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"file_path": "widget.dart"}',
            "call_id": "call_123",
        },
    )()
    first_response = type(
        "Response",
        (),
        {"id": "response_1", "output_parsed": None, "output": [tool_call]},
    )()
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    final_response = type(
        "Response",
        (),
        {"output_parsed": parsed_result, "output": []},
    )()
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return [first_response, final_response][len(calls) - 1]

    monkeypatch.setattr(
        main, "read_file", lambda file_path, file_access_policy: "class Widget {}\n"
    )
    client = type("Client", (), {"responses": FakeResponses()})()
    file_access_policy = file_access_policy_for(tmp_path)

    assert (
        main.analyze_text(
            "entry point", "Analyze related code.", client, file_access_policy=file_access_policy
        )
        is final_response
    )
    assert calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": (
                '{"file_path": "widget.dart", "line_count": 1, '
                '"numbered_content": "1: class Widget {}"}'
            ),
        }
    ]
    assert calls[1]["previous_response_id"] == "response_1"
    assert capsys.readouterr().out == (
        "Tool call: read_file(widget.dart)\nTool result: read widget.dart (1 lines)\n"
    )


def test_analyze_text_executes_a_requested_file_list_and_returns_its_output(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "list_files",
            "arguments": '{"directory_path": "widgets"}',
            "call_id": "call_456",
        },
    )()
    first_response = type(
        "Response",
        (),
        {"id": "response_1", "output_parsed": None, "output": [tool_call]},
    )()
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    final_response = type(
        "Response",
        (),
        {"output_parsed": parsed_result, "output": []},
    )()
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return [first_response, final_response][len(calls) - 1]

    monkeypatch.setattr(
        main, "list_files", lambda directory_path, file_access_policy: "widgets/button.py"
    )
    client = type("Client", (), {"responses": FakeResponses()})()
    file_access_policy = file_access_policy_for(tmp_path)

    assert (
        main.analyze_text(
            "entry point", "Analyze related code.", client, file_access_policy=file_access_policy
        )
        is final_response
    )
    assert calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_456",
            "output": "widgets/button.py",
        }
    ]
    assert capsys.readouterr().out == (
        "Tool call: list_files(widgets)\nTool result: listed 1 files\n"
    )


def test_analyze_text_executes_a_requested_code_search_and_returns_its_output(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "search_code",
            "arguments": '{"query": "Button"}',
            "call_id": "call_789",
        },
    )()
    first_response = type(
        "Response",
        (),
        {"id": "response_1", "output_parsed": None, "output": [tool_call]},
    )()
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    final_response = type(
        "Response",
        (),
        {"output_parsed": parsed_result, "output": []},
    )()
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return [first_response, final_response][len(calls) - 1]

    monkeypatch.setattr(
        main,
        "search_code",
        lambda query, file_access_policy: SearchCodeResult(
            matches=(SearchResult(path="button.dart", line_number=7, text="class Button {}"),),
            total_matches=1,
            truncated=False,
        ),
    )
    client = type("Client", (), {"responses": FakeResponses()})()
    file_access_policy = file_access_policy_for(tmp_path)

    assert (
        main.analyze_text(
            "entry point", "Analyze related code.", client, file_access_policy=file_access_policy
        )
        is final_response
    )
    assert calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_789",
            "output": (
                '{"matches": [{"path": "button.dart", "line_number": 7, '
                '"text": "class Button {}"}], "total_matches": 1, "truncated": false}'
            ),
        }
    ]
    terminal_output = capsys.readouterr().out
    assert "Tool call: search_code(Button)" in terminal_output
    assert "Tool result: found 1 matches (returned 1)" in terminal_output


def test_file_tool_errors_are_returned_to_the_model(monkeypatch, tmp_path) -> None:
    tool_call = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"file_path": "../.env"}',
            "call_id": "call_123",
        },
    )()
    response = type("Response", (), {"output": [tool_call]})()
    monkeypatch.setattr(
        main,
        "read_file",
        lambda file_path, file_access_policy: (_ for _ in ()).throw(
            ValueError("path is not allowed")
        ),
    )

    assert main.execute_file_tool_calls(
        response, file_access_policy=file_access_policy_for(tmp_path)
    ) == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Could not execute file tool: path is not allowed",
        }
    ]


def test_number_source_lines_adds_one_based_line_numbers() -> None:
    assert main.number_source_lines("first line\nsecond line") == "1: first line\n2: second line"


def test_analyze_text_wraps_request_errors() -> None:
    class FailingResponses:
        def parse(self, **kwargs):
            raise RuntimeError("invalid API key")

    client = type("Client", (), {"responses": FailingResponses()})()

    with pytest.raises(main.AnalysisError, match="OpenAI request failed: invalid API key"):
        main.analyze_text("hello world", "Be concise.", client)


def test_analyze_text_rejects_missing_parsed_output() -> None:
    response = type("Response", (), {"output_parsed": None})()
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return response

    client = type("Client", (), {"responses": FakeResponses()})()

    with pytest.raises(main.AnalysisError, match="invalid analysis after 3 attempts"):
        main.analyze_text("hello world", "Be concise.", client)

    assert len(calls) == 3


def test_analyze_text_retries_missing_parsed_output_before_succeeding(capsys) -> None:
    empty_response = type("Response", (), {"output_parsed": None})()
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    valid_response = type("Response", (), {"output_parsed": parsed_result})()
    responses = [empty_response, valid_response]

    class FakeResponses:
        def parse(self, **kwargs):
            return responses.pop(0)

    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("hello world", "Be concise.", client) is valid_response
    assert capsys.readouterr().out == "Invalid analysis response. Retrying (1/2)...\n"


def test_analyze_text_retries_a_validation_error_before_succeeding(capsys) -> None:
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    valid_response = type("Response", (), {"output_parsed": parsed_result})()
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                main.AnalysisResult.model_validate({})
            return valid_response

    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("hello world", "Be concise.", client) is valid_response
    assert len(calls) == 2
    assert capsys.readouterr().out == "Invalid analysis response. Retrying (1/2)...\n"


def test_analyze_text_can_simulate_a_validation_error_once(capsys) -> None:
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    valid_response = type("Response", (), {"output_parsed": parsed_result})()
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return valid_response

    client = type("Client", (), {"responses": FakeResponses()})()

    assert (
        main.analyze_text(
            "hello world",
            "Be concise.",
            client,
            simulate_validation_error_once=True,
        )
        is valid_response
    )
    assert len(calls) == 1
    assert capsys.readouterr().out == "Invalid analysis response. Retrying (1/2)...\n"


def test_analyze_text_rejects_findings_outside_the_numbered_input() -> None:
    parsed_result = main.AnalysisResult(
        summary="One issue found.",
        findings=[
            main.Finding(
                start_line=3,
                end_line=3,
                problem="The input is not validated.",
                solution="Validate the input before using it.",
                severity="medium",
            )
        ],
    )
    response = type("Response", (), {"output_parsed": parsed_result})()

    class FakeResponses:
        def parse(self, **kwargs):
            return response

    client = type("Client", (), {"responses": FakeResponses()})()

    with pytest.raises(main.AnalysisError, match="input contains only 2 lines"):
        main.analyze_text("first line\nsecond line", "Be concise.", client)


def test_format_response_returns_pretty_json() -> None:
    response = type("Response", (), {"model_dump": lambda self: {"id": "resp_123"}})()

    assert main.format_response(response) == '{\n  "id": "resp_123"\n}'


def test_format_output_items_returns_pretty_json() -> None:
    item = type("OutputItem", (), {"model_dump": lambda self: {"type": "message"}})()
    response = type("Response", (), {"output": [item]})()

    assert main.format_output_items(response) == '[\n  {\n    "type": "message"\n  }\n]'


def test_format_analysis_result_returns_pretty_json() -> None:
    parsed_result = main.AnalysisResult(summary="No issues found.", findings=[])
    response = type("Response", (), {"output_parsed": parsed_result})()

    assert (
        main.format_analysis_result(response)
        == '{\n  "summary": "No issues found.",\n  "findings": []\n}'
    )


def test_format_analysis_result_object_returns_readable_report() -> None:
    result = main.AnalysisResult(
        summary="One issue found.",
        findings=[
            main.Finding(
                start_line=2,
                end_line=3,
                problem="The input is not validated.",
                solution="Validate the input before using it.",
                severity="medium",
                start_character=10,
                end_character=20,
            )
        ],
    )

    assert main.format_analysis_result_object(result) == (
        "Analysis Result\n"
        "Summary: One issue found.\n"
        "\n"
        "Findings:\n"
        "  1. [MEDIUM] lines 2-3, characters 10-20\n"
        "     Problem: The input is not validated.\n"
        "     Solution: Validate the input before using it."
    )


def test_format_token_usage_returns_input_and_output_counts() -> None:
    usage = type("Usage", (), {"input_tokens": 123, "output_tokens": 45})()
    response = type("Response", (), {"usage": usage})()

    assert (
        main.format_token_usage(response)
        == "Token usage:\n  Input tokens: 123\n  Output tokens: 45"
    )


def test_format_token_usage_handles_missing_usage() -> None:
    response = type("Response", (), {"usage": None})()

    assert main.format_token_usage(response) == "Token usage: unavailable"


def test_format_analysis_metrics_returns_usage_cost_and_elapsed_time() -> None:
    usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 50})()
    response = type("Response", (), {"usage": usage})()

    assert main.format_analysis_metrics(response, 1.234) == (
        "Analysis Metrics\n"
        "  Input tokens: 100\n"
        "  Output tokens: 50\n"
        "  Estimated token cost: $0.000080\n"
        "  Analysis time: 1.23 seconds"
    )


def test_format_analysis_metrics_handles_missing_usage() -> None:
    response = type("Response", (), {"usage": None})()

    assert main.format_analysis_metrics(response, 1.234) == (
        "Analysis Metrics\n"
        "  Token usage: unavailable\n"
        "  Estimated token cost: unavailable\n"
        "  Analysis time: 1.23 seconds"
    )


def test_build_run_metrics_uses_agent_run_state() -> None:
    run_state = main.AgentRunState(
        changed_file_paths={"lib/changed.dart"},
        repair_attempts=2,
        read_file_paths={"lib/read.dart", "lib/changed.dart"},
        model_calls=3,
        tool_calls=8,
        searched_file_paths={"lib/read.dart"},
        patch_attempts=4,
        input_tokens=120,
        output_tokens=45,
    )

    assert main.build_run_metrics(
        run_state,
        1.25,
        main.AgentCompletionCheck(passed=True, failures=()),
    ) == main.RunMetrics(
        model_calls=3,
        tool_calls=8,
        files_searched=1,
        files_read=2,
        files_changed=1,
        patch_attempts=4,
        repair_attempts=2,
        input_tokens=120,
        output_tokens=45,
        runtime_seconds=1.25,
        completion_passed=True,
    )


def test_format_changed_files_reports_sorted_paths_or_no_changes() -> None:
    assert main.format_changed_files({"lib/b.dart", "lib/a.dart"}) == (
        "Changed Files\n  - lib/a.dart\n  - lib/b.dart"
    )
    assert main.format_changed_files(set()) == "Changed Files\n  No files changed."


def test_temperature_accepts_values_from_zero_to_two() -> None:
    assert main.temperature("0.7") == 0.7

    with pytest.raises(argparse.ArgumentTypeError):
        main.temperature("2.1")


def test_max_output_tokens_accepts_values_from_one_to_model_limit() -> None:
    assert main.max_output_tokens("200") == 200

    with pytest.raises(argparse.ArgumentTypeError):
        main.max_output_tokens("0")


def test_main_prints_model_response(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["code-analysis", "--root-path", str(tmp_path), "Be concise."],
    )
    monkeypatch.setattr(main, "OpenAI", lambda: object())
    response = type("Response", (), {"output_parsed": object()})()

    def fake_analyze_text(
        text,
        instructions,
        client,
        simulate_validation_error_once,
        permission_policy,
        file_access_policy,
        run_state,
    ):
        return response

    monkeypatch.setattr(
        main,
        "analyze_text",
        fake_analyze_text,
    )
    monkeypatch.setattr(main, "format_response", lambda response: "Full API response")
    monkeypatch.setattr(
        main,
        "format_analysis_result_object",
        lambda result: "Formatted analysis result",
    )
    monkeypatch.setattr(main, "format_run_metrics", lambda metrics: "Metrics")
    timestamps = iter([10.0, 11.5])
    monkeypatch.setattr(main.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(main, "format_token_usage", lambda response: "Token usage: 10")

    main.main()

    assert (
        capsys.readouterr().out
        == "Allowed permissions: execute, read, write\nFormatted analysis result\nMetrics\n"
        "Changed Files\n  No files changed.\n"
        'Completion Check: {"passed": true, "failures": []}\n'
    )


def test_main_reports_analysis_errors(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["code-analysis", "--root-path", str(tmp_path), "Be concise."])
    monkeypatch.setattr(main, "OpenAI", lambda: object())
    monkeypatch.setattr(
        main,
        "analyze_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            main.AnalysisError("OpenAI request failed: invalid API key")
        ),
    )

    with pytest.raises(SystemExit):
        main.main()

    assert "error: OpenAI request failed: invalid API key" in capsys.readouterr().err
