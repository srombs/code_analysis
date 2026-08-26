import argparse
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from code_analysis import main
from code_analysis.schemas import AnalysisResult, Finding
from code_analysis.tool_schemas import (
    DEFAULT_APPROVED_DIRECTORY,
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    READ_SOURCE_LINE_TOOL,
    SEARCH_CODE_MAX_RESULTS,
    SEARCH_CODE_TOOL,
    SearchCodeResult,
    SearchResult,
    list_files,
    read_file,
    read_source_line,
    search_code,
)


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


def test_read_file_has_the_door_opener_lib_directory_as_its_default() -> None:
    assert DEFAULT_APPROVED_DIRECTORY == Path("/Users/rombs/Documents/gits/door-opener/lib")


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

    assert read_file("example.dart", tmp_path) == "print('hello')\n"


def test_read_file_rejects_paths_outside_the_approved_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="approved directory"):
        read_file("../outside.txt", tmp_path)


def test_read_file_rejects_non_dart_files(tmp_path) -> None:
    (tmp_path / "settings.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must have the \.dart extension"):
        read_file("settings.json", tmp_path)


def test_search_code_returns_a_result_for_each_matching_line(tmp_path) -> None:
    (tmp_path / "example.dart").write_text(
        "first line\nfind this\n",
        encoding="utf-8",
    )
    widgets_directory = tmp_path / "widgets"
    widgets_directory.mkdir()
    (widgets_directory / "button.dart").write_text("find this too\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("find this\n", encoding="utf-8")

    assert search_code("find this", tmp_path) == SearchCodeResult(
        matches=(
            SearchResult(path="example.dart", line_number=2, text="find this"),
            SearchResult(path="widgets/button.dart", line_number=1, text="find this too"),
        ),
        total_matches=2,
        truncated=False,
    )


def test_search_code_rejects_an_empty_query(tmp_path) -> None:
    (tmp_path / "example.dart").write_text("anything", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        search_code("", tmp_path)


def test_search_code_stops_after_thirty_matches(tmp_path) -> None:
    (tmp_path / "many_matches.dart").write_text(
        "\n".join("match" for _ in range(31)),
        encoding="utf-8",
    )

    result = search_code("match", tmp_path)

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

    assert list_files("widgets", tmp_path) == "widgets/button.py\nwidgets/input.py"
    assert list_files("", tmp_path) == "widgets/\nzebra.py"


def test_list_files_rejects_paths_outside_the_approved_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="approved directory"):
        list_files("../", tmp_path)


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
            "tools": [main.READ_FILE_TOOL, main.LIST_FILES_TOOL, main.SEARCH_CODE_TOOL],
            "tool_choice": "auto",
        }
    ]


def test_analyze_text_executes_a_requested_file_read_and_returns_its_output(
    monkeypatch,
    capsys,
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

    monkeypatch.setattr(main, "read_file", lambda file_path: "class Widget {}\n")
    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("entry point", "Analyze related code.", client) is final_response
    assert calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "1: class Widget {}",
        }
    ]
    assert calls[1]["previous_response_id"] == "response_1"
    assert capsys.readouterr().out == (
        "Tool call: read_file(widget.dart)\nTool result: read widget.dart (1 lines)\n"
    )


def test_analyze_text_executes_a_requested_file_list_and_returns_its_output(
    monkeypatch,
    capsys,
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

    monkeypatch.setattr(main, "list_files", lambda directory_path: "widgets/button.py")
    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("entry point", "Analyze related code.", client) is final_response
    assert calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_456",
            "output": "widgets/button.py",
        }
    ]
    assert capsys.readouterr().out == (
        "Tool call: list_files(widgets)\n"
        "Tool result: listed 1 files\n"
        "Tool output:\n"
        "widgets/button.py\n"
    )


def test_analyze_text_executes_a_requested_code_search_and_returns_its_output(
    monkeypatch,
    capsys,
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
        lambda query: SearchCodeResult(
            matches=(SearchResult(path="button.dart", line_number=7, text="class Button {}"),),
            total_matches=1,
            truncated=False,
        ),
    )
    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("entry point", "Analyze related code.", client) is final_response
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
    assert '"total_matches": 1' in terminal_output


def test_file_tool_errors_are_returned_to_the_model(monkeypatch) -> None:
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
        lambda file_path: (_ for _ in ()).throw(ValueError("path is not allowed")),
    )

    assert main.execute_file_tool_calls(response) == [
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


def test_temperature_accepts_values_from_zero_to_two() -> None:
    assert main.temperature("0.7") == 0.7

    with pytest.raises(argparse.ArgumentTypeError):
        main.temperature("2.1")


def test_max_output_tokens_accepts_values_from_one_to_model_limit() -> None:
    assert main.max_output_tokens("200") == 200

    with pytest.raises(argparse.ArgumentTypeError):
        main.max_output_tokens("0")


def test_main_prints_model_response(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["code-analysis", "Be concise."],
    )
    monkeypatch.setattr(main, "OpenAI", lambda: object())
    response = type("Response", (), {"output_parsed": object()})()
    monkeypatch.setattr(
        main,
        "analyze_text",
        lambda text, instructions, client, simulate_validation_error_once: response,
    )
    monkeypatch.setattr(main, "format_response", lambda response: "Full API response")
    monkeypatch.setattr(
        main,
        "format_analysis_result_object",
        lambda result: "Formatted analysis result",
    )
    monkeypatch.setattr(main, "format_analysis_metrics", lambda response, elapsed: "Metrics")
    timestamps = iter([10.0, 11.5])
    monkeypatch.setattr(main.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(main, "format_token_usage", lambda response: "Token usage: 10")

    main.main()

    assert capsys.readouterr().out == "Formatted analysis result\nMetrics\n"


def test_main_reports_analysis_errors(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["code-analysis", "Be concise."])
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
