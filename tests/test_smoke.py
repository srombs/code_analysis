import argparse
import sys

import pytest
from pydantic import ValidationError

from code_analysis import main


def test_main_is_available() -> None:
    assert callable(main.main)


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
            "instructions": "Be concise.",
            "input": "1: first line\n2: second line",
            "text_format": main.AnalysisResult,
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

    class FakeResponses:
        def parse(self, **kwargs):
            return response

    client = type("Client", (), {"responses": FakeResponses()})()

    with pytest.raises(main.AnalysisError, match="did not contain a parsed analysis result"):
        main.analyze_text("hello world", "Be concise.", client)


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
        ["code-analysis", "example.py", "Be concise."],
    )
    monkeypatch.setattr(main, "OpenAI", lambda: object())
    monkeypatch.setattr(main, "read_text_file_with_retry", lambda file_path: "hello world")
    response = type("Response", (), {"output_parsed": "AnalysisResult(...)"})()
    monkeypatch.setattr(main, "analyze_text", lambda text, instructions, client: response)
    monkeypatch.setattr(main, "format_response", lambda response: "Full API response")
    monkeypatch.setattr(
        main, "format_analysis_result", lambda response: "Structured analysis result"
    )
    monkeypatch.setattr(main, "format_token_usage", lambda response: "Token usage: 10")

    main.main()

    assert capsys.readouterr().out == "AnalysisResult(...)\nStructured analysis result\n"


def test_main_reports_analysis_errors(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["code-analysis", "example.py", "Be concise."])
    monkeypatch.setattr(main, "read_text_file_with_retry", lambda file_path: "hello world")
    monkeypatch.setattr(main, "OpenAI", lambda: object())
    monkeypatch.setattr(
        main,
        "analyze_text",
        lambda *args: (_ for _ in ()).throw(
            main.AnalysisError("OpenAI request failed: invalid API key")
        ),
    )

    with pytest.raises(SystemExit):
        main.main()

    assert "error: OpenAI request failed: invalid API key" in capsys.readouterr().err
