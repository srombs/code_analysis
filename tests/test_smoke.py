import argparse
import sys

import pytest

from code_analysis import main


def test_main_is_available() -> None:
    assert callable(main.main)


def test_read_text_file_returns_file_contents(tmp_path) -> None:
    source_file = tmp_path / "example.py"
    source_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    assert main.read_text_file(str(source_file)) == "def add(a, b):\n    return a + b\n"


def test_analyze_text_uses_luna_model() -> None:
    calls = []
    response = type("Response", (), {"model_dump": lambda self: {"id": "resp_123"}})()

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return response

    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("hello world", "Be concise.", 0.3, 200, client) is response
    assert calls == [
        {
            "model": "gpt-5.6-luna",
            "instructions": "Be concise.",
            "input": "hello world",
            "temperature": 0.3,
            "max_output_tokens": 200,
            "reasoning": {"effort": "none"},
        }
    ]


def test_format_response_returns_pretty_json() -> None:
    response = type("Response", (), {"model_dump": lambda self: {"id": "resp_123"}})()

    assert main.format_response(response) == '{\n  "id": "resp_123"\n}'


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
        ["code-analysis", "hello world", "Be concise.", "0.3", "200"],
    )
    monkeypatch.setattr(main, "OpenAI", lambda: object())
    monkeypatch.setattr(main, "read_text_file", lambda file_path: "hello world")
    monkeypatch.setattr(
        main,
        "analyze_text",
        lambda text, instructions, temperature, max_tokens, client: object(),
    )
    monkeypatch.setattr(main, "format_response", lambda response: "Full API response")

    main.main()

    assert capsys.readouterr().out == "Full API response\n"
