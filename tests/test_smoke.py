import argparse
import sys

import pytest

from code_analysis import main


def test_main_is_available() -> None:
    assert callable(main.main)


def test_analyze_text_uses_luna_model() -> None:
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return type("Response", (), {"output_text": "Analysis complete."})()

    client = type("Client", (), {"responses": FakeResponses()})()

    assert main.analyze_text("hello world", "Be concise.", 0.3, 200, client) == "Analysis complete."
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
    monkeypatch.setattr(
        main,
        "analyze_text",
        lambda text, instructions, temperature, max_tokens, client: (
            f"{instructions} ({temperature}, {max_tokens}) Analyzed: {text}"
        ),
    )

    main.main()

    assert capsys.readouterr().out == "Be concise. (0.3, 200) Analyzed: hello world\n"
