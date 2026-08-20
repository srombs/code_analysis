"""Command-line entry point for code-analysis."""

import argparse
import json
from pathlib import Path

from openai import OpenAI

MODEL = "gpt-5.6-luna"


def read_text_file(file_path: str) -> str:
    """Read a UTF-8 text file for use as model input."""
    return Path(file_path).read_text(encoding="utf-8")


def temperature(value: str) -> float:
    """Parse a temperature value accepted by the Responses API."""
    parsed_value = float(value)
    if not 0 <= parsed_value <= 2:
        raise argparse.ArgumentTypeError("temperature must be between 0 and 2")
    return parsed_value


def max_output_tokens(value: str) -> int:
    """Parse an output-token limit accepted by the selected model."""
    parsed_value = int(value)
    if not 1 <= parsed_value <= 128_000:
        raise argparse.ArgumentTypeError("max output tokens must be between 1 and 128000")
    return parsed_value


def analyze_text(
    text: str,
    instructions: str,
    temperature: float,
    max_output_tokens: int,
    client: OpenAI,
) -> object:
    """Send text to the configured OpenAI model and return the API response."""
    return client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=text,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "none"},
    )


def format_response(response: object) -> str:
    """Serialize an SDK response as formatted JSON for terminal output."""
    return json.dumps(response.model_dump(), indent=2)


def main() -> None:
    """Analyze a text file supplied from the command line."""
    parser = argparse.ArgumentParser(description="Analyze a piece of text.")
    parser.add_argument("file_path", help="Path to the UTF-8 text file to analyze")
    parser.add_argument("instructions", help="Instructions for the analysis")
    parser.add_argument(
        "temperature",
        type=temperature,
        help="Sampling temperature from 0 (focused) to 2 (more varied)",
    )
    parser.add_argument(
        "max_output_tokens",
        type=max_output_tokens,
        help="Maximum output tokens from 1 to 128000",
    )
    args = parser.parse_args()

    try:
        text = read_text_file(args.file_path)
    except OSError as error:
        parser.error(f"could not read {args.file_path}: {error.strerror or error}")

    response = analyze_text(
        text,
        args.instructions,
        args.temperature,
        args.max_output_tokens,
        OpenAI(),
    )
    print(format_response(response))


if __name__ == "__main__":
    main()
