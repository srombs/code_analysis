"""Command-line entry point for code-analysis."""

import argparse

from openai import OpenAI

MODEL = "gpt-5.6-luna"


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
) -> str:
    """Send text to the configured OpenAI model and return its output."""
    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=text,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "none"},
    )
    return response.output_text


def main() -> None:
    """Analyze text supplied from the command line."""
    parser = argparse.ArgumentParser(description="Analyze a piece of text.")
    parser.add_argument("text", help="Text to analyze")
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

    print(
        analyze_text(
            args.text,
            args.instructions,
            args.temperature,
            args.max_output_tokens,
            OpenAI(),
        )
    )


if __name__ == "__main__":
    main()
