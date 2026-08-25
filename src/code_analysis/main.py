"""Command-line entry point for code-analysis."""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MODEL = "gpt-5.6-luna"
MAX_VALIDATION_RETRIES = 2
INPUT_TOKEN_COST_PER_MILLION = 0.20
OUTPUT_TOKEN_COST_PER_MILLION = 1.20


class AnalysisError(Exception):
    """Raised when the model request cannot be completed."""


class Finding(BaseModel):
    """One actionable issue found in a source file."""

    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)
    problem: str
    solution: str
    severity: Literal["low", "medium", "high"]
    start_character: int | None = Field(default=None, ge=0)
    end_character: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_positions(self) -> "Finding":
        """Ensure every reported source range runs forward."""
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if (
            self.start_character is not None
            and self.end_character is not None
            and self.end_character < self.start_character
        ):
            raise ValueError("end_character must be greater than or equal to start_character")
        return self


class AnalysisResult(BaseModel):
    """The validated, application-specific result of a code analysis."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    findings: list[Finding]


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
    source_line_count = len(numbered_source.splitlines())

    for finding_number, finding in enumerate(result.findings, start=1):
        if finding.end_line > source_line_count:
            raise AnalysisError(
                f"Finding {finding_number} references lines {finding.start_line}-"
                f"{finding.end_line}, but the input contains only {source_line_count} lines"
            )


def analyze_text(
    text: str,
    instructions: str,
    client: OpenAI,
    simulate_validation_error_once: bool = False,
) -> object:
    """Send text for analysis and return a response containing an AnalysisResult."""
    numbered_source = number_source_lines(text)

    for attempt in range(MAX_VALIDATION_RETRIES + 1):
        try:
            if simulate_validation_error_once and attempt == 0:
                AnalysisResult.model_validate({})

            response = client.responses.parse(
                model=MODEL,
                instructions=instructions,
                input=numbered_source,
                text_format=AnalysisResult,
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
    parser = argparse.ArgumentParser(description="Analyze a piece of text.")
    parser.add_argument("file_path", help="Path to the UTF-8 text file to analyze")
    parser.add_argument("instructions", help="Instructions for the analysis")
    parser.add_argument(
        "--simulate-validation-error-once",
        action="store_true",
        help="Testing only: force one validation error before calling the API",
    )
    args = parser.parse_args()

    text = read_text_file_with_retry(args.file_path)

    try:
        client = OpenAI()
    except Exception as error:
        parser.error(f"could not initialize OpenAI client: {error}")

    analysis_started_at = time.perf_counter()

    try:
        response = analyze_text(
            text,
            args.instructions,
            client,
            simulate_validation_error_once=args.simulate_validation_error_once,
        )
    except AnalysisError as error:
        parser.error(str(error))

    analysis_elapsed_seconds = time.perf_counter() - analysis_started_at

    print(format_analysis_result_object(response.output_parsed))
    print(format_analysis_metrics(response, analysis_elapsed_seconds))
    # print(format_token_usage(response))


if __name__ == "__main__":
    main()
