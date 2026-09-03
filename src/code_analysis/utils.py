"""Pure parsing and formatting helpers for the code-analysis application."""

import argparse
import difflib
import json
import re

from pydantic import BaseModel

from code_analysis.schemas import AnalysisResult
from code_analysis.tool_schemas import DartAnalyzeResult

MAX_ANALYZER_DIFF_CHARS = 20_000
INPUT_TOKEN_COST_PER_MILLION = 0.20
OUTPUT_TOKEN_COST_PER_MILLION = 1.20


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


def number_source_lines(text: str) -> str:
    """Add one-based line numbers so the model can reference source locations."""
    return "\n".join(
        f"{line_number}: {line}" for line_number, line in enumerate(text.splitlines(), start=1)
    )


def analyzer_output(analyze_result: DartAnalyzeResult) -> str:
    """Combine analyzer streams into a stable baseline representation."""
    return f"stdout:\n{analyze_result.stdout}\nstderr:\n{analyze_result.stderr}"


def analyzer_output_diff(baseline_output: str, current_output: str) -> str:
    """Return a bounded unified diff between analyzer runs."""
    diff = "\n".join(
        difflib.unified_diff(
            baseline_output.splitlines(),
            current_output.splitlines(),
            fromfile="baseline",
            tofile="current",
            lineterm="",
        )
    )
    if len(diff) <= MAX_ANALYZER_DIFF_CHARS:
        return diff
    edge_chars = MAX_ANALYZER_DIFF_CHARS // 2
    return diff[:edge_chars] + diff[-edge_chars:]


def analyzer_issue_count(analyzer_output: str) -> int | None:
    """Extract Dart analyzer's reported issue count from its output."""
    if re.search(r"\bno issues found\b", analyzer_output, flags=re.IGNORECASE):
        return 0
    match = re.search(r"\b(\d+) issues? found\b", analyzer_output, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


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
    """Format a response's input and output token counts."""
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


def format_run_metrics(metrics: BaseModel) -> str:
    """Serialize final run metrics for terminal logging."""
    return f"Run Metrics: {metrics.model_dump_json()}"


def format_changed_files(file_paths: set[str]) -> str:
    """Format the project-relative files changed during one agent run."""
    lines = ["Changed Files"]
    if not file_paths:
        lines.append("  No files changed.")
    else:
        lines.extend(f"  - {path}" for path in sorted(file_paths))
    return "\n".join(lines)
