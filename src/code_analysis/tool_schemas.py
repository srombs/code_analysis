"""Schemas and implementations for custom code-analysis tools."""

READ_SOURCE_LINE_TOOL = {
    "type": "function",
    "name": "read_source_line",
    "description": "Read one numbered line from the source file currently being analyzed.",
    "parameters": {
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
    },
    "strict": True,
}


def read_source_line(source_text: str, line_number: int) -> str:
    """Return one one-based line from the source currently being analyzed.

    The tool receives source text rather than an arbitrary file path. That keeps
    the tool limited to the exact file content the user already chose to analyze.
    """
    if isinstance(line_number, bool) or not isinstance(line_number, int):
        raise TypeError("line_number must be an integer.")

    source_lines = source_text.splitlines()
    if line_number < 1 or line_number > len(source_lines):
        raise ValueError(
            f"line_number must be between 1 and {len(source_lines)}, got {line_number}."
        )

    return f"{line_number}: {source_lines[line_number - 1]}"
