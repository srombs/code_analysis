"""Schemas and implementations for custom code-analysis tools."""

from pathlib import Path

DEFAULT_APPROVED_DIRECTORY = Path("/Users/rombs/Documents/gits/door-opener/lib")

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


READ_SOURCE_FILE_TOOL = {
    "type": "function",
    "name": "read_source_file",
    "description": "Read a UTF-8 text source file from the approved project directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The relative path of the source file to read.",
            }
        },
        "required": ["file_path"],
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


def read_source_file(
    file_path: str,
    approved_directory: Path = DEFAULT_APPROVED_DIRECTORY,
) -> str:
    """Read a UTF-8 text file only when it is inside ``approved_directory``.

    ``file_path`` is the value requested by the model. ``approved_directory``
    comes from the app, never from the model, so the app keeps control over
    which files may be exposed.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")

    approved_path = approved_directory.resolve()
    requested_path = (approved_path / file_path).resolve()
    if not requested_path.is_relative_to(approved_path):
        raise ValueError("file_path must stay inside the approved directory.")
    if not requested_path.is_file():
        raise ValueError(f"file_path does not identify a file: {file_path}")

    return requested_path.read_text(encoding="utf-8")
