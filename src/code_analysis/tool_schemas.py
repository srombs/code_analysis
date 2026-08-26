"""Schemas and implementations for custom code-analysis tools."""

from pathlib import Path

DEFAULT_APPROVED_DIRECTORY = Path("/Users/rombs/Documents/gits/door-opener/lib")
ALLOWED_FILE_EXTENSION = ".dart"

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


READ_FILE_TOOL = {
    "type": "function",
    "name": "read_file",
    "description": "Read a UTF-8 Dart source file from the approved project directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The relative path of the .dart source file to read.",
            }
        },
        "required": ["file_path"],
        "additionalProperties": False,
    },
    "strict": True,
}


LIST_FILES_TOOL = {
    "type": "function",
    "name": "list_files",
    "description": (
        "List files and subdirectories in a directory inside the approved project "
        "directory. Returned directories end with a slash and can be passed to this "
        "tool again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": (
                    "The relative directory path to list. Use an empty string to list "
                    "the approved project directory root."
                ),
            }
        },
        "required": ["directory_path"],
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


def read_file(
    file_path: str,
    approved_directory: Path = DEFAULT_APPROVED_DIRECTORY,
) -> str:
    """Read a UTF-8 Dart file only when it is inside ``approved_directory``.

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
    if requested_path.suffix != ALLOWED_FILE_EXTENSION:
        raise ValueError(f"file_path must have the {ALLOWED_FILE_EXTENSION} extension.")
    if not requested_path.is_file():
        raise ValueError(f"file_path does not identify a file: {file_path}")

    return requested_path.read_text(encoding="utf-8")


def list_files(
    directory_path: str,
    approved_directory: Path = DEFAULT_APPROVED_DIRECTORY,
) -> str:
    """List direct files and directories as paths relative to the approved root."""
    if not isinstance(directory_path, str):
        raise TypeError("directory_path must be a string.")

    approved_path = approved_directory.resolve()
    requested_path = (approved_path / directory_path).resolve()
    if not requested_path.is_relative_to(approved_path):
        raise ValueError("directory_path must stay inside the approved directory.")
    if not requested_path.is_dir():
        raise ValueError(f"directory_path does not identify a directory: {directory_path}")

    paths = []
    for path in sorted(requested_path.iterdir()):
        relative_path = str(path.relative_to(approved_path))
        if path.is_dir():
            paths.append(f"{relative_path}/")
        elif path.is_file():
            paths.append(relative_path)

    return "\n".join(paths)
