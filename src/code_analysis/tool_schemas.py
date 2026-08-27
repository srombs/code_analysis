"""Schemas and implementations for custom code-analysis tools."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

APPROVED_DIRECTORIES = (
    Path("/Users/rombs/Documents/gits/door-opener/lib"),
    Path("/Users/rombs/Documents/gits/FW_IMP"),
)
PROTECTED_FILE_SUFFIXES = frozenset(
    {
        ".cer",
        ".crt",
        ".der",
        ".jks",
        ".kdbx",
        ".key",
        ".keystore",
        ".mobileprovision",
        ".p12",
        ".pem",
        ".pfx",
        ".ppk",
    }
)
PROTECTED_FILE_NAMES = frozenset(
    {
        ".ds_store",
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
PROTECTED_DIRECTORY_NAMES = frozenset({".aws", ".git", ".hg", ".ssh", ".svn"})
SEARCH_FILE_EXTENSION = ".dart"
SEARCH_CODE_MAX_RESULTS = 30
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class ToolPermission(StrEnum):
    """Permission categories enforced by this application, not the OpenAI API."""

    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    EXECUTE = "execute"


@dataclass(frozen=True)
class SearchResult:
    """One text match found by the ``search_code`` tool."""

    path: str
    line_number: int
    text: str


@dataclass(frozen=True)
class SearchCodeResult:
    """The returned search matches and the total number found."""

    matches: tuple[SearchResult, ...]
    total_matches: int
    truncated: bool


@dataclass(frozen=True)
class FileState:
    """The metadata and numbered contents of one source file sent to the model."""

    file_path: str
    line_count: int
    numbered_content: str


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
    "description": (
        "Read a UTF-8 text file from an approved project directory. Secret, credential, "
        "and version-control files are protected and cannot be read. Files larger than "
        "10 MiB cannot be read. Relative "
        "paths are rooted at /Users/rombs/Documents/gits/door-opener/lib. To read "
        "FW_IMP, use an absolute path under /Users/rombs/Documents/gits/FW_IMP. "
        "Return an object with file_path, line_count, and numbered_content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "A non-protected text-file path within an approved directory.",
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
        "List files and subdirectories inside an approved directory. Relative paths are "
        "rooted at /Users/rombs/Documents/gits/door-opener/lib. To browse FW_IMP, "
        "start with /Users/rombs/Documents/gits/FW_IMP. Returned directories end with "
        "a slash and can be passed to this tool again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": (
                    "A directory path within an approved directory. Use an empty string "
                    "to list the Door Opener root."
                ),
            }
        },
        "required": ["directory_path"],
        "additionalProperties": False,
    },
    "strict": True,
}


SEARCH_CODE_TOOL = {
    "type": "function",
    "name": "search_code",
    "description": (
        "Search Dart source files in every approved directory for text. Return an object "
        "with: matches (up to 30 results, each with a path, one-based line number, and "
        "line text); total_matches (the count before limiting); and truncated (whether "
        "additional matches were omitted)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The exact text to search for in Dart source files.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


# Keep authorization metadata separate from the schemas sent to the model.
TOOL_PERMISSIONS: dict[str, frozenset[ToolPermission]] = {
    READ_SOURCE_LINE_TOOL["name"]: frozenset({ToolPermission.READ}),
    READ_FILE_TOOL["name"]: frozenset({ToolPermission.READ}),
    LIST_FILES_TOOL["name"]: frozenset({ToolPermission.READ}),
    SEARCH_CODE_TOOL["name"]: frozenset({ToolPermission.READ}),
}


def get_tool_permissions(tool_name: str) -> frozenset[ToolPermission]:
    """Return the permissions a tool requires, or none for an unknown tool."""
    return TOOL_PERMISSIONS.get(tool_name, frozenset())


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


def _resolved_approved_directories(
    approved_directories: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Resolve the app-controlled directories allowed for file tools."""
    if not approved_directories:
        raise ValueError("At least one approved directory is required.")

    return tuple(directory.resolve() for directory in approved_directories)


def _resolve_approved_path(
    requested_path: str,
    approved_directories: tuple[Path, ...],
    path_name: str,
) -> tuple[Path, tuple[Path, ...]]:
    """Resolve a requested path and ensure it belongs to an allowed directory."""
    approved_paths = _resolved_approved_directories(approved_directories)
    path = Path(requested_path)
    resolved_path = path.resolve() if path.is_absolute() else (approved_paths[0] / path).resolve()

    if not any(resolved_path.is_relative_to(approved_path) for approved_path in approved_paths):
        raise PermissionError(
            f"Access is not available outside the approved directories: {path_name}."
        )

    return resolved_path, approved_paths


def _path_for_model(path: Path, approved_directories: tuple[Path, ...]) -> str:
    """Keep Door Opener paths relative; make other-root paths unambiguous."""
    primary_directory = approved_directories[0]
    if path.is_relative_to(primary_directory):
        return str(path.relative_to(primary_directory))

    return str(path)


def _is_protected_path(path: Path) -> bool:
    """Return whether a path looks like a secret or version-control artifact."""
    normalized_name = path.name.casefold()
    normalized_parts = (part.casefold() for part in path.parts)

    return (
        normalized_name in PROTECTED_FILE_NAMES
        or normalized_name.startswith(".env.")
        or path.suffix.casefold() in PROTECTED_FILE_SUFFIXES
        or any(part in PROTECTED_DIRECTORY_NAMES for part in normalized_parts)
    )


def read_file(
    file_path: str,
    approved_directories: tuple[Path, ...] = APPROVED_DIRECTORIES,
) -> str:
    """Read a UTF-8 text file only when it is inside an approved directory.

    ``file_path`` is the value requested by the model. ``approved_directories``
    comes from the app, never from the model, so the app keeps control over
    which files may be exposed.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")

    requested_path, _ = _resolve_approved_path(
        file_path,
        approved_directories,
        "file_path",
    )
    if _is_protected_path(requested_path):
        raise ValueError("file_path identifies a protected file or directory.")
    if not requested_path.is_file():
        raise ValueError(f"file_path does not identify a file: {file_path}")
    file_size_bytes = requested_path.stat().st_size
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"file_path is too large ({file_size_bytes} bytes); the maximum is "
            f"{MAX_FILE_SIZE_BYTES} bytes."
        )

    return requested_path.read_text(encoding="utf-8")


def search_code(
    query: str,
    approved_directories: tuple[Path, ...] = APPROVED_DIRECTORIES,
) -> SearchCodeResult:
    """Return up to 30 matches and metadata for all approved Dart files."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query:
        raise ValueError("query must not be empty.")

    approved_paths = _resolved_approved_directories(approved_directories)
    results = []
    total_matches = 0

    for approved_path in approved_paths:
        for source_path in sorted(approved_path.rglob(f"*{SEARCH_FILE_EXTENSION}")):
            if (
                not source_path.is_file()
                or _is_protected_path(source_path)
                or not source_path.resolve().is_relative_to(approved_path)
            ):
                continue

            result_path = _path_for_model(source_path, approved_paths)
            source_text = read_file(str(source_path), approved_paths)
            for line_number, line in enumerate(source_text.splitlines(), start=1):
                if query not in line:
                    continue

                total_matches += 1
                if len(results) < SEARCH_CODE_MAX_RESULTS:
                    results.append(
                        SearchResult(path=result_path, line_number=line_number, text=line)
                    )

    return SearchCodeResult(
        matches=tuple(results),
        total_matches=total_matches,
        truncated=total_matches > SEARCH_CODE_MAX_RESULTS,
    )


def list_files(
    directory_path: str,
    approved_directories: tuple[Path, ...] = APPROVED_DIRECTORIES,
) -> str:
    """List direct files and directories inside an approved directory."""
    if not isinstance(directory_path, str):
        raise TypeError("directory_path must be a string.")

    requested_path, approved_paths = _resolve_approved_path(
        directory_path,
        approved_directories,
        "directory_path",
    )
    if not requested_path.is_dir():
        raise ValueError(f"directory_path does not identify a directory: {directory_path}")

    paths = []
    for path in sorted(requested_path.iterdir()):
        if _is_protected_path(path):
            continue

        model_path = _path_for_model(path, approved_paths)
        if path.is_dir():
            paths.append(f"{model_path}/")
        elif path.is_file():
            paths.append(model_path)

    return "\n".join(paths)
