"""Schemas and implementations for custom code-analysis tools."""

import os
import re
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SEARCH_CODE_MAX_RESULTS = 30
MAX_TOOL_OUTPUT_CHARS = 20_000
TOOL_OUTPUT_EDGE_CHARS = MAX_TOOL_OUTPUT_CHARS // 2
SEARCH_FILE_EXTENSIONS = frozenset(
    {
        ".dart",
        ".gradle",
        ".h",
        ".java",
        ".json",
        ".kt",
        ".kts",
        ".m",
        ".mm",
        ".plist",
        ".podspec",
        ".properties",
        ".swift",
        ".xcconfig",
        ".xml",
        ".yaml",
        ".yml",
    }
)


class ToolPermission(StrEnum):
    """Permission categories enforced by this application, not the OpenAI API."""

    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    EXECUTE = "execute"


@dataclass(frozen=True)
class FileAccessPolicy:
    """The filesystem boundaries enforced by every file-access tool."""

    root_path: Path
    protected_file_suffixes: frozenset[str] = frozenset(
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
    protected_file_names: frozenset[str] = frozenset(
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
    protected_directory_names: frozenset[str] = frozenset({".aws", ".git", ".hg", ".ssh", ".svn"})
    ignored_directory_names: frozenset[str] = frozenset(
        {
            ".dart_tool",
            ".gradle",
            ".idea",
            ".symlinks",
            ".vscode",
            "DerivedData",
            "Pods",
            "__pycache__",
            "build",
            "coverage",
            "node_modules",
        }
    )
    max_file_size_bytes: int = 10 * 1024 * 1024


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


@dataclass(frozen=True)
class FlutterTestResult:
    """The outcome and terminal output from one Flutter test-suite run."""

    exit_code: int
    stdout: str
    stderr: str
    success: bool
    output_truncated: bool


@dataclass(frozen=True)
class DartAnalyzeResult:
    """The outcome and terminal output from one Dart analyzer run."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    success: bool
    output_truncated: bool
    issues: tuple["DartAnalyzeIssue", ...]


@dataclass(frozen=True)
class DartAnalyzeIssue:
    """One issue reported by the Dart analyzer."""

    path: str
    line_number: int
    severity: str
    message: str


@dataclass(frozen=True)
class DartFormatResult:
    """The outcome and terminal output from one Dart formatter run."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PatchRequest:
    """One model-proposed text replacement."""

    path: str
    old_text: str
    new_text: str


@dataclass(frozen=True)
class PatchResult:
    """The outcome of applying one unambiguous text replacement."""

    success: bool
    path: str
    replacements: int
    message: str


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
        "10 MiB cannot be read. Paths must be relative to the application-provided "
        "project root. "
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
        "List files and subdirectories inside the application-provided project root. "
        "Paths must be relative to that root. Returned directories end with a slash and "
        "can be passed to this tool again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": (
                    "A directory path relative to the approved project root. Use an empty "
                    "string to list that root."
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
        "Search Dart and native mobile source/configuration files under the application-provided "
        "project root for text. Protected files, ignored directories, and files over 10 MiB "
        "are skipped. "
        "Return an object "
        "with: matches (up to 30 results, each with a path, one-based line number, and "
        "line text); total_matches (the count before limiting); and truncated (whether "
        "additional matches were omitted)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Text to search for case-insensitively in Dart and native mobile "
                    "source/configuration files."
                ),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


RUN_FLUTTER_TESTS_TOOL = {
    "type": "function",
    "name": "run_flutter_tests",
    "description": (
        "Run the full Flutter test suite with `flutter test` in the application-provided "
        "project root. This executes project code and requires execute permission. Return "
        "the process exit code, standard output, standard error, success status, and "
        "whether output was truncated."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}


RUN_DART_ANALYZE_TOOL = {
    "type": "function",
    "name": "run_dart_analyze",
    "description": (
        "Run `dart analyze` in the application-provided project root. This executes the "
        "Dart analyzer and requires execute permission. Return the process exit code, "
        "standard output, standard error, timeout status, success status, and whether "
        "output was truncated, along with structured reported issues."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}


RUN_DART_FORMAT_TOOL = {
    "type": "function",
    "name": "run_dart_format",
    "description": (
        "Run `dart format .` in the application-provided project root. This overwrites "
        "Dart source files and requires write permission. Return the process exit code, "
        "standard output, and standard error."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}


APPLY_PATCH_TOOL = {
    "type": "function",
    "name": "apply_patch",
    "description": (
        "Propose replacing text in one file inside the application-provided project root. "
        "This tool requires write permission. The old text must occur exactly once in the file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The project-relative path of the file to change.",
            },
            "old_text": {
                "type": "string",
                "description": "The exact existing text expected in that file.",
            },
            "new_text": {
                "type": "string",
                "description": "The replacement text to write in place of old_text.",
            },
        },
        "required": ["path", "old_text", "new_text"],
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
    # RUN_FLUTTER_TESTS_TOOL["name"]: frozenset({ToolPermission.EXECUTE}),
    RUN_DART_ANALYZE_TOOL["name"]: frozenset({ToolPermission.EXECUTE}),
    # RUN_DART_FORMAT_TOOL["name"]: frozenset({ToolPermission.WRITE}),
    APPLY_PATCH_TOOL["name"]: frozenset({ToolPermission.WRITE}),
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


def run_flutter_tests(
    project_directory: Path,
) -> FlutterTestResult:
    """Run the fixed Flutter project's tests without accepting arbitrary commands."""
    project_path = project_directory.resolve()
    if not (project_path / "pubspec.yaml").is_file():
        raise ValueError(f"Flutter project does not contain pubspec.yaml: {project_path}")

    try:
        completed_process = subprocess.run(
            ["flutter", "test"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("flutter test exceeded the 300-second time limit.") from error

    return FlutterTestResult(
        exit_code=completed_process.returncode,
        stdout=_truncate_tool_output(completed_process.stdout),
        stderr=_truncate_tool_output(completed_process.stderr),
        success=completed_process.returncode == 0,
        output_truncated=_tool_output_was_truncated(
            completed_process.stdout,
            completed_process.stderr,
        ),
    )


def run_dart_analyze(
    project_directory: Path,
) -> DartAnalyzeResult:
    """Run the Dart analyzer without accepting arbitrary commands or paths."""
    project_path = project_directory.resolve()
    if not (project_path / "pubspec.yaml").is_file():
        raise ValueError(f"Flutter project does not contain pubspec.yaml: {project_path}")

    try:
        completed_process = subprocess.run(
            ["dart", "analyze"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raw_stdout = _timeout_stream_to_text(error.stdout)
        raw_stderr = _timeout_stream_to_text(error.stderr)
        stdout = _truncate_tool_output(raw_stdout)
        stderr = _truncate_tool_output(raw_stderr)
        return DartAnalyzeResult(
            exit_code=None,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            success=False,
            output_truncated=_tool_output_was_truncated(raw_stdout, raw_stderr),
            issues=_parse_dart_analyze_issues(stdout, stderr),
        )

    stdout = _truncate_tool_output(completed_process.stdout)
    stderr = _truncate_tool_output(completed_process.stderr)
    return DartAnalyzeResult(
        exit_code=completed_process.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        success=completed_process.returncode == 0,
        output_truncated=_tool_output_was_truncated(
            completed_process.stdout,
            completed_process.stderr,
        ),
        issues=_parse_dart_analyze_issues(stdout, stderr),
    )


def _timeout_stream_to_text(value: str | bytes | None) -> str:
    """Normalize subprocess timeout output for a JSON-serializable tool result."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(encoding="utf-8", errors="replace")
    return value


def _truncate_tool_output(text: str) -> str:
    """Limit tool output while retaining its beginning and end for model context."""
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:TOOL_OUTPUT_EDGE_CHARS] + text[-TOOL_OUTPUT_EDGE_CHARS:]


def _tool_output_was_truncated(stdout: str, stderr: str) -> bool:
    """Return whether either captured stream exceeded the model-output limit."""
    return len(stdout) > MAX_TOOL_OUTPUT_CHARS or len(stderr) > MAX_TOOL_OUTPUT_CHARS


def _parse_dart_analyze_issues(stdout: str, stderr: str) -> tuple[DartAnalyzeIssue, ...]:
    """Parse supported human-readable Dart analyzer issue formats."""
    issues = []
    for output_line in f"{stdout}\n{stderr}".splitlines():
        path_first_match = re.fullmatch(
            r"(?P<severity>error|warning|info)\s*-\s*"
            r"(?P<path>.+?):(?P<line>\d+):\d+\s*-\s*"
            r"(?P<message>.+?)(?:\s*-\s*\S+)?",
            output_line,
            flags=re.IGNORECASE,
        )
        message_first_match = re.fullmatch(
            r"(?P<severity>error|warning|info)\s*-\s*"
            r"(?P<message>.+?)\s*-\s*"
            r"(?P<path>.+?):(?P<line>\d+):\d+(?:\s*-\s*\S+)?",
            output_line,
            flags=re.IGNORECASE,
        )
        match = message_first_match or path_first_match
        if match is None:
            continue

        issues.append(
            DartAnalyzeIssue(
                path=match["path"],
                line_number=int(match["line"]),
                severity=match["severity"].lower(),
                message=match["message"],
            )
        )
    return tuple(issues)


def run_dart_format(
    project_directory: Path,
) -> DartFormatResult:
    """Format fixed-project Dart source without accepting arbitrary commands or paths."""
    project_path = project_directory.resolve()
    if not (project_path / "pubspec.yaml").is_file():
        raise ValueError(f"Flutter project does not contain pubspec.yaml: {project_path}")

    try:
        completed_process = subprocess.run(
            ["dart", "format", "."],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("dart format exceeded the 300-second time limit.") from error

    return DartFormatResult(
        exit_code=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
    )


def _resolved_root_path(file_access_policy: FileAccessPolicy) -> Path:
    """Resolve the app-controlled project root used by every file tool."""
    root_path = file_access_policy.root_path.resolve()
    if not root_path.is_dir():
        raise ValueError(f"Project root is not a directory: {root_path}")
    return root_path


def _resolve_approved_path(
    requested_path: str,
    file_access_policy: FileAccessPolicy,
    path_name: str,
) -> Path:
    """Resolve a relative path and ensure it remains inside the project root."""
    root_path = _resolved_root_path(file_access_policy)
    path = Path(requested_path)
    if path.is_absolute():
        raise PermissionError(f"{path_name} must be relative to the approved project root.")

    resolved_path = (root_path / path).resolve()

    if not resolved_path.is_relative_to(root_path):
        raise PermissionError(f"Access is not available outside the project root: {path_name}.")

    return resolved_path


def _path_for_model(path: Path, root_path: Path) -> str:
    """Return a project-relative path for a model-facing tool result."""
    return str(path.relative_to(root_path))


def _is_protected_path(path: Path, file_access_policy: FileAccessPolicy) -> bool:
    """Return whether a path looks like a secret or version-control artifact."""
    normalized_name = path.name.casefold()
    normalized_parts = (part.casefold() for part in path.parts)

    return (
        normalized_name in file_access_policy.protected_file_names
        or normalized_name.startswith(".env.")
        or path.suffix.casefold() in file_access_policy.protected_file_suffixes
        or any(part in file_access_policy.protected_directory_names for part in normalized_parts)
    )


def _is_ignored_path(path: Path, file_access_policy: FileAccessPolicy) -> bool:
    """Return whether a path belongs to an ignored repository directory."""
    root_path = _resolved_root_path(file_access_policy)
    try:
        path_parts = path.resolve().relative_to(root_path).parts
    except ValueError:
        return False

    ignored_names = {name.casefold() for name in file_access_policy.ignored_directory_names}
    return any(part.casefold() in ignored_names for part in path_parts)


def read_file(
    file_path: str,
    file_access_policy: FileAccessPolicy,
) -> str:
    """Read a UTF-8 text file only when it is inside an approved directory.

    ``file_path`` is the value requested by the model. ``file_access_policy``
    comes from the app, never from the model, so the app keeps control over
    which files may be exposed.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")

    requested_path = _resolve_approved_path(
        file_path,
        file_access_policy,
        "file_path",
    )
    if _is_protected_path(requested_path, file_access_policy):
        raise ValueError("file_path identifies a protected file or directory.")
    if _is_ignored_path(requested_path, file_access_policy):
        raise ValueError("file_path identifies an ignored directory.")
    if not requested_path.is_file():
        raise ValueError(f"file_path does not identify a file: {file_path}")
    file_size_bytes = requested_path.stat().st_size
    if file_size_bytes > file_access_policy.max_file_size_bytes:
        raise ValueError(
            f"file_path is too large ({file_size_bytes} bytes); the maximum is "
            f"{file_access_policy.max_file_size_bytes} bytes."
        )

    return requested_path.read_text(encoding="utf-8")


def apply_patch(
    patch_request: PatchRequest,
    file_access_policy: FileAccessPolicy,
    read_file_paths: set[str],
) -> PatchResult:
    """Replace one unique text match in an approved UTF-8 project file."""
    if not isinstance(patch_request.path, str):
        raise TypeError("path must be a string.")
    if not isinstance(patch_request.old_text, str):
        raise TypeError("old_text must be a string.")
    if not isinstance(patch_request.new_text, str):
        raise TypeError("new_text must be a string.")
    if not patch_request.old_text:
        raise ValueError("old_text must not be empty.")
    if patch_request.path not in read_file_paths:
        return PatchResult(
            success=False,
            path=patch_request.path,
            replacements=0,
            message="The file must be read with read_file before applying a patch.",
        )

    source_text = read_file(patch_request.path, file_access_policy)
    old_text_match_count = source_text.count(patch_request.old_text)
    if old_text_match_count > 1:
        return PatchResult(
            success=False,
            path=patch_request.path,
            replacements=0,
            message=(
                "old_text matched more than once in the requested file; "
                f"found {old_text_match_count} matches."
            ),
        )
    if old_text_match_count == 0:
        return PatchResult(
            success=False,
            path=patch_request.path,
            replacements=0,
            message="old_text was not found in the requested file.",
        )

    requested_path = _resolve_approved_path(
        patch_request.path,
        file_access_policy,
        "path",
    )
    requested_path.write_text(
        source_text.replace(patch_request.old_text, patch_request.new_text, 1),
        encoding="utf-8",
    )
    return PatchResult(
        success=True,
        path=patch_request.path,
        replacements=1,
        message="Patch applied successfully.",
    )


def search_code(
    query: str,
    file_access_policy: FileAccessPolicy,
) -> SearchCodeResult:
    """Return up to 30 matches and metadata for eligible project text files."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query:
        raise ValueError("query must not be empty.")

    normalized_query = query.casefold()
    root_path = _resolved_root_path(file_access_policy)
    results = []
    total_matches = 0

    for directory, directory_names, file_names in os.walk(root_path):
        directory_path = Path(directory)
        directory_names[:] = [
            name
            for name in directory_names
            if not _is_protected_path(directory_path / name, file_access_policy)
            and not _is_ignored_path(directory_path / name, file_access_policy)
        ]
        for file_name in sorted(file_names):
            source_path = directory_path / file_name
            if (
                source_path.suffix.casefold() not in SEARCH_FILE_EXTENSIONS
                or _is_protected_path(source_path, file_access_policy)
                or _is_ignored_path(source_path, file_access_policy)
                or not source_path.resolve().is_relative_to(root_path)
            ):
                continue

            result_path = _path_for_model(source_path, root_path)
            try:
                source_text = read_file(str(source_path.relative_to(root_path)), file_access_policy)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            for line_number, line in enumerate(source_text.splitlines(), start=1):
                if normalized_query not in line.casefold():
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
    file_access_policy: FileAccessPolicy,
) -> str:
    """List direct files and directories inside an approved directory."""
    if not isinstance(directory_path, str):
        raise TypeError("directory_path must be a string.")

    requested_path = _resolve_approved_path(
        directory_path,
        file_access_policy,
        "directory_path",
    )
    if not requested_path.is_dir():
        raise ValueError(f"directory_path does not identify a directory: {directory_path}")

    paths = []
    for path in sorted(requested_path.iterdir()):
        if _is_protected_path(path, file_access_policy) or _is_ignored_path(
            path, file_access_policy
        ):
            continue

        model_path = _path_for_model(path, _resolved_root_path(file_access_policy))
        if path.is_dir():
            paths.append(f"{model_path}/")
        elif path.is_file():
            paths.append(model_path)

    return "\n".join(paths)
