# code-analysis

Tools for analyzing code.

## Overview

- Agent orchestration
  - Creates requests for the model and supplies only the tools allowed for the current run.
  - Processes model-requested tools across multiple rounds.
  - Keeps the run active when the completion gate identifies unfinished work.
  - Prints the final analysis, changed-file record, completion result, and run metrics.

- Tool layer
  - Defines the model-visible contracts for repository exploration, verification, and edits.
  - Implements project-bound file reads, directory listing, code search, and targeted patching.
  - Supports Dart analysis and Flutter tests when those tools are enabled.
  - Returns structured patch, verification, and analyzer-issue results to the model.

- Access control
  - Enforces the permissions granted for each run before executing a tool.
  - Restricts filesystem operations to the approved project root.
  - Rejects protected files, ignored directories, absolute paths, and path traversal.
  - Requires files to be inspected before the harness permits patching them.

- Response schemas
  - Defines structured data for the final analysis and findings.
  - Defines tool-result data for patches, analyzer issues, completion checks, and run metrics.
  - Keeps model-facing outputs predictable and easier for the harness to validate.

- Run state
  - Retains what the harness knows throughout an agent run.
  - Tracks read and changed files, patch and repair attempts, and verification status.
  - Stores Dart analyzer baseline information for comparisons after edits.
  - Accumulates model calls, tool calls, file activity, and token usage for metrics.

- Completion gate
  - Evaluates the final run state before allowing the agent to finish.
  - Rejects completion when verification is pending, verification failed, or repair limits were reached.
  - Sends an explanatory system message back to the model so it can continue with a grounded next step.

- Metrics
  - Produces a structured end-of-run summary.
  - Reports model and tool calls, searched/read/changed files, patch and repair attempts, token usage, runtime, and completion status.

## Getting started

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Usage

Set an API key, then pass the path to a UTF-8 text file as the first argument:

```bash
export OPENAI_API_KEY="your_api_key"
uv run python -m code_analysis.main \
  ./example.py \
  "Explain this function in three bullets." \
  0.3 \
  200
```

The first argument is read into a `text` variable and sent to the model. The remaining arguments
are the model instruction, a temperature between `0` and `2`, and maximum output tokens between
`1` and `128000`. A lower temperature produces more focused output; a higher value yields more
varied output. The command sends them to `gpt-5.6-luna` through the Responses API and prints the
structured model output items as formatted JSON, followed by the input and output token counts.

If the file path cannot be read, the command prompts for another path.
