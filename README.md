# code-analysis

Tools for analyzing code.

## Getting started

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Usage

Set an API key, then pass text as a positional argument:

```bash
export OPENAI_API_KEY="your_api_key"
uv run python -m code_analysis.main \
  "def add(a, b): return a + b" \
  "Explain this function in three bullets." \
  0.3 \
  200
```

The arguments are the text to analyze, the model instruction, and a temperature between `0` and
`2`, followed by the maximum output tokens between `1` and `128000`. A lower temperature produces
more focused output; a higher value yields more varied output. The command sends them to
`gpt-5.6-luna` through the Responses API and prints the result.
