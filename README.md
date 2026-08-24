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
