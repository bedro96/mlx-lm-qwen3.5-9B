# Copilot Instructions

## Project Overview

A terminal-based multi-turn chat client **and** Flask web UI for a **smart factory agent** persona, running a local MLX LLM (Apple Silicon) via an OpenAI-compatible API endpoint.

## ⚠️ Smoke Test Requirement

**Always smoke test against the real API before considering any change done.**

```zsh
uv run smoke_test.py            # tests models, non-streaming, streaming
uv run smoke_test.py --think    # also tests streaming with thinking mode ON
```

- Target: `http://127.0.0.1:8080/v1` (real MLX server, **never mocks**)
- Flask UI test (`GET http://127.0.0.1:5000`) is skipped gracefully when Flask isn't running
- Exit code 0 = all tests passed

## Architecture

The app requires the MLX server to be running separately:

```zsh
uvx --from mlx-lm mlx_lm.server
```

**Files:**
- `client.py` — shared logic: `make_client()`, `resolve_model()`, `stream_reply()`, constants
- `main.py` — CLI chat client with streaming and `--think` flag
- `app.py` — Flask web UI with SSE streaming (`GET /`, `GET /models`, `POST /chat`)
- `templates/index.html` — browser chat UI using the `EventSource`/fetch API
- `smoke_test.py` — smoke tests against real API at `http://127.0.0.1:8080/v1`

**Running:**
```zsh
uv run main.py           # CLI (thinking mode OFF by default)
uv run main.py --think   # CLI with thinking mode ON
uv run app.py            # Flask web UI at http://127.0.0.1:5000
```

`resolve_model()` auto-selects the model: prefers `mlx-community/Qwen3.5-9B-MLX-4bit`, falls back to the first available model.

`stream_reply()` is the single authoritative streaming implementation. It filters `<think>…</think>` blocks from `delta.content` using a streaming state machine (tags may span multiple chunks).

## Package Manager

This project uses **`uv`**. Always use `uv`, not `pip` directly.

```zsh
uv add <package>        # add a dependency
uv add --dev <package>  # add a dev dependency
uv sync                 # install all dependencies
```

Requires Python 3.13+.

## Dev Commands

```zsh
uv run black main.py app.py client.py smoke_test.py     # format
uv run flake8 main.py app.py client.py smoke_test.py    # lint
uv run mypy main.py app.py client.py smoke_test.py      # type-check (strict)
uv run bandit main.py app.py client.py smoke_test.py    # security scan
```

Config: `pyproject.toml` (`[tool.black]`, `[tool.mypy]`, `[tool.bandit]`) and `.flake8`. Line length: 99.

## Qwen3 Thinking Mode

The model supports chain-of-thought "thinking". **Default: OFF** (cleaner, faster).

| Control Method | Enable | Disable |
|---|---|---|
| `extra_body` (API) | `{"chat_template_kwargs": {"enable_thinking": True}}` | `{"chat_template_kwargs": {"enable_thinking": False}}` |
| Prompt soft-switch (per turn) | prefix user message with `/think` | prefix with `/no_think` |

**How the server actually streams thinking** (verified against mlx-community/Qwen3.5-9B-MLX-4bit):
- Reasoning tokens arrive in `delta.model_extra["reasoning"]` — NOT in `delta.content`
- `delta.content` is `None` during the entire reasoning phase
- After reasoning finishes, final answer tokens arrive in `delta.content`
- Thinking does **not** use `<think>` tags in `delta.content`

**⚠️ OOM risk**: Without `max_tokens`, the model can generate thousands of reasoning tokens before answering, crashing the MLX server. Always set `max_tokens` when thinking is ON. The `_THINKING_MAX_TOKENS = 16384` constant in `client.py` handles this.

**Smoke test probe for thinking=ON** must use a very short prompt (e.g., "Answer with exactly OK.") — complex prompts generate 500–1500+ reasoning chunks before answering.

**Recommended sampling params** (set in `client.py`):
- Thinking ON: `temperature=0.6, top_p=0.95, top_k=20`
- Thinking OFF: `temperature=0.7, top_p=0.8, top_k=20`

`top_k` is a local-server extension passed inside `extra_body`, not as a top-level API param.

**Key functions** in `client.py`:
- `stream_reply()` — yields only final answer tokens (hides reasoning)
- `stream_reply_with_thinking()` — yields `(kind, token)` tuples where `kind` is `"thinking"` or `"answer"`
- `_extract_reasoning_token(delta)` — reads `delta.model_extra["reasoning"]`

## Key Conventions

- **ANSI color scheme** (CLI only): `ORANGE` for user prompt, `WHITE` for user input, `GREEN` for agent output, `RESET` after each turn.
- **Message history**: plain `list[ChatCompletionMessageParam]`. System prompt is always first and never removed. Flask UI maintains history client-side in JS.
- **All constants in `client.py`**: `BASE_URL`, `DEFAULT_MODEL`, `SYSTEM_PROMPT`, sampling params — adjust here to change server target, model, or persona.
- **Strict mypy**: all files pass `mypy --strict`. No `type: ignore` without a specific error code.
