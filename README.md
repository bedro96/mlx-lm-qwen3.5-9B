# 🏭 Smart Factory Agent — Local LLM on Apple Silicon

A terminal-based multi-turn chat client **and** Flask web UI for a **smart factory agent** persona, powered by a local LLM running on Apple Silicon via [MLX](https://github.com/apple/mlx).

No cloud API keys needed — everything runs locally on your Mac.

---

## 📁 Project Structure

```
foundry_local/
├── client.py          # Shared logic: OpenAI client, model resolution, streaming
├── main.py            # CLI chat client (terminal, streaming, --think flag)
├── app.py             # Flask web UI with SSE streaming
├── app_mcp.py         # Flask web UI with SSE + stdio MCP tool calls
├── main_mcp.py        # CLI chat client with stdio MCP tool calls
├── mcp_server.py      # Local stdio MCP server for mock machine temperature tools
├── foundry_local.py   # Hybrid Foundry Local SDK manager + MLX OpenAI backend
├── smoke_test.py      # Integration tests against the real MLX API server
├── templates/
│   └── index.html     # Browser chat UI (EventSource/fetch, marked.js rendering)
├── img/               # Chart images embedded in this README
├── pyproject.toml     # Project metadata, dependencies, tool configs
├── uv.lock            # Locked dependency versions
├── .flake8            # Flake8 linter config
└── .github/
    └── copilot-instructions.md
```

| File | Role |
|------|------|
| `client.py` | `make_client()`, `resolve_model()`, `stream_reply()`, constants (base URL, system prompt, sampling params) |
| `main.py` | Interactive CLI chat with ANSI colors, streaming output, `--think` flag |
| `app.py` | Flask server at `:5000` — serves web UI, exposes `/chat` SSE endpoint |
| `main_mcp.py` | CLI variant that connects the MLX OpenAI-compatible endpoint to the local stdio MCP server |
| `app_mcp.py` | Flask web variant that resolves MCP tool calls before streaming the final answer to the browser |
| `mcp_server.py` | FastMCP stdio server that exposes mock factory tools such as `get_machine_temperature()` |
| `foundry_local.py` | Hybrid client: boots the Microsoft `foundry-local-sdk` control plane, then streams chat through the MLX server |
| `templates/index.html` | Single-page chat UI with markdown rendering, thinking mode toggle |
| `smoke_test.py` | Validates models endpoint, non-streaming, streaming, and Flask UI |

---

## 🧠 What is MLX?

**[MLX](https://github.com/apple/mlx)** is an open-source machine learning framework created by **Apple**, specifically designed and optimized for **Apple Silicon** (M1/M2/M3/M4) chips.

### Key Features

| Feature | Description |
|---------|-------------|
| **Unified Memory** | Leverages Apple Silicon's shared CPU/GPU memory — no data copying between devices |
| **NumPy-like API** | Familiar Python interface for ML researchers and developers |
| **Lazy Evaluation** | Computations are only materialized when needed, enabling efficient memory use |
| **On-Device Privacy** | All inference runs locally — no data leaves your machine |
| **LLM Optimized** | First-class support for running large language models via `mlx-lm` |

### Why MLX for Local LLMs?

- **No cloud dependency** — works offline, zero API costs
- **Privacy** — factory/industrial data never leaves the device
- **Low latency** — no network round-trip
- **Apple Silicon efficiency** — leverages Neural Engine and unified memory architecture

---

## 🚀 How to Run the MLX Server

The MLX LLM server provides an **OpenAI-compatible API** at `http://127.0.0.1:8080/v1`:

```zsh
# Start the MLX inference server (downloads model on first run)
uvx --from mlx-lm mlx_lm.server
```

The server will:
1. Load the model into unified memory
2. Expose `/v1/models`, `/v1/chat/completions` endpoints
3. Support streaming and non-streaming responses

---

## 🤖 Model: Qwen3.5-9B-MLX-4bit

### Background

**Qwen3.5-9B** is a 9-billion parameter vision-language model developed by **Alibaba Cloud (通义千问)**. Released under the **Apache 2.0** license, it represents a significant leap in open-source LLM capability.

The `mlx-community/Qwen3.5-9B-MLX-4bit` variant is a **4-bit quantized** version optimized for Apple Silicon via MLX.

### Architecture

| Spec | Value |
|------|-------|
| Parameters | 9B |
| Architecture | Gated DeltaNet + Mixture-of-Experts (MoE) |
| Context Length | 262,144 tokens (native) |
| Hidden Dimension | 4,096 |
| Layers | 32 |
| Quantization | 4-bit (group size 64) |
| Disk Size | ~5.6 GB |
| License | Apache 2.0 |

### Key Characteristics

- **Unified Vision-Language**: Early fusion training on multimodal tokens (text, image, video)
- **Efficient Hybrid Architecture**: Gated Delta Networks (linear attention) + sparse MoE for high throughput
- **Thinking Mode**: Built-in chain-of-thought reasoning (`enable_thinking: true`)
- **Multilingual**: 201 languages and dialects
- **Giant Killer**: Outperforms GPT-OSS-120B (a 120B model) in many benchmarks

---

## 📊 Benchmark Results

### Official Qwen3.5 Benchmark Comparison

[![Qwen3.5 Small Model Benchmarks](https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen3.5/Figures/qwen3.5_small_size_score.png)](https://huggingface.co/Qwen/Qwen3.5-9B)

### Language & Reasoning

![Language & Reasoning Benchmarks](img/benchmark-language-reasoning.png)

### Multimodal / Vision

![Multimodal / Vision Benchmarks](img/benchmark-multimodal-vision.png)

> Note: GPT-5-Nano does not report a Video-MME score.

### Math & Coding

![Math & Coding Benchmarks](img/benchmark-math-coding.png)

> Qwen3.5-9B also reports ~97% Mathematics (General) and ~92% Coding (General),
> for which no GPT-OSS-120B comparison value is published.

> **Key Takeaway**: Qwen3.5-9B consistently outperforms models **10–13× its size** in reasoning, vision, and multilingual tasks, making it ideal for efficient on-device deployment.

---

## 💻 Hardware

### Original Model Requirements

![Qwen3.5-9B Memory Footprint by Quantization](img/hardware-memory-by-quantization.png)

> Add ~1 GB per additional 8K context tokens.

### Test Hardware Used

| Component | Spec |
|-----------|------|
| **Machine** | Mac Mini (2024) |
| **Chip** | Apple M4 |
| **Unified Memory** | 16 GB |
| **OS** | macOS Sequoia |
| **Model** | `mlx-community/Qwen3.5-9B-MLX-4bit` |
| **Inference Speed** | ~25–35 tokens/sec |
| **Memory at Inference** | ~6.5 GB (model + KV cache @ 4K context) |

The Mac Mini M4 with 16 GB RAM comfortably runs the 4-bit quantized model with headroom for the OS and other applications.

---

## ⚡ Getting Started

### Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager

### Installation

```zsh
# Clone the repository
git clone https://github.com/bedro96/foundry_local.git
cd foundry_local

# Install dependencies
uv sync
```

### Running

```zsh
# 1. Start the MLX inference server (separate terminal)
uvx --from mlx-lm mlx_lm.server

# 2a. Run the CLI chat client
uv run main.py                  # Thinking mode OFF (default)
uv run main.py --think          # Thinking mode ON
uv run main_mcp.py              # CLI with stdio MCP tools
uv run main_mcp.py --think      # CLI with stdio MCP tools + thinking mode
uv run foundry_local.py         # Hybrid Foundry Local SDK + MLX backend
uv run foundry_local.py --think # Hybrid client with thinking mode ON

# 2b. Or run the Flask web UI
uv run app.py                   # Opens at http://127.0.0.1:5000
uv run app_mcp.py               # Flask UI with stdio MCP tool integration
```

### Smoke Test

```zsh
uv run smoke_test.py            # Tests: models, non-streaming, streaming
uv run smoke_test.py --think    # Also tests streaming with thinking mode ON
```

---

## 🛠 Development

### Dev Commands

```zsh
uv run black main.py app.py app_mcp.py client.py foundry_local.py main_mcp.py mcp_server.py smoke_test.py
uv run flake8 main.py app.py app_mcp.py client.py foundry_local.py main_mcp.py mcp_server.py smoke_test.py
uv run mypy main.py app.py app_mcp.py client.py foundry_local.py main_mcp.py mcp_server.py smoke_test.py
uv run bandit main.py app.py app_mcp.py client.py foundry_local.py main_mcp.py mcp_server.py smoke_test.py
```

### Configuration

- **Line length**: 99 characters
- **Python version**: 3.13+
- **Type checking**: `mypy --strict` (all files)
- **Package manager**: `uv` (not pip)

### Key Constants (in `client.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `BASE_URL` | `http://127.0.0.1:8080/v1` | MLX server endpoint |
| `DEFAULT_MODEL` | `mlx-community/Qwen3.5-9B-MLX-4bit` | Preferred model (fallback to first available) |
| `_THINKING_MAX_TOKENS` | 16,384 | Max tokens with thinking ON |
| `_NO_THINK_MAX_TOKENS` | 8,192 | Max tokens with thinking OFF |

---

## 🔧 Features

- **Multi-turn conversation** — maintains full chat history
- **SSE streaming** — real-time token-by-token response in CLI and web UI
- **Thinking mode** — chain-of-thought reasoning (collapsible in web UI)
- **Markdown rendering** — LLM responses rendered as styled HTML with custom quirk fixes
- **stdio MCP support** — works with a local MCP server over stdio, not just plain chat completions
- **Factory tool calling** — Qwen3.5 on the MLX OpenAI-compatible endpoint successfully calls MCP tools and uses their results in final answers
- **Dark theme** — purpose-built dark UI for factory/industrial context
- **Smart Factory persona** — specialized system prompt for manufacturing operations

---

## 🔌 MCP Integration

This project now includes a fully working **local stdio MCP** integration path for both the CLI and Flask web app.

### MCP Files

| File | Purpose |
|------|---------|
| `mcp_server.py` | FastMCP stdio server exposing mock factory tools |
| `main_mcp.py` | CLI client that performs OpenAI-style tool calling against the MLX server |
| `app_mcp.py` | Flask web UI that resolves MCP tool calls before streaming the final answer |

### Available Tools

- `get_machine_temperature(machine_id)` — returns a mock live temperature reading and status
- `list_machines()` — returns all factory machines, zones, and normal temperature ranges

### Verified Result

The MLX server running `mlx-community/Qwen3.5-9B-MLX-4bit` **works with stdio MCP** through its OpenAI-compatible chat completions API:

1. The app sends MCP tools in OpenAI function-calling format.
2. Qwen3.5 emits structured `tool_calls`.
3. The local MCP stdio server executes the request and returns sensor data.
4. The app streams the final answer back to the user with the tool result incorporated.

### Example

```text
User: What is the temperature of FURNACE-001?
Tool call: get_machine_temperature({"machine_id":"FURNACE-001"})
Tool result: 880.2°C, NORMAL, range 800–950°C
Final answer: The current temperature of FURNACE-001 is 880.2°C and within the normal range.
```

### Test Results

- `main_mcp.py` successfully called both `get_machine_temperature` and `list_machines`
- `app_mcp.py` successfully served `/`, `/models`, and `/chat`
- `/chat` on `app_mcp.py` successfully triggered stdio MCP tool calls and streamed the final response over SSE
- `uv run smoke_test.py` passed against the live MLX API server after the MCP web integration work

---

## 📚 References

- [Apple MLX Framework](https://github.com/apple/mlx)
- [mlx-lm: LLM inference for MLX](https://github.com/ml-explore/mlx-lm)
- [Qwen3.5-9B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B)
- [mlx-community/Qwen3.5-9B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)
- [Qwen3.5 Official Blog](https://qwen.ai/blog?id=qwen3.5)

---

## 📄 License

This project is open source. The Qwen3.5-9B model is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
