"""Shared OpenAI client utilities for the local MLX server."""

from collections.abc import Generator
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

SYSTEM_PROMPT = (
    "You are a stable smart factory agent. "
    "You identify factory problems, analyze root causes, and present the best "
    "practical solutions for production, quality, safety, maintenance, IoT, "
    "workforce, and manufacturing operations."
)

# Recommended sampling params per Qwen3 official docs.
# top_k is a local-server extension; passed inside extra_body.
_THINKING_TEMPERATURE = 0.6
_THINKING_TOP_P = 0.95
_THINKING_TOP_K = 20
_THINKING_MAX_TOKENS = 16384
_NO_THINK_TEMPERATURE = 0.7
_NO_THINK_TOP_P = 0.8
_NO_THINK_TOP_K = 20
_NO_THINK_MAX_TOKENS = 8192


def make_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key="not-needed")


def resolve_model(client: OpenAI) -> str:
    try:
        models = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve models from the local MLX server. "
            f"Check that the server is running at {BASE_URL}."
        ) from exc

    model_items = list(models)
    if not model_items:
        raise RuntimeError("The local MLX server returned no models.")

    available_model_ids = [str(item.id) for item in model_items if item.id]
    if not available_model_ids:
        raise RuntimeError("The local MLX server returned models without IDs.")

    if DEFAULT_MODEL in available_model_ids:
        return DEFAULT_MODEL

    return available_model_ids[0]


def _extract_reasoning_token(delta: Any) -> str:
    model_extra = getattr(delta, "model_extra", None)
    if not isinstance(model_extra, dict):
        return ""

    reasoning = model_extra.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def _stream_reply_events(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    enable_thinking: bool,
) -> Generator[tuple[str, str], None, None]:
    temperature = _THINKING_TEMPERATURE if enable_thinking else _NO_THINK_TEMPERATURE
    top_p = _THINKING_TOP_P if enable_thinking else _NO_THINK_TOP_P
    top_k = _THINKING_TOP_K if enable_thinking else _NO_THINK_TOP_K

    extra_body: dict[str, object] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "top_k": top_k,
    }
    request_args: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "extra_body": extra_body,
    }
    if enable_thinking:
        request_args["max_tokens"] = _THINKING_MAX_TOKENS
    else:
        request_args["max_tokens"] = _NO_THINK_MAX_TOKENS

    stream = client.chat.completions.create(**request_args)

    try:
        for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning = _extract_reasoning_token(delta)
            if reasoning:
                yield ("thinking", reasoning)

            content = delta.content
            if content:
                yield ("answer", content)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def stream_reply(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    enable_thinking: bool = False,
) -> Generator[str, None, None]:
    """Yield only final answer tokens from the MLX server."""
    thinking_tokens: list[str] = []
    for kind, token in _stream_reply_events(client, model, messages, enable_thinking):
        if kind == "thinking":
            thinking_tokens.append(token)
            continue
        yield token


def stream_reply_with_thinking(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
) -> Generator[tuple[str, str], None, None]:
    """Yield streamed tokens tagged as either thinking or final answer."""
    yield from _stream_reply_events(client, model, messages, enable_thinking=True)
