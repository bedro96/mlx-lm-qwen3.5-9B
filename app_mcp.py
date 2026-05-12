"""
Flask web UI for the Smart Factory Agent with stdio MCP tool integration.

Start the server:
    uv run app_mcp.py

Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

import asyncio
import json
import subprocess  # nosec B404 - fixed local process launch for the MLX server
import time
from collections.abc import Generator
from typing import Any, cast

import httpx
from flask import Flask, Response, render_template, request, stream_with_context
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall

from client import (
    BASE_URL,
    SYSTEM_PROMPT,
    make_client,
    resolve_model,
    stream_reply,
    stream_reply_with_thinking,
)

app = Flask(__name__)
_client = make_client()

_THINKING_TEMPERATURE = 0.6
_THINKING_TOP_P = 0.95
_THINKING_TOP_K = 20
_THINKING_MAX_TOKENS = 16384
_NO_THINK_TEMPERATURE = 0.7
_NO_THINK_TOP_P = 0.8
_NO_THINK_TOP_K = 20
_NO_THINK_MAX_TOKENS = 8192

_TOOL_PROMPT_SUFFIX = (
    "You have access to real-time factory sensor tools. "
    "When a user asks about machine temperatures or factory conditions, "
    "use the available tools to get actual sensor data before answering. "
    "Always call get_machine_temperature or list_machines when relevant."
)
_TOOL_AWARE_SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n\n{_TOOL_PROMPT_SUFFIX}"
_MCP_SERVER_PARAMS = StdioServerParameters(command="uv", args=["run", "mcp_server.py"])


def _server_is_alive() -> bool:
    try:
        response = httpx.get(f"{BASE_URL}/models", timeout=3)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_mlx_server() -> None:
    """Start the MLX server if it is not already running."""
    if _server_is_alive():
        return

    subprocess.Popen(  # nosec
        ["uvx", "--from", "mlx-lm", "mlx_lm.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(120):
        time.sleep(1)
        if _server_is_alive():
            return

    raise RuntimeError(f"Could not start the local MLX server at {BASE_URL}.")


def mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    openai_tools: list[dict[str, Any]] = []
    for tool in mcp_tools:
        schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return openai_tools


def _build_request_args(
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[dict[str, Any]],
    enable_thinking: bool,
) -> dict[str, Any]:
    temperature = _THINKING_TEMPERATURE if enable_thinking else _NO_THINK_TEMPERATURE
    top_p = _THINKING_TOP_P if enable_thinking else _NO_THINK_TOP_P
    top_k = _THINKING_TOP_K if enable_thinking else _NO_THINK_TOP_K
    max_tokens = _THINKING_MAX_TOKENS if enable_thinking else _NO_THINK_MAX_TOKENS

    extra_body: dict[str, object] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "top_k": top_k,
    }

    args: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "extra_body": extra_body,
    }
    if tools:
        args["tools"] = tools
        args["tool_choice"] = "auto"
    return args


def _with_tool_system_prompt(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    if not messages:
        return [
            cast(
                ChatCompletionMessageParam,
                {"role": "system", "content": _TOOL_AWARE_SYSTEM_PROMPT},
            )
        ]

    prepared = list(messages)
    first_message = cast(dict[str, Any], prepared[0])
    if first_message.get("role") == "system":
        first_content = first_message.get("content")
        if isinstance(first_content, str) and _TOOL_PROMPT_SUFFIX not in first_content:
            prepared[0] = cast(
                ChatCompletionMessageParam,
                {"role": "system", "content": f"{first_content}\n\n{_TOOL_PROMPT_SUFFIX}"},
            )
        return prepared

    return [
        cast(ChatCompletionMessageParam, {"role": "system", "content": _TOOL_AWARE_SYSTEM_PROMPT}),
        *prepared,
    ]


async def _run_tool_call(session: ClientSession, tool_call: ChatCompletionMessageToolCall) -> str:
    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid tool arguments for {name}: {tool_call.function.arguments}"
        ) from exc

    result = await session.call_tool(name, arguments)
    texts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


async def _prepare_messages_for_streaming(
    model: str,
    messages: list[ChatCompletionMessageParam],
    enable_thinking: bool,
) -> list[ChatCompletionMessageParam]:
    prepared_messages = _with_tool_system_prompt(messages)

    async with stdio_client(_MCP_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_list = await session.list_tools()
            openai_tools = mcp_tools_to_openai(tool_list.tools)

            for _ in range(5):
                response = _client.chat.completions.create(
                    **_build_request_args(model, prepared_messages, openai_tools, enable_thinking)
                )
                message = response.choices[0].message

                if not message.tool_calls:
                    return prepared_messages

                prepared_messages.append(cast(ChatCompletionMessageParam, message.to_dict()))
                for tool_call in message.tool_calls:
                    tool_result = await _run_tool_call(session, tool_call)
                    prepared_messages.append(
                        cast(
                            ChatCompletionMessageParam,
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result,
                            },
                        )
                    )

    raise RuntimeError("Tool-calling loop exceeded the maximum number of rounds.")


@app.route("/")
def index() -> str:
    return render_template("index.html", system_prompt=_TOOL_AWARE_SYSTEM_PROMPT)


@app.route("/models")
def models() -> Response:
    ensure_mlx_server()
    model = resolve_model(_client)
    return Response(
        json.dumps({"model": model, "mcp_transport": "stdio"}),
        content_type="application/json",
    )


@app.route("/chat", methods=["POST"])
def chat() -> Response:
    """SSE endpoint: streams reply tokens to the browser after optional MCP tool calls."""
    payload = request.get_json(force=True)
    data = payload if isinstance(payload, dict) else {}
    raw_messages = data.get("messages")
    messages = cast(list[ChatCompletionMessageParam], raw_messages)
    if not isinstance(raw_messages, list):
        messages = [
            cast(
                ChatCompletionMessageParam,
                {"role": "system", "content": _TOOL_AWARE_SYSTEM_PROMPT},
            )
        ]

    enable_thinking = bool(data.get("enable_thinking", False))

    def generate() -> Generator[str, None, None]:
        try:
            ensure_mlx_server()
            model = resolve_model(_client)
            prepared_messages = asyncio.run(
                _prepare_messages_for_streaming(model, messages, enable_thinking)
            )

            if enable_thinking:
                for kind, token in stream_reply_with_thinking(_client, model, prepared_messages):
                    payload_out: dict[str, Any] = {"kind": kind, "token": token}
                    yield f"data: {json.dumps(payload_out)}\n\n"
            else:
                for token in stream_reply(
                    _client, model, prepared_messages, enable_thinking=False
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    ensure_mlx_server()
    app.run(debug=False, threaded=True)
