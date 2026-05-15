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
import threading
import time
from collections.abc import Generator
from concurrent.futures import Future
from typing import Any, cast

import httpx
from flask import Flask, Response, render_template, request, stream_with_context
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai.types.chat import ChatCompletionMessageParam

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
_NO_THINK_MAX_TOKENS = 10240

_TOOL_PROMPT_SUFFIX = (
    "You have access to real-time factory sensor tools AND web browsing tools. "
    "Use get_machine_temperature or list_machines for factory sensor data. "
    "Use web_search to search the internet for up-to-date information, "
    "and fetch_page to read the full content of a specific URL."
)
_TOOL_AWARE_SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n\n{_TOOL_PROMPT_SUFFIX}"
_MCP_FACTORY_PARAMS = StdioServerParameters(command="uv", args=["run", "mcp_server.py"])
_MCP_WEB_PARAMS = StdioServerParameters(command="uv", args=["run", "mcp_web_server.py"])


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


class _MCPManager:
    """Manages persistent stdio connections to both MCP servers.

    Starts a dedicated asyncio event loop in a background thread so MCP
    sessions remain open for the lifetime of the Flask process.  Flask
    request handlers call the synchronous ``call_tool`` / ``get_tools``
    methods; the heavy async lifting happens inside the background loop.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-manager")
        self._tools: list[dict[str, Any]] = []
        self._tool_sessions: dict[str, ClientSession] = {}
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None

    # ------------------------------------------------------------------
    # Public interface (called from Flask threads)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background loop and block until both MCP servers are ready."""
        self._thread.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("MCP servers did not become ready within 60 s.")
        if self._error:
            raise self._error

    def get_tools(self) -> list[dict[str, Any]]:
        """Return the combined OpenAI-format tool list (empty if not started)."""
        return self._tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Synchronously execute an MCP tool and return its text result."""
        fut: Future[str] = Future()
        asyncio.run_coroutine_threadsafe(self._call_tool_async(name, arguments, fut), self._loop)
        return fut.result(timeout=60)

    # ------------------------------------------------------------------
    # Internal async implementation (runs inside _loop)
    # ------------------------------------------------------------------

    async def _call_tool_async(self, name: str, arguments: dict[str, Any], fut: Future[str]) -> None:
        try:
            session = self._tool_sessions.get(name)
            if session is None:
                fut.set_result(json.dumps({"error": f"No MCP session for tool '{name}'"}))
                return
            result = await session.call_tool(name, arguments)
            texts = [
                block.text
                for block in result.content
                if isinstance(getattr(block, "text", None), str)
            ]
            fut.set_result("\n".join(texts))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with (
                stdio_client(_MCP_FACTORY_PARAMS) as (r_f, w_f),
                stdio_client(_MCP_WEB_PARAMS) as (r_w, w_w),
            ):
                async with (
                    ClientSession(r_f, w_f) as factory_session,
                    ClientSession(r_w, w_w) as web_session,
                ):
                    await factory_session.initialize()
                    await web_session.initialize()

                    ft = await factory_session.list_tools()
                    wt = await web_session.list_tools()

                    self._tools = mcp_tools_to_openai(ft.tools + wt.tools)
                    for t in ft.tools:
                        self._tool_sessions[t.name] = factory_session
                    for t in wt.tools:
                        self._tool_sessions[t.name] = web_session

                    self._ready.set()  # unblock Flask startup
                    await self._stop.wait()  # keep sessions alive until shutdown
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()


_mcp = _MCPManager()


def _prepare_messages_for_streaming(
    model: str,
    messages: list[ChatCompletionMessageParam],
    enable_thinking: bool,
) -> list[ChatCompletionMessageParam]:
    """Run the tool-calling loop (sync) and return the prepared message list."""
    prepared_messages = _with_tool_system_prompt(messages)
    openai_tools = _mcp.get_tools()

    for _ in range(5):
        response = _client.chat.completions.create(
            **_build_request_args(model, prepared_messages, openai_tools, enable_thinking)
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return prepared_messages

        prepared_messages.append(cast(ChatCompletionMessageParam, message.to_dict()))
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid tool arguments for {tool_call.function.name}: "
                    f"{tool_call.function.arguments}"
                ) from exc
            tool_result = _mcp.call_tool(tool_call.function.name, arguments)
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
    tool_names = [t["function"]["name"] for t in _mcp.get_tools()]
    return Response(
        json.dumps({"model": model, "mcp_transport": "stdio", "tools": tool_names}),
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
            prepared_messages = _prepare_messages_for_streaming(model, messages, enable_thinking)

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
    _mcp.start()
    app.run(debug=False, threaded=True)
