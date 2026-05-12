"""
Smart Factory Agent CLI with MCP tool integration.

This client connects to:
  1. A local MCP stdio server (mcp_server.py) for factory sensor data
  2. The local MLX LLM server for chat completions with tool calling

The LLM can autonomously invoke MCP tools (e.g. get_machine_temperature)
when answering user queries about factory conditions.

Usage:
    uv run main_mcp.py           # thinking mode OFF (default)
    uv run main_mcp.py --think   # thinking mode ON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess  # nosec B404 - fixed local process launch for the MLX server
import sys
import time
from typing import Any, cast

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall

from client import (
    BASE_URL,
    SYSTEM_PROMPT,
    make_client,
    resolve_model,
)

# ANSI color codes
ORANGE = "\033[38;5;208m"
WHITE = "\033[97m"
GREEN = "\033[32m"
CYAN = "\033[36m"
DIM_GRAY = "\033[2;37m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# Sampling params (duplicated from client.py to keep this file self-contained for tool calls)
_THINKING_TEMPERATURE = 0.6
_THINKING_TOP_P = 0.95
_THINKING_TOP_K = 20
_THINKING_MAX_TOKENS = 16384
_NO_THINK_TEMPERATURE = 0.7
_NO_THINK_TOP_P = 0.8
_NO_THINK_TOP_K = 20
_NO_THINK_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# MLX server health check & auto-start
# ---------------------------------------------------------------------------


def _server_is_alive() -> bool:
    """Check if the MLX server is responding."""
    try:
        resp = httpx.get(f"{BASE_URL}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_mlx_server() -> None:
    """Start the MLX server if it is not already running."""
    if _server_is_alive():
        return

    print(f"{YELLOW}MLX server not responding at {BASE_URL}. Starting it…{RESET}")
    subprocess.Popen(  # nosec
        ["uvx", "--from", "mlx-lm", "mlx_lm.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 120 seconds for the server to be ready.
    for i in range(120):
        time.sleep(1)
        if _server_is_alive():
            print(f"{GREEN}MLX server started successfully.{RESET}")
            return
        if i % 10 == 9:
            print(f"{DIM_GRAY}  …still waiting ({i + 1}s){RESET}")

    print("\033[31mFailed to start MLX server after 120 s. Exiting.\033[0m")
    sys.exit(1)


# ---------------------------------------------------------------------------
# MCP tool ↔ OpenAI tool conversion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM completion with tool-calling loop
# ---------------------------------------------------------------------------


def _build_request_args(
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[dict[str, Any]],
    enable_thinking: bool,
) -> dict[str, Any]:
    """Build the kwargs dict for client.chat.completions.create()."""
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


async def run_tool_call(session: ClientSession, tool_call: ChatCompletionMessageToolCall) -> str:
    """Execute a single tool call via the MCP session and return the result as a string."""
    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON arguments: {tool_call.function.arguments}"})

    print(f"{CYAN}  🔧 Calling tool: {name}({json.dumps(arguments, ensure_ascii=False)}){RESET}")

    result = await session.call_tool(name, arguments)
    # MCP returns a list of content blocks; concatenate text content.
    texts = [block.text for block in result.content if hasattr(block, "text")]
    return "\n".join(texts)


async def chat_completion_with_tools(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[dict[str, Any]],
    session: ClientSession,
    enable_thinking: bool,
) -> str:
    """Run a chat completion and handle tool calls until the model returns final text."""
    max_rounds = 5  # safety limit on tool-call rounds

    for _ in range(max_rounds):
        args = _build_request_args(model, messages, tools, enable_thinking)

        # Non-streaming call so we can inspect tool_calls cleanly.
        response = client.chat.completions.create(**args)
        choice = response.choices[0]
        msg = choice.message

        # If there are tool calls, execute them and continue the loop.
        if msg.tool_calls:
            # Append assistant message with tool calls.
            messages.append(cast(ChatCompletionMessageParam, msg.to_dict()))

            for tc in msg.tool_calls:
                tool_result = await run_tool_call(session, tc)
                messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        },
                    )
                )

            continue  # re-call LLM with tool results

        # No tool calls — we have a final answer.
        answer = msg.content or ""

        # Stream the final answer for a nicer UX by re-calling with stream=True.
        if answer:
            print(f"{GREEN}Agent > ", end="", flush=True)
            print(answer, end="", flush=True)
            print(RESET)
            return answer.strip()

    return "(No response after maximum tool-call rounds)"


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def async_main(enable_thinking: bool) -> None:
    """Main chat loop with MCP + LLM integration."""

    # 1. Ensure MLX server is running.
    ensure_mlx_server()
    llm_client = make_client()
    model = resolve_model(llm_client)

    # 2. Start MCP server and connect.
    server_params = StdioServerParameters(command="uv", args=["run", "mcp_server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 3. Discover MCP tools.
            tool_list = await session.list_tools()
            openai_tools = mcp_tools_to_openai(tool_list.tools)
            tool_names = [t["function"]["name"] for t in openai_tools]

            think_status = "thinking ON" if enable_thinking else "thinking OFF"
            print(
                f"{GREEN}Smart Factory Agent ready (MCP mode).\n"
                f"  Model: {model} | {think_status}\n"
                f"  MCP tools: {', '.join(tool_names)}\n"
                f"  Type 'exit' to quit.{RESET}\n"
            )

            # Enhanced system prompt with tool awareness.
            system_prompt = (
                SYSTEM_PROMPT + "\n\nYou have access to real-time factory sensor tools. "
                "When a user asks about machine temperatures or factory conditions, "
                "use the available tools to get actual sensor data before answering. "
                "Always call get_machine_temperature or list_machines when relevant."
            )

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
            ]

            # 4. Chat loop.
            while True:
                try:
                    user_input = input(f"{ORANGE}You > {WHITE}")
                except (EOFError, KeyboardInterrupt):
                    print(f"\n{GREEN}Goodbye!{RESET}")
                    break

                print(RESET, end="")

                if user_input.strip().lower() in {"exit", "quit"}:
                    print(f"{GREEN}Goodbye!{RESET}")
                    break

                if not user_input.strip():
                    continue

                messages.append({"role": "user", "content": user_input})

                try:
                    reply = await chat_completion_with_tools(
                        llm_client, model, messages, openai_tools, session, enable_thinking
                    )
                    messages.append({"role": "assistant", "content": reply})
                except Exception as exc:
                    print(f"{RESET}\n\033[31mError: {exc}\033[0m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Factory Agent CLI (MCP mode)")
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Enable Qwen3 chain-of-thought thinking mode",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args.think))


if __name__ == "__main__":
    main()
