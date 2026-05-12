"""
Foundry Local orchestration + MLX LLM backend via OpenAI-compatible API.

Usage:
    uv run main.py           # default: thinking mode OFF
    uv run main.py --think   # enable Qwen3 chain-of-thought thinking
"""

import argparse

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from client import (
    SYSTEM_PROMPT,
    make_client,
    resolve_model,
    stream_reply,
    stream_reply_with_thinking,
)

# ANSI color codes
ORANGE = "\033[38;5;208m"
WHITE = "\033[97m"
GREEN = "\033[32m"
DIM_GRAY = "\033[2;37m"
RESET = "\033[0m"


def create_completion(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    enable_thinking: bool = False,
) -> str:
    """Stream a completion from the MLX server, printing tokens as they arrive."""
    answer_tokens: list[str] = []

    if enable_thinking:
        thinking_started = False
        answer_started = False

        for kind, token in stream_reply_with_thinking(client, model, messages):
            if kind == "thinking":
                if not thinking_started:
                    print(f"{DIM_GRAY}Thinking > ", end="", flush=True)
                    thinking_started = True
                print(token, end="", flush=True)
                continue

            if thinking_started and not answer_started:
                print(RESET)

            if not answer_started:
                print(f"{GREEN}Agent > ", end="", flush=True)
                answer_started = True

            print(token, end="", flush=True)
            answer_tokens.append(token)
    else:
        print(f"{GREEN}Agent > ", end="", flush=True)
        for token in stream_reply(client, model, messages, enable_thinking=False):
            print(token, end="", flush=True)
            answer_tokens.append(token)

    full_reply = "".join(answer_tokens).strip()
    print(RESET)

    if not full_reply:
        raise RuntimeError(
            f"Model '{model}' returned an empty response. "
            "If thinking mode is on, the model may have only produced reasoning "
            "tokens. Try toggling --think or check the server logs."
        )

    return full_reply


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Factory Agent CLI")
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Enable Qwen3 chain-of-thought thinking mode (slower, higher quality)",
    )
    args = parser.parse_args()

    client = make_client()
    model = resolve_model(client)

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    think_status = "thinking ON" if args.think else "thinking OFF"
    print(
        f"{GREEN}Smart Factory Agent ready. "
        f"Model: {model} | {think_status}. Type 'exit' to quit.{RESET}\n"
    )

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
            full_reply = create_completion(client, model, messages, enable_thinking=args.think)
            messages.append({"role": "assistant", "content": full_reply})

        except Exception as exc:
            print(f"{RESET}\n\033[31mError: {exc}\033[0m")


if __name__ == "__main__":
    main()
