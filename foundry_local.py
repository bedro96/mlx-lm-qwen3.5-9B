"""
Foundry Local Agent: hybrid Foundry Local SDK manager + MLX OpenAI backend.

Boots Microsoft's `foundry-local-sdk` to start the Foundry Local control-plane
service (and surface its bound URL), but routes the actual streaming chat
through the existing MLX server at http://127.0.0.1:8080/v1.

Usage:
    uv run foundry_local.py           # default: thinking mode OFF
    uv run foundry_local.py --think   # enable Qwen3 chain-of-thought thinking
"""

import argparse
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from client import (
    SYSTEM_PROMPT,
    make_client,
    resolve_model,
    stream_reply,
    stream_reply_with_thinking,
)

ORANGE = "\033[38;5;208m"
WHITE = "\033[97m"
GREEN = "\033[32m"
DIM_GRAY = "\033[2;37m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


class FoundryLocalHandle:
    """Lightweight wrapper around FoundryLocalManager singleton lifecycle."""

    def __init__(self, app_name: str = "SmartFactoryAgent") -> None:
        self.app_name = app_name
        self.urls: list[str] = []
        self.catalog_size: int = 0
        self.cached_aliases: list[str] = []
        self.startup_error: Optional[str] = None
        self._started = False

    def start(self) -> None:
        try:
            from foundry_local_sdk import (  # type: ignore[import-untyped]
                Configuration,
                FoundryLocalManager,
            )
            from foundry_local_sdk.configuration import (  # type: ignore[import-untyped]
                LogLevel,
            )
        except Exception as exc:  # pragma: no cover - import failure path
            self.startup_error = f"foundry-local-sdk import failed: {exc}"
            return

        try:
            config = Configuration(app_name=self.app_name, log_level=LogLevel.WARNING)
            FoundryLocalManager.initialize(config)
            mgr = FoundryLocalManager.instance
            mgr.start_web_service()
            self.urls = list(mgr.urls or [])
            self.catalog_size = len(mgr.catalog.list_models())
            self.cached_aliases = [m.alias for m in mgr.catalog.get_cached_models()]
            self._started = True
        except Exception as exc:
            self.startup_error = f"FoundryLocalManager start failed: {exc}"

    def stop(self) -> None:
        if not self._started:
            return
        try:
            from foundry_local_sdk import FoundryLocalManager

            FoundryLocalManager.instance.stop_web_service()
        except Exception:
            pass
        self._started = False


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
                print(f"{GREEN}Foundry Local Agent > ", end="", flush=True)
                answer_started = True

            print(token, end="", flush=True)
            answer_tokens.append(token)
    else:
        print(f"{GREEN}Foundry Local Agent > ", end="", flush=True)
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


def print_foundry_status(handle: FoundryLocalHandle) -> None:
    if handle.startup_error:
        print(
            f"{YELLOW}Foundry Local Service > unavailable "
            f"({handle.startup_error}). Continuing with MLX backend only.{RESET}"
        )
        return

    endpoint = handle.urls[0] if handle.urls else "<no web service>"
    print(
        f"{GREEN}Foundry Local Service > endpoint={endpoint} | "
        f"catalog={handle.catalog_size} models | cached={len(handle.cached_aliases)}{RESET}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundry Local Agent CLI")
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Enable Qwen3 chain-of-thought thinking mode (slower, higher quality)",
    )
    args = parser.parse_args()

    handle = FoundryLocalHandle()
    handle.start()
    print_foundry_status(handle)

    try:
        client = make_client()
        model = resolve_model(client)

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        think_status = "thinking ON" if args.think else "thinking OFF"
        print(
            f"{GREEN}Foundry Local Agent ready. Backend: MLX | "
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
                print(f"{RESET}\n{RED}Error: {exc}{RESET}")
    finally:
        handle.stop()


if __name__ == "__main__":
    main()
