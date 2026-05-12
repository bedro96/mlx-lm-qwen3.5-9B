"""
Smoke tests against the real MLX server at http://127.0.0.1:8080/v1.

IMPORTANT: This requires the MLX server to be running:
    uvx --from mlx-lm mlx_lm.server

Usage:
    uv run smoke_test.py
    uv run smoke_test.py --think     # also test with thinking mode ON

Exit code 0 = all tests passed. Non-zero = at least one failure.
"""

import argparse
import sys
import urllib.error
import urllib.request

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from client import (
    BASE_URL,
    SYSTEM_PROMPT,
    _NO_THINK_TEMPERATURE,
    _NO_THINK_TOP_K,
    _NO_THINK_TOP_P,
    resolve_model,
    stream_reply,
    stream_reply_with_thinking,
)

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"

PROBE_MESSAGE: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Reply with exactly one sentence about factory safety."},
]
THINKING_PROBE_MESSAGE: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a helpful assistant. Keep reasoning brief."},
    {"role": "user", "content": "Answer with exactly OK."},
]


def _pass(name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  {GREEN}✓ PASS{RESET}  {name}{suffix}")


def _fail(name: str, detail: str) -> None:
    print(f"  {RED}✗ FAIL{RESET}  {name} — {detail}")


def test_models(client: OpenAI) -> bool:
    """GET /v1/models returns at least one model."""
    name = "GET /v1/models"
    try:
        models = client.models.list()
        ids = [m.id for m in models.data if getattr(m, "id", None)]
        if not ids:
            _fail(name, "no models returned")
            return False
        _pass(name, f"found: {ids[0]!r}")
        return True
    except Exception as exc:
        _fail(name, str(exc))
        return False


def test_non_streaming(client: OpenAI, model: str) -> bool:
    """POST /v1/chat/completions (stream=False) returns non-empty content."""
    name = "POST /v1/chat/completions (non-streaming)"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=PROBE_MESSAGE,
            stream=False,
            max_tokens=80,
            temperature=_NO_THINK_TEMPERATURE,
            top_p=_NO_THINK_TOP_P,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": _NO_THINK_TOP_K,
            },
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            _fail(name, "empty response content")
            return False
        _pass(name, f"{len(content)} chars received")
        return True
    except Exception as exc:
        _fail(name, str(exc))
        return False


def test_streaming(client: OpenAI, model: str, enable_thinking: bool = False) -> bool:
    """POST /v1/chat/completions (stream=True) yields non-empty answer tokens."""
    label = "thinking=ON" if enable_thinking else "thinking=OFF"
    name = f"POST /v1/chat/completions (streaming, {label})"
    try:
        answer_tokens: list[str] = []

        if enable_thinking:
            thinking_chunks = 0
            stream = stream_reply_with_thinking(client, model, THINKING_PROBE_MESSAGE)
            try:
                for kind, token in stream:
                    if kind == "thinking":
                        thinking_chunks += 1
                        continue
                    answer_tokens.append(token)
                    if "".join(answer_tokens).strip():
                        break
            finally:
                stream.close()

            full = "".join(answer_tokens).strip()
            if not full:
                _fail(name, "stream yielded no answer tokens")
                return False
            _pass(name, f"{thinking_chunks} thinking chunks before answer")
            return True

        for token in stream_reply(client, model, PROBE_MESSAGE, enable_thinking=False):
            answer_tokens.append(token)

        full = "".join(answer_tokens).strip()
        if not full:
            _fail(name, "stream yielded no content tokens")
            return False
        _pass(name, f"{len(answer_tokens)} chunks, {len(full)} chars")
        return True
    except Exception as exc:
        _fail(name, str(exc))
        return False


def test_flask_ui() -> bool:
    """GET http://127.0.0.1:5000 returns 200 (Flask UI must be running)."""
    name = "GET http://127.0.0.1:5000 (Flask UI)"
    try:
        with urllib.request.urlopen("http://127.0.0.1:5000", timeout=3) as resp:  # nosec B310
            if resp.status == 200:
                _pass(name)
                return True
            _fail(name, f"HTTP {resp.status}")
            return False
    except urllib.error.URLError as exc:
        print(f"  {YELLOW}⚠ SKIP{RESET}  {name} — Flask not running ({exc.reason})")
        return True  # not a hard failure; Flask is optional during CI
    except Exception as exc:
        print(f"  {YELLOW}⚠ SKIP{RESET}  {name} — Flask not running ({exc})")
        return True  # not a hard failure; Flask is optional during CI


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke tests for the MLX server")
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Also run the streaming test with thinking mode ON",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}Smoke tests → {BASE_URL}{RESET}\n")

    client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    results: list[bool] = []

    # 1. Models endpoint
    ok = test_models(client)
    results.append(ok)

    # Resolve model for subsequent tests
    model = None
    if ok:
        try:
            model = resolve_model(client)
        except RuntimeError as exc:
            print(f"  {RED}Cannot resolve model: {exc}{RESET}")

    # 2. Non-streaming
    if model:
        results.append(test_non_streaming(client, model))

        # 3. Streaming — thinking OFF
        results.append(test_streaming(client, model, enable_thinking=False))

        # 4. Streaming — thinking ON (optional)
        if args.think:
            results.append(test_streaming(client, model, enable_thinking=True))

    # 5. Flask UI (skipped gracefully if not running)
    results.append(test_flask_ui())

    passed = sum(results)
    total = len(results)
    color = GREEN if passed == total else RED
    print(f"\n{color}{BOLD}{passed}/{total} tests passed.{RESET}\n")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
