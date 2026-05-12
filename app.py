"""
Flask web UI for the Smart Factory Agent.

Start the server:
    uv run app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import json
from collections.abc import Generator
from typing import Any, cast

from flask import Flask, Response, render_template, request, stream_with_context
from openai.types.chat import ChatCompletionMessageParam

from client import (
    SYSTEM_PROMPT,
    make_client,
    resolve_model,
    stream_reply,
    stream_reply_with_thinking,
)

app = Flask(__name__)
_client = make_client()


@app.route("/")
def index() -> str:
    return render_template("index.html", system_prompt=SYSTEM_PROMPT)


@app.route("/models")
def models() -> Response:
    model = resolve_model(_client)
    return Response(json.dumps({"model": model}), content_type="application/json")


@app.route("/chat", methods=["POST"])
def chat() -> Response:
    """SSE endpoint: streams reply tokens to the browser."""
    payload = request.get_json(force=True)
    data = payload if isinstance(payload, dict) else {}
    raw_messages = data.get("messages")
    messages = cast(list[ChatCompletionMessageParam], raw_messages)
    if not isinstance(raw_messages, list):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    enable_thinking = bool(data.get("enable_thinking", False))
    model = resolve_model(_client)

    def generate() -> Generator[str, None, None]:
        try:
            if enable_thinking:
                for kind, token in stream_reply_with_thinking(_client, model, messages):
                    payload_out: dict[str, Any] = {"kind": kind, "token": token}
                    yield f"data: {json.dumps(payload_out)}\n\n"
            else:
                for token in stream_reply(_client, model, messages, enable_thinking=False):
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
    app.run(debug=False, threaded=True)
