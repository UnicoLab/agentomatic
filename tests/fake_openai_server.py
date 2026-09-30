"""A deterministic OpenAI-compatible model server for offline optimization tests.

It plays every role the optimization stack sends to an LLM, recognising each
by its prompt:

* **agent** (``Policy snippets: … Question: …``) — answers *well* (quotes the
  snippets) only when the system prompt asks it to quote policy, and vaguely
  otherwise. A better prompt therefore measurably scores higher, which is
  what makes an optimization run meaningful. A prompt that lists
  ``Q: <question>`` / ``A: <answer>`` pairs is followed literally for those
  exact questions — the memorising prompt an overfitting optimizer produces.
* **judge** (``You are an expert evaluation judge``) — returns the
  ``overall_score`` JSON ``LocalJudgeMetric`` expects, scored by how many
  words of the expected answer the response recalls.
* **rewriter** (asks for an improved system prompt) — returns
  :data:`IMPROVED_PROMPT` after a ``---`` line.
* **augmenter** (``dataset augmentation expert``) — returns a JSON array of
  paraphrased seed queries.

Every request body is recorded on :attr:`FakeOpenAIServer.requests`.
"""

from __future__ import annotations

import itertools
import json
import re
import socket
import threading
import time
from typing import Any

from fastapi import FastAPI, Request

#: The prompt the fake rewriter proposes. Contains the trigger word ("quote").
IMPROVED_PROMPT = (
    "You are a precise Acme Cloud support assistant. Quote the relevant policy "
    "snippet verbatim, then answer in one short sentence. Never invent policy."
)

_VAGUE_ANSWER = "Thanks for reaching out! Our team is happy to help with that."
_AUGMENT_COUNTER = itertools.count(1)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9>:.-]+", text.lower()) if len(w) > 3}


def _section(prompt: str, header: str) -> str:
    match = re.search(rf"## {re.escape(header)}[^\n]*\n(.*?)(?:\n## |\Z)", prompt, re.S)
    return match.group(1).strip() if match else ""


def _agent_reply(system: str, user: str) -> str:
    question = user.rsplit("Question:", 1)[-1].strip()
    memorised = dict(re.findall(r"^Q: (.+?)\s*\nA: (.+?)\s*$", system, re.M))
    if question in memorised:
        return memorised[question]
    snippets = [line[2:].strip() for line in user.splitlines() if line.startswith("- ")]
    if "quote" in system.lower() and snippets:
        return " ".join(snippets)
    return _VAGUE_ANSWER


def _judge_reply(prompt: str) -> str:
    response = _section(prompt, "AI Response")
    # A dataset reference nests its own "## Expected answer" section.
    expected = _section(prompt, "Expected answer") or _section(
        prompt, "Expected / Quality Reference"
    )
    wanted = _words(expected)
    score = round(len(wanted & _words(response)) / len(wanted), 2) if wanted else 0.5
    return json.dumps(
        {
            "overall_score": score,
            "feedback": "Grounded in policy." if score >= 0.5 else "Too vague; cite the policy.",
            "motivation": f"Recalled {score:.0%} of the reference facts.",
            "what_worked": ["polite tone"],
            "what_failed": [] if score >= 0.5 else ["does not state the policy"],
            "improvement_hints": ["Quote the relevant policy snippet verbatim."],
            "dimensions": {"correctness": score, "completeness": score, "relevance": score},
        }
    )


def _augment_reply(prompt: str) -> str:
    """Answer a one-seed augmentation prompt with ``k`` distinct variations."""
    seed_match = re.search(r"## Seed example\n(\{.*?\})\n", prompt, re.S)
    seed = json.loads(seed_match.group(1)) if seed_match else {"query": "question"}
    count = re.search(r"Write (\d+) new", prompt)
    k = int(count.group(1)) if count else 1
    keep = "must stay exactly the seed's expected_answer" in prompt
    query = str(seed.get("query", "")).rstrip("?")
    out = []
    for i in range(k):
        n = next(_AUGMENT_COUNTER)
        out.append(
            {
                "query": f"Variant {n}: {query.lower()}?",
                "expected_answer": seed.get("expected_answer", "")
                if keep
                else f"Generated answer {n}",
            }
        )
    return json.dumps(out)


def respond(messages: list[dict[str, Any]]) -> str:
    """Return the fake model's reply for one chat-completions request."""
    system = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    user = str(messages[-1].get("content", "")) if messages else ""
    text = f"{system}\n{user}"
    if "Policy snippets:" in user:
        return _agent_reply(system, user)
    if "You are an expert evaluation judge" in text:
        return _judge_reply(text)
    if "dataset augmentation expert" in text:
        return _augment_reply(text)
    if "system prompt" in text.lower() and ("improv" in text.lower() or "rewrite" in text.lower()):
        return f"Here is the improved prompt.\n---\n{IMPROVED_PROMPT}"
    if "json" in text.lower():
        return json.dumps({"score": 0.5, "reasoning": "fake"})
    return "OK"


def _build_app(server: FakeOpenAIServer) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"data": [{"id": "fake-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> dict[str, Any]:
        body = await request.json()
        server.requests.append(body)
        content = respond(body.get("messages") or [])
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "fake-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    return app


class FakeOpenAIServer:
    """Run the fake model server on a free local port in a background thread.

    Usage::

        with FakeOpenAIServer() as server:
            base_url = server.base_url  # "http://127.0.0.1:<port>/v1"
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def roles(self) -> dict[str, int]:
        """Count recorded requests per role (agent / judge / rewrite / augment)."""
        counts = {"agent": 0, "judge": 0, "rewrite": 0, "augment": 0, "other": 0}
        for body in self.requests:
            reply = respond(body.get("messages") or [])
            text = json.dumps(body.get("messages") or [])
            if "Policy snippets:" in text:
                counts["agent"] += 1
            elif "expert evaluation judge" in text:
                counts["judge"] += 1
            elif "dataset augmentation expert" in text:
                counts["augment"] += 1
            elif reply.endswith(IMPROVED_PROMPT):
                counts["rewrite"] += 1
            else:
                counts["other"] += 1
        return counts

    def __enter__(self) -> FakeOpenAIServer:
        import uvicorn

        config = uvicorn.Config(
            _build_app(self), host="127.0.0.1", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("fake OpenAI server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
