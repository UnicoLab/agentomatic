"""A local OpenAI-compatible server that stands in for a small instruct model.

**This is a test double, not a language model.** It generates nothing; it
follows a fixed set of rules. Point the live optimization suites at a real
local model (oMLX, llama.cpp, vLLM, LM Studio, Ollama) whenever you have one —
they speak the same protocol and need no changes here::

    export OMLX_BASE_URL=http://127.0.0.1:8000/v1
    export OMLX_API_KEY=…
    export AGENTOMATIC_LIVE_MODEL=omlx/your-model
    uv run pytest tests/test_live_omlx_optimize.py tests/test_live_omlx_keras_optimize.py \
        -q --override-ini='addopts='

Why it exists: those suites skip entirely when no OpenAI-compatible endpoint
is reachable, which leaves the whole ``omlx/`` provider path, the prompt
fitter, and the Keras-style ``fit()`` loop unexercised on any machine without
a local model — including CI. Running them against this server exercises all
of it deterministically.

What makes it a valid optimization *target* rather than a constant function:
answer quality genuinely depends on the system prompt. Each directive a prompt
carries ("answer as JSON", "cite your source", "state your confidence", or a
required marker token) makes the response satisfy one more property the metric
rewards, so a better prompt scores strictly higher. An optimizer that really
searches and selects will climb; one that does not, will not.

It also plays the two other roles the fitter needs. As the **rewriter** it
reads the optimization briefing — current prompt, failing I/O, expected
answers, judge guidance — and folds the missing required tokens into a new
prompt, which is the move a real rewrite model makes. As the **judge** it
returns the exact schema the metric asked for, scoring the same properties the
metric does so the two signals agree.

What it cannot tell you: whether a real model writes *good* prompts. It proves
the machinery — search, evaluation, selection, early stopping, checkpointing,
config application — not the quality of a real model's language.

Run it::

    uv run python scripts/local_slm_server.py --port 8000
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="stub-slm", version="1.0.0")

MODEL_ID = "stub-slm-1b-instruct"

#: Directives the responder understands, and the behaviour each unlocks.
DIRECTIVES: dict[str, tuple[str, ...]] = {
    "json": ("json", "structured output", "as json", "json object"),
    "cite": ("cite", "citation", "source", "reference"),
    "concise": ("concise", "brief", "short", "succinct"),
    "steps": ("step by step", "step-by-step", "reasoning", "explain how"),
    "confidence": ("confidence", "certainty", "how sure"),
}

#: Ground-truth answers the responder knows, keyed by a term in the question.
FACTS: dict[str, str] = {
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of brazil": "Brasilia",
    "largest ocean": "the Pacific Ocean",
    "speed of light": "299792458 metres per second",
    "boiling point": "100 degrees Celsius at sea level",
}


def _seed(text: str) -> int:
    """Deterministic seed so identical requests give identical answers."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _active_directives(system_prompt: str) -> set[str]:
    """Return the directive names the system prompt asks for."""
    low = system_prompt.lower()
    return {name for name, words in DIRECTIVES.items() if any(w in low for w in words)}


def _lookup(question: str) -> str | None:
    """Return the known answer for *question*, if the responder knows one."""
    low = question.lower()
    for key, answer in FACTS.items():
        if key in low:
            return answer
    return None


#: Words too common to be a meaningful optimization target.
_STOPWORDS = frozenset(
    """a an the and or of to in on for with is are be as at by from that this it
    answer response output result query question expected actual""".split()
)


def _expected_keywords(briefing: str) -> list[str]:
    """Mine the required tokens out of an optimization briefing.

    A briefing states ground truth in several places, and a competent rewrite
    model reads all of them:

    * ``Expected:`` / ``## Expected answer`` — the target answer itself.
    * Judge guidance naming marker tokens the ideal answer contains.
    * Metric feedback of the form ``missing the required marker token 'x'``.

    Args:
        briefing: The full rewrite or mutation prompt sent by the fitter.

    Returns:
        Distinctive required tokens, most frequently demanded first.
    """
    sources: list[str] = []
    sources += re.findall(r"^\s*[-*]?\s*Expected(?:\s+answer)?:\s*(.+)$", briefing, re.MULTILINE)
    sources += re.findall(r"^##\s*Expected answer\s*\n(.+)$", briefing, re.MULTILINE)

    counts: dict[str, int] = {}
    for line in sources:
        for token in re.split(r"[\s,;]+", line.strip()):
            word = token.strip("\"'`.:()[]{}").strip()
            if len(word) < 2 or word.lower() in _STOPWORDS or word.startswith("#"):
                continue
            counts[word] = counts.get(word, 0) + 3  # ground truth outranks hints

    # Tokens the guidance or the metric feedback names explicitly.
    for quoted in re.findall(
        r"(?:marker token[s]?|required token[s]?|keyword[s]?)[^\n]*?((?:'[^']+'|\"[^\"]+\")(?:[^\n]*?(?:'[^']+'|\"[^\"]+\"))*)",
        briefing,
        re.IGNORECASE,
    ):
        for word in re.findall(r"['\"]([^'\"]+)['\"]", quoted):
            word = word.strip()
            if len(word) < 2 or word.lower() in _STOPWORDS:
                continue
            counts[word] = counts.get(word, 0) + 2
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _current_prompt(briefing: str) -> str:
    """Return the prompt the briefing is asking us to improve."""
    for header in ("### System prompt", "## Draft system prompt", "## Draft"):
        idx = briefing.find(header)
        if idx == -1:
            continue
        block = re.search(r"```(?:\w+)?\n(.*?)```", briefing[idx:], re.DOTALL)
        if block:
            return block.group(1).strip()
    return ""


def _pass_kind(briefing: str) -> str:
    """Return DRAFT / CRITIQUE / REVISE for a multi-pass rewrite briefing."""
    match = re.search(r"##\s*Task\s*\(pass\s*\d+/\d+:\s*(\w+)\)", briefing, re.IGNORECASE)
    return match.group(1).upper() if match else "DRAFT"


def _rewrite_prompt(user_prompt: str) -> str:
    """Play the prompt-rewriter across the fitter's DRAFT/CRITIQUE/REVISE passes.

    The fitter hands over a briefing containing the current prompt, the
    failing I/O, and the expected answers. The move a real rewrite model makes
    is to fold the missing expected tokens into the prompt while keeping the
    role intact — so that is what happens here, deterministically.
    """
    kind = _pass_kind(user_prompt)
    keywords = _expected_keywords(user_prompt)
    base = _current_prompt(user_prompt) or "You are a helpful assistant."
    # Keep the role sentence, drop any requirement clause a previous pass added.
    role = base.split("Always include")[0].strip().rstrip(".")

    if kind == "CRITIQUE":
        gaps = [
            f"- The prompt never mentions the required token '{k}', which every "
            f"expected answer contains."
            for k in keywords[:3]
        ]
        gaps.append("- The output format is not stated explicitly.")
        return "\n".join(gaps)

    if not keywords:
        return f"---\n{role}."
    required = ", ".join(f"'{k}'" for k in keywords[:4])
    improved = (
        f"{role}. Always include {required} in your answer, exactly as written, for every query."
    )
    return f"---\n{improved}"


def _requested_dimensions(prompt: str) -> list[str]:
    """Return the dimension names a judge prompt asks to be scored."""
    block = re.search(r'"dimensions"\s*:\s*\{(.*?)\}', prompt, re.DOTALL)
    if not block:
        return []
    return re.findall(r'"(\w+)"\s*:', block.group(1))


def _judge_payload(prompt: str) -> str:
    """Grade a candidate answer in the exact schema the judge asked for.

    Structure and attribution are what the metric rewards, so the judge
    rewards them too — an LLM-as-judge that disagreed with the metric would
    give the optimizer a contradictory signal.
    """
    low = prompt.lower()
    score = 0.2
    for marker, weight in (("source", 0.25), ("confidence", 0.2), ("{", 0.25), ("reasoning", 0.1)):
        if marker in low:
            score += weight
    score = min(1.0, round(score, 3))
    payload: dict[str, Any] = {
        "overall_score": score,
        "score": score,
        "feedback": "Graded on structure, attribution and stated confidence.",
        "motivation": (
            "The response was checked for a structured body, an explicit source "
            "and a stated confidence; each present raises the score."
        ),
        "what_worked": ["structured output"] if "{" in low else [],
        "what_failed": [] if score > 0.6 else ["missing attribution or structure"],
        "improvement_hints": (
            [] if score > 0.6 else ["Require JSON output and an explicit source in the prompt."]
        ),
    }
    dims = _requested_dimensions(prompt)
    if dims:
        payload["dimensions"] = {d: score for d in dims}
    return json.dumps(payload)


def _literal_echo(prompt: str) -> str | None:
    """Honour 'reply with exactly X' instructions, as an instruct model would."""
    match = re.search(
        r"repl(?:y|ies)\s+with\s+exactly\s+(?:the\s+word\s+)?[\"'`]?([\w .-]{1,40}?)[\"'`]?"
        r"(?:\s+and\s+nothing\s+else)?[.!]?\s*$",
        prompt.strip(),
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _echo_requested_json(prompt: str) -> str | None:
    """Return the JSON object a prompt literally asks to be returned."""
    match = re.search(r"[Rr]eturn\s+JSON\s*(\{.*?\})", prompt, re.DOTALL)
    if not match:
        return None
    try:
        return json.dumps(json.loads(match.group(1)))
    except Exception:  # noqa: BLE001 - not a literal object, fall through
        return None


def _is_rewrite_request(prompt: str) -> bool:
    """Return whether the caller is asking for an improved prompt."""
    low = prompt.lower()
    return any(
        w in low
        for w in (
            "rewrite",
            "improved prompt",
            "improved system prompt",
            "new system prompt",
            "better prompt",
            "propose a prompt",
            "draft system prompt",
            "optimization briefing",
            "mutated system prompt",
            "prompt mutation",
        )
    )


def _is_judge_request(prompt: str) -> bool:
    """Return whether the caller is asking for a graded evaluation.

    Keyed on ``overall_score``, which only the judge schema asks for. Looser
    words do not work: a judge prompt talks about "the prompt rewriter", so
    matching on "rewrite" sends grading requests down the rewrite path.
    """
    low = prompt.lower()
    return "overall_score" in low or "evaluation judge" in low or "dimensions to score" in low


def _answer(system_prompt: str, question: str) -> str:
    """Answer *question* honouring whatever the system prompt asked for.

    This is the behaviour that makes the server a real optimization target:
    every directive the prompt carries adds a property the metric rewards, so
    a better prompt produces a strictly better answer.
    """
    directives = _active_directives(system_prompt)
    fact = _lookup(question)
    core = fact if fact else "I don't have a definitive answer for that."

    if "json" in directives:
        payload: dict[str, Any] = {"answer": core}
        if "cite" in directives:
            payload["source"] = "internal-knowledge-base"
        if "confidence" in directives:
            payload["confidence"] = 0.92 if fact else 0.20
        if "steps" in directives:
            payload["reasoning"] = "Matched the question against known facts."
        return json.dumps(payload)

    parts = [core]
    if "steps" in directives:
        parts.append("Reasoning: matched the question against known facts.")
    if "cite" in directives:
        parts.append("Source: internal-knowledge-base.")
    if "confidence" in directives:
        parts.append(f"Confidence: {0.92 if fact else 0.20}.")
    text = " ".join(parts)
    if "concise" in directives:
        text = parts[0] if len(parts) == 1 else " ".join(parts[:2])
    return text


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """OpenAI-compatible chat completion."""
    body = await request.json()
    messages = body.get("messages") or []
    system_prompt = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    )
    user_prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")

    literal = _literal_echo(user_prompt)
    echoed = _echo_requested_json(user_prompt)
    if literal is not None:
        content = literal
    elif echoed is not None:
        content = echoed
    elif _is_judge_request(user_prompt):
        content = _judge_payload(user_prompt)
    elif _is_rewrite_request(user_prompt) or _is_rewrite_request(system_prompt):
        content = _rewrite_prompt(user_prompt)
    else:
        content = _answer(system_prompt, user_prompt)

    wants_json = (body.get("response_format") or {}).get("type") == "json_object"
    if wants_json and not content.lstrip().startswith("{"):
        content = json.dumps({"answer": content})

    prompt_tokens = max(1, len((system_prompt + user_prompt).split()))
    completion_tokens = max(1, len(content.split()))
    return JSONResponse(
        {
            "id": f"chatcmpl-{_seed(system_prompt + user_prompt):08x}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


@app.post("/v1/completions")
async def completions(request: Request) -> JSONResponse:
    """Legacy completion endpoint, for callers that still use it."""
    body = await request.json()
    prompt = str(body.get("prompt", ""))
    literal = _literal_echo(prompt)
    if literal is not None:
        content = literal
    elif _is_rewrite_request(prompt):
        content = _rewrite_prompt(prompt)
    else:
        content = _answer("", prompt)
    return JSONResponse(
        {
            "id": f"cmpl-{_seed(prompt):08x}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.get("model") or MODEL_ID,
            "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def main() -> None:
    """Serve the stand-in model on the requested port."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
