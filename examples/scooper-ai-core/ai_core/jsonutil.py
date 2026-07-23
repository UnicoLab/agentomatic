"""Strict-JSON extraction and repair helpers for small-LLM outputs.

The local MLX model is small and frequently wraps JSON in prose or code
fences, emits trailing commas, or truncates. These helpers extract the first
JSON value and apply conservative repairs before falling back to ``None`` so
callers can trigger a rule-based fallback path.

Thinking / reasoning preambles are stripped via agentomatic when available
(:func:`agentomatic.providers.strip_thinking_for_json`).
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(
    r"(?is)(?:<\s*(?:think|thinking)\s*>.*?</\s*(?:think|thinking)\s*>|"
    r"Thinking Process:.*?(?=\{|\[|$)|"
    r"Reasoning:.*?(?=\{|\[|$))"
)


def _strip_fences(text: str) -> str:
    """Remove Markdown code fences, keeping the fenced body when present."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _strip_thinking(text: str) -> str:
    """Drop chain-of-thought prefixes some instruct models emit before JSON."""
    try:
        from agentomatic.providers import strip_thinking_for_json

        cleaned = strip_thinking_for_json(text)
        if cleaned:
            return cleaned
    except Exception:  # noqa: BLE001 - local fallback for isolated unit tests
        pass
    cleaned = _THINK_RE.sub("", text)
    return cleaned.strip()


def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced ``open_ch``..``close_ch`` span in *text*."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair(candidate: str) -> str:
    """Apply conservative textual repairs to a near-JSON string."""
    repaired = candidate
    # Remove trailing commas before } or ].
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Normalise smart quotes.
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    return repaired


def extract_json(text: str, *, expect: str = "object") -> Any | None:
    """Extract and parse the first JSON value from *text*.

    Args:
        text: Raw LLM output (may contain prose / code fences).
        expect: ``"object"`` to look for ``{...}`` first, ``"array"`` for
            ``[...]``. The other bracket type is tried as a fallback.

    Returns:
        The parsed Python value, or ``None`` when no valid JSON is found.
    """
    if not text or not text.strip():
        return None
    body = _strip_thinking(_strip_fences(text))

    # Fast path: the whole body is JSON.
    for attempt in (body, _repair(body)):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            pass

    # Prefer the first balanced span after thinking strip (nested objects stay
    # intact). Later spans are tried only if earlier ones fail to parse.
    pairs = [("{", "}"), ("[", "]")]
    if expect == "array":
        pairs.reverse()
    for open_ch, close_ch in pairs:
        spans: list[str] = []
        search_from = 0
        while True:
            start = body.find(open_ch, search_from)
            if start == -1:
                break
            span = _find_balanced(body[start:], open_ch, close_ch)
            if span:
                spans.append(span)
                search_from = start + max(len(span), 1)
            else:
                break
        for span in spans:
            for attempt in (span, _repair(span)):
                try:
                    return json.loads(attempt)
                except (json.JSONDecodeError, ValueError):
                    continue
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object, returning ``None`` if the value is not a dict."""
    value = extract_json(text, expect="object")
    return value if isinstance(value, dict) else None


def extract_json_array(text: str) -> list[Any] | None:
    """Extract a JSON array, returning ``None`` if the value is not a list."""
    value = extract_json(text, expect="array")
    return value if isinstance(value, list) else None
