"""Normalize modern LLM responses that include thinking / reasoning.

Models such as Qwen3.x, DeepSeek-R1, and some Gemma/oMLX builds may return:

* ``<think>...</think>`` (or ``<thinking>``) tags around chain-of-thought
* ``reasoning_content`` / ``reasoning`` fields on the message or in
  ``additional_kwargs`` / ``response_metadata``
* OpenAI-style content blocks ``{"type": "reasoning", ...}``
* A prose ``Thinking Process:`` preamble before the final answer

Agents should use :func:`message_text` for **user-facing / JSON** content
(answer only) and :func:`message_thinking` / :func:`llm_result_metadata`
when they need the reasoning trail for debugging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_THINK_TAG_RE = re.compile(
    r"(?is)<\s*(?:think|thinking|reason|reasoning)\s*>(.*?)<\s*/\s*"
    r"(?:think|thinking|reason|reasoning)\s*>"
)
# Public alias for streaming helpers / tests.
THINK_TAG_RE = _THINK_TAG_RE
_THINK_PREAMBLE_RE = re.compile(
    r"(?is)^(?:Thinking Process:|Reasoning:|Chain of Thought:)\s*.*?(?=\n\s*[{\[`\"A-ZÀ-Ö]|\\Z)"
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SplitMessage:
    """Answer text separated from optional thinking / reasoning."""

    answer: str
    thinking: str = ""
    source_fields: tuple[str, ...] = field(default_factory=tuple)


def _as_text(value: Any) -> str:
    """Coerce heterogeneous content parts to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = str(block.get("type") or "").lower()
                if btype in {"thinking", "reasoning", "reason"}:
                    continue  # handled separately
                if "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif "content" in block:
                    parts.append(str(block.get("content") or ""))
                else:
                    parts.append(str(block))
            else:
                text = getattr(block, "text", None)
                parts.append(str(text) if text is not None else str(block))
        return "".join(parts)
    return str(value)


def _collect_structured_thinking(result: Any) -> tuple[str, tuple[str, ...]]:
    """Pull thinking from message attributes / metadata / content blocks."""
    chunks: list[str] = []
    sources: list[str] = []

    def _take(label: str, value: Any) -> None:
        text = _as_text(value).strip()
        if text:
            chunks.append(text)
            sources.append(label)

    for attr in ("reasoning_content", "reasoning", "thinking"):
        if hasattr(result, attr):
            _take(attr, getattr(result, attr))

    additional = getattr(result, "additional_kwargs", None) or {}
    if isinstance(additional, dict):
        for key in (
            "reasoning_content",
            "reasoning",
            "thinking",
            "reasoning_text",
            "chain_of_thought",
        ):
            if key in additional:
                _take(f"additional_kwargs.{key}", additional.get(key))

    meta = getattr(result, "response_metadata", None) or {}
    if isinstance(meta, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            if key in meta:
                _take(f"response_metadata.{key}", meta.get(key))
        model_extra = meta.get("model_extra") or meta.get("extra") or {}
        if isinstance(model_extra, dict):
            for key in ("reasoning_content", "reasoning", "thinking"):
                if key in model_extra:
                    _take(f"response_metadata.extra.{key}", model_extra.get(key))

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "").lower()
            if btype in {"thinking", "reasoning", "reason"}:
                _take(
                    f"content_block.{btype}", block.get("thinking") or block.get("text") or block
                )

    return "\n\n".join(chunks).strip(), tuple(sources)


def split_thinking_text(text: str) -> SplitMessage:
    """Split a raw string into ``(thinking, answer)``.

    Prefer tagged ``<think>...</think>`` regions; otherwise strip a leading
    ``Thinking Process:`` preamble when the remainder looks like the answer.
    """
    if not text:
        return SplitMessage(answer="", thinking="")

    tagged = list(_THINK_TAG_RE.finditer(text))
    if tagged:
        thinking = "\n\n".join(m.group(1).strip() for m in tagged if m.group(1).strip())
        answer = _THINK_TAG_RE.sub("", text).strip()
        return SplitMessage(answer=answer, thinking=thinking, source_fields=("think_tag",))

    # Prose preamble used by some Qwen builds when tags are disabled.
    preamble = _THINK_PREAMBLE_RE.match(text)
    if preamble and preamble.end() < len(text):
        thinking = preamble.group(0).strip()
        answer = text[preamble.end() :].strip()
        if answer:
            return SplitMessage(
                answer=answer,
                thinking=thinking,
                source_fields=("thinking_preamble",),
            )

    return SplitMessage(answer=text.strip(), thinking="")


def split_llm_message(result: Any) -> SplitMessage:
    """Split an LLM result (AIMessage / str / dict) into answer + thinking."""
    if result is None:
        return SplitMessage(answer="", thinking="")

    structured_thinking, sources = _collect_structured_thinking(result)

    if isinstance(result, dict):
        raw = (
            result.get("content")
            or result.get("message")
            or result.get("text")
            or result.get("output")
            or result
        )
        if isinstance(raw, dict) and "content" in raw:
            raw = raw["content"]
        text = _as_text(raw)
    else:
        content = getattr(result, "content", None)
        text = _as_text(content) if content is not None else str(result)

    split = split_thinking_text(text)
    thinking_parts = [p for p in (structured_thinking, split.thinking) if p]
    thinking = "\n\n".join(thinking_parts).strip()
    all_sources = tuple(dict.fromkeys((*sources, *split.source_fields)))
    return SplitMessage(answer=split.answer, thinking=thinking, source_fields=all_sources)


def message_text(result: Any, *, include_thinking: bool = False) -> str:
    """Return the final answer text from an LLM result.

    By default thinking / reasoning is **excluded** so agents can safely feed
    the string into JSON parsers or French user-facing fields.
    """
    split = split_llm_message(result)
    if include_thinking and split.thinking:
        return f"{split.thinking}\n\n{split.answer}".strip()
    return split.answer


def message_thinking(result: Any) -> str:
    """Return only the thinking / reasoning portion (may be empty)."""
    return split_llm_message(result).thinking


def llm_result_metadata(result: Any) -> dict[str, Any]:
    """Build a small metadata dict with optional thinking for debug logs."""
    split = split_llm_message(result)
    meta: dict[str, Any] = {
        "has_thinking": bool(split.thinking),
        "thinking_sources": list(split.source_fields),
        "answer_chars": len(split.answer),
        "thinking_chars": len(split.thinking),
    }
    if split.thinking:
        meta["thinking"] = split.thinking
    return meta


def strip_thinking_for_json(text: str) -> str:
    """Return text safe for JSON extraction (answer only, fences preferred)."""
    split = split_thinking_text(text or "")
    body = split.answer.strip()
    fence = _FENCE_RE.search(body)
    if fence:
        return fence.group(1).strip()
    return body


def attach_thinking_metadata(result: Any, *, strip_content: bool = True) -> Any:
    """Return a result whose ``.content`` is the answer and metadata has thinking.

    Prefer LangChain ``AIMessage.model_copy`` when available; otherwise return a
    lightweight namespace that still exposes ``.content``.
    """
    split = split_llm_message(result)
    if not split.thinking and not strip_content:
        return result

    additional = dict(getattr(result, "additional_kwargs", None) or {})
    if split.thinking:
        additional["thinking"] = split.thinking
        additional["reasoning_content"] = split.thinking

    response_metadata = dict(getattr(result, "response_metadata", None) or {})
    response_metadata["agentomatic_thinking"] = {
        "has_thinking": bool(split.thinking),
        "sources": list(split.source_fields),
    }

    new_content = split.answer if strip_content else _as_text(getattr(result, "content", result))

    if hasattr(result, "model_copy"):
        try:
            return result.model_copy(
                update={
                    "content": new_content,
                    "additional_kwargs": additional,
                    "response_metadata": response_metadata,
                }
            )
        except Exception:  # noqa: BLE001 - fall through to namespace
            pass

    from types import SimpleNamespace

    return SimpleNamespace(
        content=new_content,
        additional_kwargs=additional,
        response_metadata=response_metadata,
        raw=result,
    )
