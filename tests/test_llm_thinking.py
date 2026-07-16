"""Tests for thinking / reasoning normalization in LLM providers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from agentomatic.providers import (
    StructuredOutputFallbackWrapper,
    attach_thinking_metadata,
    invoke_with_retry,
    llm_result_metadata,
    message_text,
    message_thinking,
    split_llm_message,
    split_thinking_text,
    strip_thinking_for_json,
)


class _Out(BaseModel):
    project_name: str = ""
    ok: bool = False


def test_split_think_tags() -> None:
    raw = '<think>internal plan</think>\n{"project_name": "Portail", "ok": true}'
    split = split_thinking_text(raw)
    assert "internal plan" in split.thinking
    assert "Portail" in split.answer
    assert "<think>" not in split.answer


def test_message_text_excludes_thinking_by_default() -> None:
    msg = SimpleNamespace(
        content="<think>secret</think>Bonjour le monde",
        additional_kwargs={},
        response_metadata={},
    )
    assert message_text(msg) == "Bonjour le monde"
    assert "secret" in message_thinking(msg)
    assert "secret" in message_text(msg, include_thinking=True)


def test_reasoning_content_attribute() -> None:
    msg = SimpleNamespace(
        content='{"ok": true}',
        reasoning_content="step by step",
        additional_kwargs={},
        response_metadata={},
    )
    split = split_llm_message(msg)
    assert split.answer == '{"ok": true}'
    assert split.thinking == "step by step"
    meta = llm_result_metadata(msg)
    assert meta["has_thinking"] is True
    assert meta["thinking"] == "step by step"


def test_attach_thinking_metadata_strips_content() -> None:
    msg = SimpleNamespace(
        content="<think>plan</think>FINAL",
        additional_kwargs={},
        response_metadata={},
    )
    out = attach_thinking_metadata(msg)
    assert out.content == "FINAL"
    assert out.additional_kwargs.get("thinking") == "plan"


def test_strip_thinking_for_json() -> None:
    raw = 'Thinking Process:\n1. foo\n\n```json\n{"ok": true}\n```'
    cleaned = strip_thinking_for_json(raw)
    assert cleaned.strip().startswith("{") or "ok" in cleaned


@pytest.mark.asyncio
async def test_invoke_with_retry_strips_thinking() -> None:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content="<think>x</think>answer-only",
            additional_kwargs={},
            response_metadata={},
        )
    )
    result = await invoke_with_retry(llm, [{"role": "user", "content": "hi"}], max_retries=0)
    assert result.content == "answer-only"
    assert result.additional_kwargs.get("thinking") == "x"


@pytest.mark.asyncio
async def test_structured_fallback_parses_after_thinking() -> None:
    class _LLM:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(
                content='<think>draft</think>\n{"project_name": "Portail Sinistres Digital", "ok": true}',
                additional_kwargs={},
            )

    wrapped = StructuredOutputFallbackWrapper(_LLM(), _Out)
    out = await wrapped.ainvoke([{"role": "user", "content": "x"}])
    assert isinstance(out, _Out)
    assert out.project_name == "Portail Sinistres Digital"
    assert out.ok is True
