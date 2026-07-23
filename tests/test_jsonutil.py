"""Tests for public JSON extract/repair helpers."""

from __future__ import annotations

from agentomatic.providers.jsonutil import (
    extract_json,
    extract_json_array,
    extract_json_object,
    loads_repaired,
    repair_json,
)


def test_extract_fenced_json() -> None:
    """Markdown-fenced JSON is extracted."""
    text = 'Here you go:\n```json\n{"a": 1}\n```\n'
    assert extract_json_object(text) == {"a": 1}


def test_repair_trailing_comma() -> None:
    """Trailing commas are repaired."""
    text = '{"a": 1, "b": 2,}'
    assert extract_json_object(text) == {"a": 1, "b": 2}
    assert repair_json(text) == '{"a": 1, "b": 2}'


def test_thinking_preamble_stripped() -> None:
    """Thinking Process preamble is stripped before parse."""
    text = 'Thinking Process: I should answer.\n{"ok": true}'
    assert extract_json_object(text) == {"ok": True}


def test_truncated_returns_none() -> None:
    """Truncated JSON returns None rather than raising."""
    assert extract_json('{"a": 1') is None
    assert loads_repaired("not json") is None


def test_extract_array() -> None:
    """Array extraction prefers arrays when expect=array."""
    text = "Results: [1, 2, 3]"
    assert extract_json_array(text) == [1, 2, 3]
