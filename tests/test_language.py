"""Tests for language detection helpers."""

from __future__ import annotations

from agentomatic.agents.language import (
    detect_language,
    language_name,
    output_directive,
    resolve_language,
)


def test_detect_french() -> None:
    """French stopwords tip detection toward fr."""
    assert detect_language("Le projet est dans les besoins pour vous") == "fr"


def test_detect_english() -> None:
    """English stopwords tip detection toward en."""
    assert detect_language("The project is for you with this need") == "en"


def test_resolve_explicit() -> None:
    """Explicit language wins over detection."""
    assert resolve_language("de", "The project is for you") == "de"


def test_output_directive() -> None:
    """Directive names the language and forbids CoT."""
    text = output_directive("fr")
    assert "French" in text
    assert language_name("fr") == "French"
    assert "chain-of-thought" in text.lower() or "Thinking Process" in text
