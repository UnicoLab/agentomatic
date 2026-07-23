"""Shared language-control layer.

Policy: agents *reason* in English (all prompts are English) but render
user-facing output in the caller's language. The caller may pass an explicit
``language`` parameter; otherwise it is detected from the input text. A short
directive is appended to prompts so the model emits user-facing strings in the
requested language.
"""

from __future__ import annotations

import re

# Common language names by ISO code, for prompt directives.
_LANGUAGE_NAMES: dict[str, str] = {
    "fr": "French",
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
}

# Lightweight stopword signals for heuristic detection (no heavy deps).
_FR_MARKERS = {
    "le",
    "la",
    "les",
    "des",
    "une",
    "un",
    "et",
    "est",
    "vous",
    "pour",
    "avec",
    "dans",
    "projet",
    "besoin",
    "qui",
    "que",
    "quel",
    "quelle",
    "sont",
    "être",
}
_EN_MARKERS = {
    "the",
    "and",
    "is",
    "are",
    "you",
    "for",
    "with",
    "this",
    "that",
    "project",
    "need",
    "which",
    "what",
    "should",
    "would",
    "will",
}

_TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûüç]+", re.IGNORECASE)


def detect_language(text: str, default: str = "fr") -> str:
    """Detect ``fr`` or ``en`` from *text* using a stopword heuristic.

    Args:
        text: Input text to inspect.
        default: Language returned when detection is inconclusive.

    Returns:
        A two-letter ISO code (``fr`` or ``en``), or *default*.
    """
    if not text:
        return default
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return default
    fr_hits = sum(1 for t in tokens if t in _FR_MARKERS)
    en_hits = sum(1 for t in tokens if t in _EN_MARKERS)
    if fr_hits == en_hits:
        return default
    return "fr" if fr_hits > en_hits else "en"


def resolve_language(explicit: str | None, *texts: str, default: str = "fr") -> str:
    """Resolve the output language from an explicit param or input text.

    Args:
        explicit: An explicit ``language`` value from the request (or ``None``).
        *texts: Candidate input texts for detection.
        default: Fallback language.

    Returns:
        A normalised two-letter language code.
    """
    if explicit:
        code = explicit.strip().lower()[:2]
        if code:
            return code
    joined = " ".join(t for t in texts if t)
    return detect_language(joined, default=default)


def language_name(code: str) -> str:
    """Return the English name for a language code (falls back to the code)."""
    return _LANGUAGE_NAMES.get(code.lower()[:2], code)


def output_directive(code: str) -> str:
    """Return a prompt directive instructing the model on the output language.

    Avoid phrasing that nudges Qwen-style models into a visible
    ``Thinking Process:`` preamble (that burns ``max_tokens`` and yields
    empty JSON after thinking-strip).
    """
    return (
        f"Write ALL user-facing text fields in {language_name(code)}. "
        f"Keep JSON keys and enum values exactly as specified "
        f"(do not translate keys or enum values). "
        f"Do not emit chain-of-thought, Thinking Process, or prose outside JSON."
    )
