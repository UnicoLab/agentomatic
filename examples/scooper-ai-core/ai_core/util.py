"""Small generic helpers shared by agents and plugins.

No frontend-specific contracts live here — callers pass schemas and shape
responses themselves.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

_VALID_STATUSES: frozenset[str] = frozenset(
    {"informed", "missing", "to_confirm", "user_completed", "user_modified"}
)


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def positive_int_days(value: Any, fallback: int = 1) -> int:
    """Coerce *value* to a strictly-positive integer number of days."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return max(1, fallback)
    if not math.isfinite(num) or num <= 0:
        return max(1, fallback)
    return max(1, int(round(num)))


def clamp_unit(value: Any, fallback: float = 0.5) -> float:
    """Clamp a value into ``[0, 1]``."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(num):
        return fallback
    return max(0.0, min(1.0, num))


def coerce_status(
    value: Any,
    *,
    has_value: bool | None = None,
    default: str = "to_confirm",
) -> str:
    """Coerce an extraction status to a known value.

    Args:
        value: The candidate status.
        has_value: When provided, used to infer a status if *value* is invalid
            (``True`` -> ``"informed"``, ``False`` -> ``"missing"``).
        default: Fallback status when nothing else applies.

    Returns:
        A normalised status string.
    """
    if isinstance(value, str) and value in _VALID_STATUSES:
        return value
    if has_value is True:
        return "informed"
    if has_value is False:
        return "missing"
    return default


def message_text(result: Any) -> str:
    """Extract answer text from an LLM result (thinking stripped).

    Delegates to :func:`agentomatic.providers.message_text` so agents share
    the framework's modern-LLM thinking / reasoning handling.
    """
    try:
        from agentomatic.providers import message_text as _am_message_text

        return _am_message_text(result)
    except Exception:  # noqa: BLE001 - keep a tiny local fallback for unit tests
        content = getattr(result, "content", None)
        if content is None:
            return str(result)
        if isinstance(content, list):
            parts = [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ]
            return "".join(parts)
        return str(content)
