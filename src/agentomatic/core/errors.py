"""Client-safe error reporting.

An exception's ``str()`` routinely carries operational detail that should not
leave the process: database drivers put full DSNs (credentials included) in
their messages, HTTP clients echo request URLs with tokens in the query string,
and auth libraries name internal hosts. Interpolating ``{exc}`` straight into
an HTTP response therefore leaks secrets to anyone who can trigger a failure.

:func:`client_safe_detail` logs the full exception server-side and returns a
sanitised payload carrying a short *error id* that correlates the response to
that log line — so operators keep full diagnostics without publishing them.

Set ``AGENTOMATIC_DEBUG_ERRORS=1`` to include raw exception text in responses
while developing locally.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from loguru import logger

#: Opt-in flag that puts raw exception text back into HTTP responses.
DEBUG_ERRORS_ENV = "AGENTOMATIC_DEBUG_ERRORS"


def debug_errors_enabled() -> bool:
    """Whether raw exception text may be returned to clients."""
    return os.getenv(DEBUG_ERRORS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def client_safe_detail(
    exc: BaseException,
    *,
    context: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log *exc* in full and return a payload that is safe to send to a client.

    Args:
        exc: The exception being handled.
        context: Short description of the failing operation, e.g.
            ``"Agent invocation failed"``. This is caller-authored text, so it
            is always safe to return.
        extra: Additional non-sensitive fields to merge into the payload.

    Returns:
        ``{"error": <context>, "error_id": <id>, "error_type": <ExcClass>}``,
        plus ``"detail"`` with the raw message when debug errors are enabled.
    """
    error_id = uuid.uuid4().hex[:12]
    # Full detail (including traceback) goes to the server log only.
    logger.opt(exception=exc).error(f"{context} [error_id={error_id}]")

    payload: dict[str, Any] = {
        "error": context,
        "error_id": error_id,
        "error_type": type(exc).__name__,
    }
    if extra:
        payload.update(extra)
    if debug_errors_enabled():
        payload["detail"] = str(exc)
    return payload
