"""Dedicated, structured audit-log sink.

Every agent/op run emits a structured audit event (agent, op, request id,
model, tokens, latency, outcome) correlated by request id. Events go to a
dedicated loguru sink (a JSONL file) that is separate from application logs and
never contains raw PII payloads — only hashes/metadata.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loguru import logger

_AUDIT_MARK = "scooper_audit"
_configured = False


def configure_audit_logging(path: str | Path = ".local/audit/audit.jsonl") -> None:
    """Attach a dedicated JSONL audit sink (idempotent).

    Args:
        path: Destination file for audit records.
    """
    global _configured
    if _configured:
        return
    candidates = [Path(path), Path("/tmp/scooper-audit/audit.jsonl")]
    last_error: Exception | None = None
    for dest in candidates:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Prove the path is writable before handing it to loguru.
            with dest.open("a", encoding="utf-8"):
                pass
            logger.add(
                dest,
                level="INFO",
                serialize=True,
                rotation="20 MB",
                retention="90 days",
                enqueue=True,
                filter=lambda record: record["extra"].get(_AUDIT_MARK) is True,
            )
            if dest != Path(path):
                logger.warning(
                    "Audit log path {} not writable; using fallback {}",
                    path,
                    dest,
                )
            _configured = True
            return
        except OSError as exc:  # pragma: no cover - volume / permission edge
            last_error = exc
            continue
    logger.warning(
        "Audit file sink disabled (could not write {}): {}",
        path,
        last_error,
    )
    _configured = True


def emit_audit_event(
    *,
    agent: str,
    op: str,
    request_id: str,
    outcome: str,
    model: str | None = None,
    tokens: int | None = None,
    latency_ms: float | None = None,
    language: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit one structured audit event and return the record dict.

    Args:
        agent: Agent or component name.
        op: Operation name.
        request_id: Correlation id (``X-Request-Id``).
        outcome: ``success`` | ``fallback`` | ``error`` | ``cancelled``.
        model: Model identifier, if any.
        tokens: Token count, if known.
        latency_ms: Wall-clock latency in milliseconds.
        language: Resolved output language code.
        extra: Additional non-PII metadata.

    Returns:
        The audit record that was logged.
    """
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": agent,
        "op": op,
        "request_id": request_id,
        "outcome": outcome,
        "model": model,
        "tokens": tokens,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "language": language,
        **(extra or {}),
    }
    logger.bind(**{_AUDIT_MARK: True}).info("audit {}", record)
    return record


class audit_timer:  # noqa: N801 - context-manager helper reads better lowercase
    """Context manager that times a block and emits an audit event on exit."""

    def __init__(self, *, agent: str, op: str, request_id: str, **kwargs: Any) -> None:
        self.agent = agent
        self.op = op
        self.request_id = request_id
        self.kwargs = kwargs
        self.outcome = "success"
        self._start = 0.0
        self.record: dict[str, Any] = {}

    def __enter__(self) -> audit_timer:
        self._start = time.perf_counter()
        return self

    def fail(self, outcome: str = "error") -> None:
        """Mark the audited block's outcome (e.g. ``fallback``/``error``)."""
        self.outcome = outcome

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is not None and self.outcome == "success":
            self.outcome = "error"
        self.record = emit_audit_event(
            agent=self.agent,
            op=self.op,
            request_id=self.request_id,
            outcome=self.outcome,
            latency_ms=latency_ms,
            **self.kwargs,
        )
