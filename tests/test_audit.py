"""Tests for the op-audit JSONL sink."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentomatic.observability.audit import (
    audit_timer,
    configure_audit_logging,
    emit_audit_event,
    reset_audit_logging,
)


def test_emit_audit_event_shape() -> None:
    """emit_audit_event returns a structured record."""
    reset_audit_logging()
    record = emit_audit_event(
        agent="demo",
        op="invoke",
        request_id="req-1",
        outcome="success",
        model="mistral",
        tokens=12,
        latency_ms=1.234,
        language="en",
    )
    assert record["agent"] == "demo"
    assert record["request_id"] == "req-1"
    assert record["latency_ms"] == 1.23


def test_configure_audit_logging(tmp_path: Path) -> None:
    """configure_audit_logging attaches a writable sink."""
    reset_audit_logging()
    path = tmp_path / "audit.jsonl"
    assert configure_audit_logging(path) is True
    emit_audit_event(
        agent="demo",
        op="chat",
        request_id="req-2",
        outcome="success",
    )
    assert path.parent.exists()


def test_audit_timer_success() -> None:
    """audit_timer emits success with latency."""
    with audit_timer(agent="a", op="op", request_id="r") as timer:
        pass
    assert timer.record["outcome"] == "success"
    assert timer.record["latency_ms"] is not None


def test_audit_timer_error() -> None:
    """audit_timer records error outcome on exception."""
    with pytest.raises(RuntimeError):
        with audit_timer(agent="a", op="op", request_id="r") as timer:
            raise RuntimeError("boom")
    assert timer.record["outcome"] == "error"
