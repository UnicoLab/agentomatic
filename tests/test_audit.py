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
    timer = audit_timer(agent="a", op="op", request_id="r")
    with pytest.raises(RuntimeError):
        with timer:
            raise RuntimeError("boom")
    assert timer.record["outcome"] == "error"


def test_emit_audit_event_resolves_contextvar_usage_and_isolates_steps() -> None:
    """emit_audit_event falls back to active runtime usage and respects agent scoping."""
    from agentomatic.logs.runtime import (
        ensure_llm_usage,
        record_llm_usage,
        reset_llm_usage,
    )

    _, tok = ensure_llm_usage(force_new=True)
    try:
        record_llm_usage("gpt-5", 10, 5, 15, agent_name="step_one")
        ev_step1 = emit_audit_event(
            agent="step_one",
            op="agent:pipeline_step",
            request_id="req-step1",
            outcome="success",
        )
        ev_step2 = emit_audit_event(
            agent="step_two_fastpath",
            op="agent:pipeline_step",
            request_id="req-step2",
            outcome="success",
        )
        ev_endpoint = emit_audit_event(
            agent="det_endpoint",
            op="endpoint:pipeline_step",
            request_id="req-ep",
            outcome="success",
        )
        ev_pipe = emit_audit_event(
            agent="two_stage",
            op="pipeline:run",
            request_id="req-pipe",
            outcome="success",
        )
        assert ev_step1["model"] == "gpt-5"
        assert ev_step1["tokens"] == 15
        assert ev_step2["model"] is None
        assert ev_step2["tokens"] is None
        assert ev_endpoint["model"] is None
        assert ev_endpoint["tokens"] is None
        assert ev_pipe["model"] == "gpt-5"
        assert ev_pipe["tokens"] == 15
    finally:
        reset_llm_usage(tok)
