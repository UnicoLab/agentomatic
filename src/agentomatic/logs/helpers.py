"""Shared helpers for recording invocations from routers and pipeline steps."""

from __future__ import annotations

from typing import Any

from agentomatic.logs.recorder import InvocationLogRecorder
from agentomatic.logs.runtime import get_invocation_log_recorder


async def record_invocation(
    *,
    resource_type: str,
    resource_name: str,
    endpoint: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
    duration_ms: float | None = None,
    status: str = "ok",
    thread_id: str | None = None,
    recorder: InvocationLogRecorder | None = None,
) -> dict[str, Any] | None:
    """Record one invocation via an explicit or platform-bound recorder.

    Args:
        resource_type: Resource kind (agent|plugin|…).
        resource_name: Resource identifier.
        endpoint: Operation label.
        input_data: Request payload.
        output_data: Response payload.
        metadata: Extra metadata.
        error: Error message when status is not ok.
        duration_ms: Duration in milliseconds.
        status: ``ok`` / ``error`` / ``suspended``.
        thread_id: Optional thread id.
        recorder: Optional explicit recorder; falls back to platform binding.

    Returns:
        Stored log dict, or ``None`` when logging is disabled/unavailable.
    """
    active = recorder if recorder is not None else get_invocation_log_recorder()
    if active is None or not active.enabled:
        return None
    return await active.record(
        resource_type=resource_type,
        resource_name=resource_name,
        endpoint=endpoint,
        input_data=input_data,
        output_data=output_data,
        metadata=metadata,
        error=error,
        duration_ms=duration_ms,
        status=status,
        thread_id=thread_id,
    )
