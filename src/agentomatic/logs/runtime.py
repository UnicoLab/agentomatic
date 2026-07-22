"""Process-wide invocation log recorder binding.

Routers and in-process pipeline steps share one recorder bound at platform
lifespan once the store is ready. This avoids ContextVar propagation issues
across FastAPI request tasks while keeping step wrappers store-agnostic.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentomatic.logs.recorder import InvocationLogRecorder

_recorder: InvocationLogRecorder | None = None
_pipeline_name: ContextVar[str | None] = ContextVar(
    "agentomatic_pipeline_log_name",
    default=None,
)

RESOURCE_TYPES = frozenset({"agent", "plugin", "pipeline", "ingestion", "endpoint"})


def set_invocation_log_recorder(recorder: InvocationLogRecorder | None) -> None:
    """Bind the platform-wide invocation log recorder (or clear it)."""
    global _recorder
    _recorder = recorder


def get_invocation_log_recorder() -> InvocationLogRecorder | None:
    """Return the bound recorder, if any."""
    return _recorder


def bind_pipeline_log_name(name: str | None):
    """Set the active pipeline name for nested in-process step logs.

    Returns:
        A context token suitable for :func:`reset_pipeline_log_name`.
    """
    return _pipeline_name.set(name)


def reset_pipeline_log_name(token) -> None:
    """Reset the active pipeline name context."""
    _pipeline_name.reset(token)


def get_pipeline_log_name() -> str | None:
    """Return the active pipeline name for step metadata, if any."""
    return _pipeline_name.get()


def normalize_resource_type(resource_type: str | None) -> str:
    """Return a validated resource type (default ``agent``)."""
    value = (resource_type or "agent").strip().lower()
    if value not in RESOURCE_TYPES:
        raise ValueError(
            f"Invalid resource_type {resource_type!r}; "
            f"expected one of {sorted(RESOURCE_TYPES)}"
        )
    return value
