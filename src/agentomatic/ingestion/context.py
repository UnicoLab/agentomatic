"""Execution context passed to an ingestor's ``ingest`` method.

The context lets a user's ingestion code report progress and check for
cancellation without depending on the task subsystem. When an ingestor runs as
a task the manager supplies a real, broadcasting context; when it runs
synchronously a :class:`NullIngestionContext` is used.

The interface intentionally mirrors :class:`agentomatic.tasks.context.TaskContext`
so the task manager's context can be passed straight through.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IngestionContext(Protocol):
    """Progress-reporting + cancellation handle for ingestion runs."""

    async def report(
        self,
        *,
        percent: float | None = None,
        message: str = "",
        current: int | None = None,
        total: int | None = None,
        stage: str = "",
        **data: Any,
    ) -> None:
        """Report a progress update."""
        ...

    @property
    def cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested."""
        ...


class NullIngestionContext:
    """No-op context used for synchronous / direct ingestion runs."""

    async def report(
        self,
        *,
        percent: float | None = None,
        message: str = "",
        current: int | None = None,
        total: int | None = None,
        stage: str = "",
        **data: Any,
    ) -> None:
        """Discard the progress update."""

    @property
    def cancelled(self) -> bool:
        """Never cancelled."""
        return False
