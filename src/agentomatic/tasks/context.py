"""Execution context handed to task dispatchers.

A :class:`TaskContext` gives running work a safe way to report progress and to
check for cancellation, without coupling dispatchers to the manager internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class TaskContext:
    """Per-execution handle for progress reporting and cancellation checks.

    Args:
        task_id: The task this context belongs to.
        report_fn: Async callback that persists/broadcasts a progress update.
        is_cancelled: Callable returning ``True`` if cancellation was requested.
    """

    def __init__(
        self,
        task_id: str,
        report_fn: Callable[..., Awaitable[None]],
        is_cancelled: Callable[[], bool],
    ) -> None:
        self.task_id = task_id
        self._report_fn = report_fn
        self._is_cancelled = is_cancelled

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
        """Report a progress update for the current task.

        When ``percent`` is omitted but ``current`` and ``total`` are provided,
        the percentage is derived automatically.
        """
        if percent is None and current is not None and total:
            percent = min(100.0, max(0.0, (current / total) * 100.0))
        await self._report_fn(
            percent=percent,
            message=message,
            current=current,
            total=total,
            stage=stage,
            data=data,
        )

    @property
    def cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested for this task."""
        return self._is_cancelled()
