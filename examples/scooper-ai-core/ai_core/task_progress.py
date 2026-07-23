"""ContextVar bridge so agent/pipeline code can report TaskManager progress.

Agentomatic dispatchers receive a :class:`~agentomatic.tasks.context.TaskContext`
but do not expose it to nested Python callables (pipeline step functions,
graph nodes). This module binds that context for the duration of a dispatcher
run so helpers like :func:`report_stage` can forward updates to
``GET /api/v1/tasks/{id}/events``.

Call :func:`install_task_progress_bridge` once after the platform is built
(see ``main.py``).
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.tasks.context import TaskContext

_task_ctx: ContextVar[Any] = ContextVar("scooper_task_ctx", default=None)
_INSTALLED = False


def bind_task_context(ctx: TaskContext | None) -> Any:
    """Bind *ctx* for the current async task; return a reset token."""
    return _task_ctx.set(ctx)


def reset_task_context(token: Any) -> None:
    """Reset the ContextVar using the token from :func:`bind_task_context`."""
    _task_ctx.reset(token)


def get_task_context() -> TaskContext | None:
    """Return the active task context, if any."""
    return _task_ctx.get()


async def report_stage(
    stage: str,
    *,
    percent: float | None = None,
    current: int | None = None,
    total: int | None = None,
    message: str = "",
) -> None:
    """Report a stage update when running under a task context (no-op otherwise)."""
    ctx = get_task_context()
    if ctx is None:
        return
    await ctx.report(
        stage=stage,
        percent=percent,
        current=current,
        total=total,
        message=message or stage,
    )


def report_stage_sync(
    stage: str,
    *,
    percent: float | None = None,
    current: int | None = None,
    total: int | None = None,
    message: str = "",
) -> None:
    """Schedule :func:`report_stage` from a sync graph node (no-op if no loop/ctx)."""
    ctx = get_task_context()
    if ctx is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(
        report_stage(
            stage,
            percent=percent,
            current=current,
            total=total,
            message=message,
        )
    )


def _wrap_dispatcher(dispatcher: Any) -> Any:
    """Wrap a TaskManager dispatcher so nested code sees the TaskContext."""

    async def wrapped(target: str, payload: Any, ctx: Any) -> Any:
        token = bind_task_context(ctx)
        try:
            return await dispatcher(target, payload, ctx)
        finally:
            reset_task_context(token)

    return wrapped


def install_task_progress_bridge(platform: Any) -> None:
    """Wrap registered TaskManager dispatchers to bind :class:`TaskContext`.

    Idempotent. Safe when the platform has no task manager (no-op).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    manager = getattr(platform, "task_manager", None) or getattr(
        platform, "_task_manager", None
    )
    if manager is None:
        logger.debug("Task progress bridge skipped — no task manager")
        return
    dispatchers = getattr(manager, "_dispatchers", None)
    if not isinstance(dispatchers, dict) or not dispatchers:
        logger.debug("Task progress bridge skipped — no dispatchers yet")
        return
    if getattr(manager, "_scooper_task_progress_wrapped", False):
        _INSTALLED = True
        return
    for key, dispatcher in list(dispatchers.items()):
        dispatchers[key] = _wrap_dispatcher(dispatcher)
    manager._scooper_task_progress_wrapped = True
    _INSTALLED = True
    logger.info(
        "Task progress bridge installed ({} dispatcher types)",
        len(dispatchers),
    )
