"""ContextVar bridge so nested code can report TaskManager progress.

Dispatchers receive a :class:`~agentomatic.tasks.context.TaskContext` but do
not expose it to nested Python callables (pipeline step functions, graph
nodes). This module binds that context for the duration of a dispatcher run
so helpers like :func:`report_stage` can forward updates to
``GET /api/v1/tasks/{id}/events``.

The bridge is installed automatically when the platform builds its task
manager. Call :func:`install_task_progress_bridge` manually only when wiring
a standalone :class:`~agentomatic.tasks.TaskManager`.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.tasks.context import TaskContext

_task_ctx: ContextVar[Any] = ContextVar("agentomatic_task_ctx", default=None)
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


def reset_task_progress_bridge() -> None:
    """Reset install state (for tests).

    Clears the module-level install flag. Per-manager wrap flags are left
    alone; tests should construct a fresh :class:`TaskManager`.
    """
    global _INSTALLED
    _INSTALLED = False


def install_task_progress_bridge(platform_or_manager: Any) -> None:
    """Wrap registered TaskManager dispatchers to bind :class:`TaskContext`.

    Accepts an :class:`~agentomatic.core.platform.AgentPlatform` or a
    :class:`~agentomatic.tasks.TaskManager`. Idempotent. Safe when there is
    no task manager (no-op). Also wraps :meth:`TaskManager.register_dispatcher`
    so dispatchers registered after install still get the ContextVar binding.
    """
    global _INSTALLED
    manager = platform_or_manager
    if not hasattr(manager, "_dispatchers"):
        manager = getattr(platform_or_manager, "task_manager", None) or getattr(
            platform_or_manager, "_task_manager", None
        )
    if manager is None:
        logger.debug("Task progress bridge skipped — no task manager")
        return
    if getattr(manager, "_agentomatic_task_progress_wrapped", False):
        _INSTALLED = True
        return
    dispatchers = getattr(manager, "_dispatchers", None)
    if not isinstance(dispatchers, dict):
        logger.debug("Task progress bridge skipped — no dispatchers dict")
        return
    for key, dispatcher in list(dispatchers.items()):
        dispatchers[key] = _wrap_dispatcher(dispatcher)

    # Ensure late register_dispatcher calls are also wrapped.
    original_register = getattr(manager, "register_dispatcher", None)
    if callable(original_register) and not getattr(
        original_register, "_agentomatic_progress_wrapped", False
    ):

        def _register_and_wrap(target_type: Any, dispatcher: Any) -> None:
            original_register(target_type, _wrap_dispatcher(dispatcher))

        _register_and_wrap._agentomatic_progress_wrapped = True  # type: ignore[attr-defined]
        manager.register_dispatcher = _register_and_wrap  # type: ignore[method-assign]

    manager._agentomatic_task_progress_wrapped = True
    _INSTALLED = True
    logger.info(
        "Task progress bridge installed ({} dispatcher types)",
        len(dispatchers),
    )
