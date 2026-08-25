"""Tests for the task-progress ContextVar bridge."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentomatic.tasks.context import TaskContext
from agentomatic.tasks.manager import TaskManager
from agentomatic.tasks.models import TargetType
from agentomatic.tasks.progress import (
    get_task_context,
    install_task_progress_bridge,
    report_stage,
    report_stage_sync,
    reset_task_progress_bridge,
)


@pytest.fixture(autouse=True)
def _reset_bridge() -> None:
    reset_task_progress_bridge()


@pytest.mark.asyncio
async def test_report_stage_noop_without_context() -> None:
    """report_stage is a no-op when no TaskContext is bound."""
    await report_stage("idle")  # must not raise


@pytest.mark.asyncio
async def test_nested_callable_sees_bound_context() -> None:
    """Wrapped dispatchers expose TaskContext to nested async code."""
    seen: list[str] = []

    async def nested() -> str:
        await report_stage("nested", percent=50.0, message="halfway")
        ctx = get_task_context()
        assert ctx is not None
        seen.append(ctx.task_id)
        return "ok"

    async def dispatcher(target: str, payload: Any, ctx: TaskContext) -> Any:
        return await nested()

    manager = TaskManager()
    manager.register_dispatcher(TargetType.AGENT, dispatcher)
    install_task_progress_bridge(manager)

    record = await manager.submit_and_wait(TargetType.AGENT, "demo", input={"x": 1})
    assert record.status.value == "succeeded"
    assert seen == [record.id]


@pytest.mark.asyncio
async def test_report_stage_forwards_to_context() -> None:
    """report_stage forwards stage updates to the bound TaskContext."""
    reports: list[dict[str, Any]] = []

    async def report_fn(**kwargs: Any) -> None:
        reports.append(kwargs)

    ctx = TaskContext(task_id="t1", report_fn=report_fn, is_cancelled=lambda: False)
    from agentomatic.tasks.progress import bind_task_context, reset_task_context

    token = bind_task_context(ctx)
    try:
        await report_stage("encode", percent=10.0, message="encoding")
    finally:
        reset_task_context(token)
    assert reports
    assert reports[0]["stage"] == "encode"
    assert reports[0]["percent"] == 10.0


@pytest.mark.asyncio
async def test_late_register_dispatcher_also_wrapped() -> None:
    """Dispatchers registered after install still bind TaskContext."""
    seen: list[str] = []

    async def nested() -> str:
        ctx = get_task_context()
        assert ctx is not None
        seen.append(ctx.task_id)
        return "ok"

    async def dispatcher(target: str, payload: Any, ctx: TaskContext) -> Any:
        return await nested()

    manager = TaskManager()
    install_task_progress_bridge(manager)
    manager.register_dispatcher(TargetType.PLUGIN, dispatcher)
    record = await manager.submit_and_wait(TargetType.PLUGIN, "late", input={})
    assert record.status.value == "succeeded"
    assert seen == [record.id]


@pytest.mark.asyncio
async def test_report_stage_sync_schedules() -> None:
    """report_stage_sync schedules an async report on the running loop."""
    reports: list[dict[str, Any]] = []

    async def report_fn(**kwargs: Any) -> None:
        reports.append(kwargs)

    ctx = TaskContext(task_id="t2", report_fn=report_fn, is_cancelled=lambda: False)
    from agentomatic.tasks.progress import bind_task_context, reset_task_context

    token = bind_task_context(ctx)
    try:
        report_stage_sync("sync-stage", percent=25.0)
        import asyncio

        await asyncio.sleep(0.05)
    finally:
        reset_task_context(token)
    assert reports
    assert reports[0]["stage"] == "sync-stage"


@pytest.mark.asyncio
async def test_report_stage_sync_keeps_strong_task_reference() -> None:
    """The scheduled task must be held strongly until it completes.

    asyncio only keeps a *weak* reference to a task created via
    ``loop.create_task`` with no reference kept elsewhere — without an
    explicit strong reference, the task can be garbage-collected mid-flight,
    silently dropping the progress report.
    """
    import gc

    from agentomatic.tasks.progress import _background_tasks, bind_task_context, reset_task_context

    async def slow_report_fn(**kwargs: Any) -> None:
        await asyncio.sleep(0.05)

    ctx = TaskContext(task_id="t3", report_fn=slow_report_fn, is_cancelled=lambda: False)
    token = bind_task_context(ctx)
    try:
        report_stage_sync("gc-stage")
        # Force a collection pass immediately — before our fix this could
        # collect the fire-and-forget task before it ever ran.
        gc.collect()
        assert len(_background_tasks) == 1
        pending = next(iter(_background_tasks))
        assert not pending.done()
        await pending
    finally:
        reset_task_context(token)
    # The done-callback must remove it from the tracking set afterwards.
    assert len(_background_tasks) == 0
