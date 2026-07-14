"""The unified task manager.

:class:`TaskManager` is the single orchestration point for running any platform
resource as a task. It provides:

* **async / background execution** — submit returns immediately with a task id;
* **a bounded in-process queue** — concurrency is capped by a semaphore so
  long-running work does not exhaust the event loop;
* **status polling** — every task has a durable, uniform :class:`TaskRecord`;
* **live progress** — dispatchers report progress that is broadcast over SSE;
* **cancellation** — running asyncio tasks are cancelled cooperatively;
* **batch** — a list of inputs is fanned out with per-item progress;
* **webhooks** — an optional ``callback_url`` is POSTed the final record.

The manager is transport-agnostic; :mod:`agentomatic.tasks.routes` exposes it
over HTTP, and the A2A endpoints reuse it for real task lifecycles.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from .context import TaskContext
from .models import (
    TargetType,
    TaskEvent,
    TaskProgress,
    TaskRecord,
    TaskStatus,
)
from .store import InMemoryTaskStore, TaskStore

if TYPE_CHECKING:
    from .dispatchers import Dispatcher


class TaskManager:
    """Run and track platform resources as uniform, cancellable tasks.

    Args:
        store: Persistence backend for task records (defaults to in-memory).
        max_concurrency: Maximum number of tasks executing simultaneously.
            Additional submissions queue until a slot frees up.
        default_batch_concurrency: Default per-task concurrency for batch items.
    """

    def __init__(
        self,
        store: TaskStore | None = None,
        *,
        max_concurrency: int = 8,
        default_batch_concurrency: int = 4,
    ) -> None:
        self.store = store or InMemoryTaskStore()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency
        self._default_batch_concurrency = default_batch_concurrency
        self._dispatchers: dict[TargetType, Dispatcher] = {}
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._cancel_requested: set[str] = set()
        self._subscribers: dict[str, list[asyncio.Queue[TaskEvent]]] = {}

    # ------------------------------------------------------------------
    # Registration & lifecycle
    # ------------------------------------------------------------------

    def register_dispatcher(self, target_type: TargetType, dispatcher: Dispatcher) -> None:
        """Register the runner used for a given resource type."""
        self._dispatchers[target_type] = dispatcher

    @property
    def supported_targets(self) -> list[str]:
        """Return the target types that have a registered dispatcher."""
        return sorted(t.value for t in self._dispatchers)

    async def initialize(self) -> None:
        """Initialise the underlying store."""
        await self.store.initialize()

    async def shutdown(self) -> None:
        """Cancel in-flight tasks and close the store."""
        for task_id in list(self._running):
            await self.cancel(task_id)
        await self.store.close()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(
        self,
        target_type: TargetType | str,
        target: str,
        *,
        input: Any = None,
        batch: list[Any] | None = None,
        mode: str = "async",
        metadata: dict[str, Any] | None = None,
        callback_url: str | None = None,
        parent_id: str | None = None,
        batch_concurrency: int | None = None,
    ) -> TaskRecord:
        """Submit work and return immediately with a ``QUEUED`` task record.

        Raises:
            ValueError: If ``target_type`` has no registered dispatcher.
        """
        ttype = TargetType(target_type)
        if ttype not in self._dispatchers:
            raise ValueError(
                f"No dispatcher registered for target_type '{ttype.value}'. "
                f"Supported: {self.supported_targets}"
            )

        record = TaskRecord(
            target_type=ttype,
            target=target,
            input=input,
            batch=batch,
            mode="batch" if batch is not None else mode,
            metadata=metadata or {},
            callback_url=callback_url,
            parent_id=parent_id,
        )
        await self.store.save(record)
        await self._emit(record, "queued")

        aio_task = asyncio.create_task(self._run(record, batch_concurrency))
        self._running[record.id] = aio_task
        aio_task.add_done_callback(lambda _t, tid=record.id: self._running.pop(tid, None))
        return record

    async def submit_and_wait(
        self,
        target_type: TargetType | str,
        target: str,
        *,
        input: Any = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> TaskRecord:
        """Submit a task and await its terminal state (synchronous mode)."""
        record = await self.submit(target_type, target, input=input, mode="sync", **kwargs)
        aio_task = self._running.get(record.id)
        if aio_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(aio_task), timeout=timeout)
            except TimeoutError:
                pass
        refreshed = await self.store.get(record.id)
        return refreshed or record

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def get(self, task_id: str) -> TaskRecord | None:
        """Return the current record for a task."""
        return await self.store.get(task_id)

    async def list(self, **filters: Any) -> list[TaskRecord]:
        """List task records with optional filters (see :class:`TaskStore`)."""
        return await self.store.list(**filters)

    async def count(self, **filters: Any) -> int:
        """Count stored task records with optional status filter."""
        return await self.store.count(**filters)

    async def stats(self) -> dict[str, Any]:
        """Return a snapshot of task counts and executor capacity.

        Useful for status dashboards: total tasks, a per-status breakdown,
        the number currently executing, and the configured concurrency.
        """
        by_status = {
            status.value: await self.store.count(status=status) for status in TaskStatus
        }
        return {
            "total": await self.store.count(),
            "by_status": by_status,
            "running": len(self._running),
            "max_concurrency": self._max_concurrency,
            "supported_targets": self.supported_targets,
        }

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel(self, task_id: str) -> bool:
        """Request cancellation of a queued or running task.

        Returns:
            ``True`` if the task existed and was not already terminal.
        """
        record = await self.store.get(task_id)
        if record is None or record.status.is_terminal:
            return False
        self._cancel_requested.add(task_id)
        aio_task = self._running.get(task_id)
        if aio_task is not None:
            aio_task.cancel()
        return True

    # ------------------------------------------------------------------
    # Progress streaming
    # ------------------------------------------------------------------

    async def subscribe(self, task_id: str) -> asyncio.Queue[TaskEvent]:
        """Return a queue that receives live :class:`TaskEvent` updates."""
        queue: asyncio.Queue[TaskEvent] = asyncio.Queue(maxsize=256)
        self._subscribers.setdefault(task_id, []).append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue[TaskEvent]) -> None:
        """Remove a previously registered progress subscriber."""
        subs = self._subscribers.get(task_id)
        if subs and queue in subs:
            subs.remove(queue)
        if subs is not None and not subs:
            self._subscribers.pop(task_id, None)

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _run(self, record: TaskRecord, batch_concurrency: int | None) -> None:
        """Execute a task record, honouring concurrency and cancellation."""
        if record.id in self._cancel_requested:
            await self._finalize(record, TaskStatus.CANCELLED, error="Cancelled before start")
            return

        async with self._semaphore:
            if record.id in self._cancel_requested:
                await self._finalize(record, TaskStatus.CANCELLED, error="Cancelled before start")
                return

            record.status = TaskStatus.RUNNING
            record.started_at = time.time()
            await self.store.save(record)
            await self._emit(record, "started")

            dispatcher = self._dispatchers[record.target_type]
            ctx = TaskContext(
                record.id,
                self._make_reporter(record),
                lambda: record.id in self._cancel_requested,
            )

            try:
                if record.batch is not None:
                    result = await self._run_batch(record, dispatcher, ctx, batch_concurrency)
                else:
                    result = await dispatcher(record.target, record.input, ctx)
                await self._finalize(record, TaskStatus.SUCCEEDED, result=result)
            except asyncio.CancelledError:
                await self._finalize(record, TaskStatus.CANCELLED, error="Cancelled")
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as task error
                logger.exception(f"Task {record.id} failed: {exc}")
                await self._finalize(record, TaskStatus.FAILED, error=str(exc))

    async def _run_batch(
        self,
        record: TaskRecord,
        dispatcher: Dispatcher,
        ctx: TaskContext,
        batch_concurrency: int | None,
    ) -> list[Any]:
        """Run every item in a batch with bounded concurrency and progress."""
        items = record.batch or []
        total = len(items)
        results: list[Any] = [None] * total
        done = 0
        sem = asyncio.Semaphore(batch_concurrency or self._default_batch_concurrency)
        lock = asyncio.Lock()

        async def run_item(index: int, payload: Any) -> None:
            nonlocal done
            async with sem:
                if ctx.cancelled:
                    raise asyncio.CancelledError
                try:
                    results[index] = await dispatcher(record.target, payload, ctx)
                except Exception as exc:  # noqa: BLE001 - collect per-item errors
                    results[index] = {"error": str(exc)}
                async with lock:
                    done += 1
                    await ctx.report(
                        current=done,
                        total=total,
                        message=f"Completed {done}/{total} batch items",
                    )

        await asyncio.gather(*(run_item(i, p) for i, p in enumerate(items)))
        return results

    def _make_reporter(self, record: TaskRecord) -> Any:
        """Build the progress-report callback bound to ``record``."""

        async def report(
            *,
            percent: float | None,
            message: str,
            current: int | None,
            total: int | None,
            stage: str,
            data: dict[str, Any],
        ) -> None:
            record.progress = TaskProgress(
                percent=percent if percent is not None else record.progress.percent,
                message=message or record.progress.message,
                current=current if current is not None else record.progress.current,
                total=total if total is not None else record.progress.total,
                stage=stage or record.progress.stage,
            )
            await self.store.save(record)
            await self._emit(record, "progress", data=data)

        return report

    async def _finalize(
        self,
        record: TaskRecord,
        status: TaskStatus,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Persist the terminal state, notify subscribers, and fire webhooks."""
        record.status = status
        record.finished_at = time.time()
        record.result = result
        record.error = error
        if status == TaskStatus.SUCCEEDED:
            record.progress = TaskProgress(
                percent=100.0,
                message="Completed",
                current=record.progress.total or record.progress.current,
                total=record.progress.total,
            )
        await self.store.save(record)
        self._cancel_requested.discard(record.id)
        await self._emit(record, status.value)
        await self._fire_webhook(record)

    async def _emit(
        self, record: TaskRecord, event: str, *, data: dict[str, Any] | None = None
    ) -> None:
        """Broadcast an event to all subscribers of a task."""
        subs = self._subscribers.get(record.id)
        if not subs:
            return
        evt = TaskEvent(
            task_id=record.id,
            event=event,
            status=record.status,
            progress=record.progress,
            data=data or {},
        )
        for queue in list(subs):
            try:
                queue.put_nowait(evt)
            except asyncio.QueueFull:  # pragma: no cover - slow consumer
                logger.debug(f"Dropping task event for slow subscriber on {record.id}")

    async def _fire_webhook(self, record: TaskRecord) -> None:
        """POST the final record to ``callback_url`` (best-effort)."""
        if not record.callback_url:
            return
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(record.callback_url, json=record.public_dict())
        except Exception as exc:  # noqa: BLE001 - webhooks must never break tasks
            logger.warning(f"Webhook to {record.callback_url} failed for {record.id}: {exc}")
