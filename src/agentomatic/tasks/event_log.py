"""Replay buffers for task progress events.

A live SSE subscriber is not enough to make streaming reliable: the moment a
client's connection drops, every event emitted while it was away is gone, and
the client can only re-read the task's *current* state — never the steps it
missed. An event log keeps a bounded, ordered history per task so a
reconnecting client can resume exactly where it left off.

The default implementation is in-process and bounded. Register a shared one
(Redis, a database, a log service) with
:func:`~agentomatic.tasks.event_log.register_event_log_provider` when tasks
must be resumable across workers or replicas — Agentomatic ships the
interface, not a vendor client.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.tasks.models import TaskEvent

#: Events retained per task before the oldest are discarded.
DEFAULT_MAX_EVENTS_PER_TASK = 512

#: Tasks retained in the log before the least recently used are discarded.
DEFAULT_MAX_TASKS = 1024

#: Bytes retained across all tasks. Most events are small, but a progress
#: payload can carry a pipeline checkpoint's ``sub_result``, so a count-only
#: bound is not a memory bound.
DEFAULT_MAX_BYTES_TOTAL = 32 * 1024 * 1024


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable to read.
        default: Value used when unset, unparseable, or not positive.

    Returns:
        The configured value, or ``default``.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer — using {default}")
        return default
    if value <= 0:
        logger.warning(f"{name}={value} must be positive — using {default}")
        return default
    return value


def event_log_from_env() -> TaskEventLog:
    """Build the task event log a deployment's environment asks for.

    The platform is env-driven everywhere else, so retention has to be
    tunable without editing Python — an operator sizing a container needs to
    cap it, and someone who wants no replay at all needs to say so.

    Reads ``AGENTOMATIC_TASK_EVENT_LOG`` (a provider name; ``none``/``off``
    selects the null log), plus ``AGENTOMATIC_TASK_EVENTS_PER_TASK``,
    ``AGENTOMATIC_TASK_EVENT_TASKS`` and ``AGENTOMATIC_TASK_EVENT_MAX_MB``.

    Returns:
        The configured :class:`TaskEventLog`.
    """
    provider = (os.getenv("AGENTOMATIC_TASK_EVENT_LOG") or "memory").strip().lower()
    if provider in {"none", "off", "disabled", "null"}:
        return NullTaskEventLog()

    options: dict[str, Any] = {}
    if provider == "memory":
        options = {
            "max_events_per_task": _env_int(
                "AGENTOMATIC_TASK_EVENTS_PER_TASK", DEFAULT_MAX_EVENTS_PER_TASK
            ),
            "max_tasks": _env_int("AGENTOMATIC_TASK_EVENT_TASKS", DEFAULT_MAX_TASKS),
            "max_bytes_total": _env_int(
                "AGENTOMATIC_TASK_EVENT_MAX_MB", DEFAULT_MAX_BYTES_TOTAL // (1024 * 1024)
            )
            * 1024
            * 1024,
        }
    try:
        return create_event_log(provider, **options)
    except ValueError as exc:
        # An unknown provider must not take the platform down; replay is a
        # convenience, and the default still works.
        logger.warning(f"{exc} Falling back to the in-memory event log.")
        return InMemoryTaskEventLog()


def estimate_event_size(event: TaskEvent) -> int:
    """Approximate an event's in-memory footprint in bytes.

    Cheap in the common case: most events carry no ``data``, so this is a
    constant. When ``data`` is present it is measured, because that is the
    field that can carry a large payload.

    Args:
        event: Event to size.

    Returns:
        An estimated byte count, never less than the fixed overhead.
    """
    overhead = 512
    if not event.data:
        return overhead
    try:
        return overhead + len(json.dumps(event.data, default=str))
    except (TypeError, ValueError):  # pragma: no cover - exotic payloads
        return overhead


class TaskEventLog(ABC):
    """Ordered, replayable history of the events emitted for each task."""

    @abstractmethod
    async def append(self, event: TaskEvent) -> None:
        """Record one event.

        Args:
            event: The event to retain. Its ``sequence`` is already assigned.
        """

    @abstractmethod
    async def replay(self, task_id: str, *, after: int = 0) -> list[TaskEvent]:
        """Return retained events for a task, oldest first.

        Args:
            task_id: Task whose history to read.
            after: Return only events with ``sequence`` greater than this.

        Returns:
            The matching events in ascending sequence order.
        """

    @abstractmethod
    async def earliest_sequence(self, task_id: str) -> int:
        """Return the lowest sequence still retained for a task.

        A caller that asks to resume from before this point cannot be served a
        complete history and must be told the stream is truncated.

        Args:
            task_id: Task to inspect.

        Returns:
            The lowest retained sequence, or 0 when nothing is retained.
        """

    @abstractmethod
    async def latest_sequence(self, task_id: str) -> int:
        """Return the highest sequence recorded for a task.

        A new subscriber uses this as its starting cursor, so it can resume
        precisely even if it drops before the first live event arrives.

        Args:
            task_id: Task to inspect.

        Returns:
            The highest recorded sequence, or 0 when nothing is retained.
        """

    @abstractmethod
    async def drop(self, task_id: str) -> None:
        """Discard a task's retained history.

        Args:
            task_id: Task whose history to forget.
        """


class InMemoryTaskEventLog(TaskEventLog):
    """Bounded in-process event history.

    Retention is capped three ways — events per task, tasks, and total bytes.
    The byte ceiling is the one that protects a container's memory limit: a
    progress event can carry a pipeline checkpoint's ``sub_result``, so
    counting events alone bounds nothing in particular. Eviction is
    least-recently-appended.

    Args:
        max_events_per_task: Events retained per task.
        max_tasks: Number of tasks retained.
        max_bytes_total: Estimated bytes retained across all tasks.
    """

    def __init__(
        self,
        *,
        max_events_per_task: int = DEFAULT_MAX_EVENTS_PER_TASK,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_bytes_total: int = DEFAULT_MAX_BYTES_TOTAL,
    ) -> None:
        self._max_events_per_task = max(1, max_events_per_task)
        self._max_tasks = max(1, max_tasks)
        self._max_bytes_total = max(1, max_bytes_total)
        # No ``maxlen``: an implicit drop would evict an event without its
        # size leaving the running total, and the accounting would drift up
        # until nothing could be retained at all.
        self._events: OrderedDict[str, deque[TaskEvent]] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._total_bytes = 0
        self._lock = asyncio.Lock()

    async def append(self, event: TaskEvent) -> None:
        """Record one event, evicting the oldest history when full."""
        size = estimate_event_size(event)
        async with self._lock:
            task_id = event.task_id
            bucket = self._events.setdefault(task_id, deque())
            bucket.append(event)
            self._sizes[task_id] = self._sizes.get(task_id, 0) + size
            self._total_bytes += size
            self._events.move_to_end(task_id)

            while len(bucket) > 1 and len(bucket) > self._max_events_per_task:
                self._drop_oldest(task_id)
            self._trim_total(keep=task_id)

    def _drop_oldest(self, task_id: str) -> None:
        """Remove a task's oldest event, keeping the byte totals honest.

        ``estimate_event_size`` is deterministic, so what is subtracted here
        is exactly what was added when the event was appended.
        """
        dropped = estimate_event_size(self._events[task_id].popleft())
        self._sizes[task_id] = self._sizes.get(task_id, 0) - dropped
        self._total_bytes -= dropped

    def _trim_bytes(self, task_id: str, limit: int) -> None:
        """Drop a task's oldest events until it fits ``limit`` bytes.

        The newest event always survives: a single oversized one should still
        be replayable rather than silently dropped.

        Args:
            task_id: Task to trim.
            limit: Byte ceiling.
        """
        bucket = self._events.get(task_id)
        if bucket is None:
            return
        while len(bucket) > 1 and self._sizes.get(task_id, 0) > limit:
            self._drop_oldest(task_id)

    def _trim_total(self, *, keep: str) -> None:
        """Evict least-recently-used tasks until the whole log fits.

        Args:
            keep: Task that must survive — it is the one just written to.
        """
        while len(self._events) > 1 and (
            len(self._events) > self._max_tasks or self._total_bytes > self._max_bytes_total
        ):
            oldest = next(iter(self._events))
            if oldest == keep:
                # ``keep`` was just moved to the end, so reaching it means
                # every other task's history is already gone.
                break
            self._forget(oldest)

        # Nothing left to evict but still over budget: the surviving task
        # holds the bytes, so trim its history rather than quietly ignoring
        # the ceiling.
        if self._total_bytes > self._max_bytes_total:
            self._trim_bytes(keep, self._max_bytes_total)

    def _forget(self, task_id: str) -> None:
        """Remove one task's history and its byte accounting."""
        self._events.pop(task_id, None)
        self._total_bytes = max(0, self._total_bytes - self._sizes.pop(task_id, 0))

    async def replay(self, task_id: str, *, after: int = 0) -> list[TaskEvent]:
        """Return retained events for a task with sequence greater than ``after``."""
        async with self._lock:
            bucket = self._events.get(task_id)
            if not bucket:
                return []
            return [evt for evt in bucket if evt.sequence > after]

    async def earliest_sequence(self, task_id: str) -> int:
        """Return the lowest sequence still retained, or 0 when empty."""
        async with self._lock:
            bucket = self._events.get(task_id)
            return bucket[0].sequence if bucket else 0

    async def latest_sequence(self, task_id: str) -> int:
        """Return the highest recorded sequence, or 0 when empty."""
        async with self._lock:
            bucket = self._events.get(task_id)
            return bucket[-1].sequence if bucket else 0

    async def drop(self, task_id: str) -> None:
        """Discard a task's retained history."""
        async with self._lock:
            self._forget(task_id)

    async def stats(self) -> dict[str, int]:
        """Return current occupancy, for tests and operational visibility.

        Returns:
            Task count, retained event count and total estimated bytes.
        """
        async with self._lock:
            return {
                "tasks": len(self._events),
                "events": sum(len(bucket) for bucket in self._events.values()),
                "bytes": self._total_bytes,
            }


class NullTaskEventLog(TaskEventLog):
    """Event log that retains nothing.

    Use it to opt out of replay entirely. Reconnecting clients still receive
    the task's current snapshot; they simply cannot recover missed steps.
    """

    async def append(self, event: TaskEvent) -> None:
        """Discard the event."""

    async def replay(self, task_id: str, *, after: int = 0) -> list[TaskEvent]:
        """Return no history."""
        return []

    async def earliest_sequence(self, task_id: str) -> int:
        """Report that nothing is retained."""
        return 0

    async def latest_sequence(self, task_id: str) -> int:
        """Report that nothing is retained."""
        return 0

    async def drop(self, task_id: str) -> None:
        """Nothing is retained, so nothing is dropped."""


#: Factories for named event-log backends, keyed by provider name.
EventLogFactory = Callable[..., TaskEventLog]

_PROVIDERS: dict[str, EventLogFactory] = {}


def register_event_log_provider(name: str, factory: EventLogFactory) -> None:
    """Register a named event-log backend.

    Cross-worker resumption needs a log every worker can read. Rather than
    ship a client for one vendor, Agentomatic lets you register any backend::

        register_event_log_provider("redis", lambda **cfg: MyRedisEventLog(**cfg))

    Args:
        name: Provider name used by :func:`create_event_log`.
        factory: Callable returning a :class:`TaskEventLog`.
    """
    _PROVIDERS[name] = factory


def available_event_log_providers() -> list[str]:
    """Return the registered provider names, built-ins included.

    Returns:
        Sorted provider names accepted by :func:`create_event_log`.
    """
    return sorted({"memory", "null", *_PROVIDERS})


def create_event_log(name: str = "memory", **options: Any) -> TaskEventLog:
    """Build an event log by provider name.

    Args:
        name: ``"memory"``, ``"null"``, or a registered provider.
        **options: Passed to the provider factory.

    Returns:
        A :class:`TaskEventLog`.

    Raises:
        ValueError: When ``name`` is not a known provider.
    """
    factory = _PROVIDERS.get(name)
    if factory is not None:
        return factory(**options)
    if name == "memory":
        return InMemoryTaskEventLog(**options)
    if name == "null":
        return NullTaskEventLog()
    raise ValueError(
        f"Unknown task event log provider '{name}'. "
        f"Known: {', '.join(available_event_log_providers())}. "
        "Register your own with register_event_log_provider()."
    )


class StreamResumption:
    """The outcome of resolving a client's resume point.

    Attributes:
        replay: Events the client missed, oldest first.
        truncated: True when the requested resume point fell off the end of
            the retained history, so ``replay`` does not cover every missed
            event. Clients should treat the following snapshot as
            authoritative rather than assume continuity.
        cursor: Sequence the client is caught up to after ``replay``.
    """

    __slots__ = ("cursor", "replay", "truncated")

    def __init__(self, replay: list[TaskEvent], *, truncated: bool, cursor: int) -> None:
        self.replay = replay
        self.truncated = truncated
        self.cursor = cursor


async def resolve_resumption(log: TaskEventLog, task_id: str, *, after: int) -> StreamResumption:
    """Work out what a reconnecting client still needs.

    Args:
        log: Event log to read.
        task_id: Task being resumed.
        after: Last sequence the client successfully processed (0 = new client).

    Returns:
        A :class:`StreamResumption` describing the gap.
    """
    if after <= 0:
        return StreamResumption([], truncated=False, cursor=0)
    replay = await log.replay(task_id, after=after)
    earliest = await log.earliest_sequence(task_id)
    # ``earliest > after + 1`` means the events between the client's cursor and
    # the oldest retained event have already been evicted.
    truncated = bool(earliest and earliest > after + 1)
    cursor = replay[-1].sequence if replay else after
    return StreamResumption(replay, truncated=truncated, cursor=cursor)


def parse_last_event_id(raw: str | None) -> int:
    """Parse an SSE ``Last-Event-ID`` value into a sequence number.

    The header is client-supplied and echoed from a previous stream, so it is
    treated as a hint: anything unparseable means "start from the beginning"
    rather than an error.

    Args:
        raw: Header value, or None.

    Returns:
        The parsed sequence, or 0 when absent or malformed.
    """
    if not raw:
        return 0
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0
