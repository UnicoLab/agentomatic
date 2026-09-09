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
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentomatic.tasks.models import TaskEvent

#: Events retained per task before the oldest are discarded.
DEFAULT_MAX_EVENTS_PER_TASK = 512

#: Tasks retained in the log before the least recently used are discarded.
DEFAULT_MAX_TASKS = 1024


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

    Retention is capped twice over — per task and across tasks — so a
    long-running platform cannot accumulate history without limit. Eviction is
    least-recently-appended.

    Args:
        max_events_per_task: Events retained per task.
        max_tasks: Number of tasks retained.
    """

    def __init__(
        self,
        *,
        max_events_per_task: int = DEFAULT_MAX_EVENTS_PER_TASK,
        max_tasks: int = DEFAULT_MAX_TASKS,
    ) -> None:
        self._max_events_per_task = max(1, max_events_per_task)
        self._max_tasks = max(1, max_tasks)
        self._events: OrderedDict[str, deque[TaskEvent]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def append(self, event: TaskEvent) -> None:
        """Record one event, evicting the oldest history when full."""
        async with self._lock:
            bucket = self._events.get(event.task_id)
            if bucket is None:
                bucket = deque(maxlen=self._max_events_per_task)
                self._events[event.task_id] = bucket
            bucket.append(event)
            self._events.move_to_end(event.task_id)
            while len(self._events) > self._max_tasks:
                self._events.popitem(last=False)

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
            if not bucket:
                return 0
            return bucket[0].sequence

    async def latest_sequence(self, task_id: str) -> int:
        """Return the highest recorded sequence, or 0 when empty."""
        async with self._lock:
            bucket = self._events.get(task_id)
            if not bucket:
                return 0
            return bucket[-1].sequence

    async def drop(self, task_id: str) -> None:
        """Discard a task's retained history."""
        async with self._lock:
            self._events.pop(task_id, None)


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
