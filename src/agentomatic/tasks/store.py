"""Pluggable persistence for task records.

The :class:`TaskStore` abstraction lets task status survive beyond a single
request and, with a durable backend, beyond process restarts. The default
:class:`InMemoryTaskStore` keeps records in a bounded, TTL-aware dict — suitable
for single-process deployments and tests. Production multi-worker deployments
should provide a shared backend (e.g. a SQLAlchemy- or Redis-backed store) that
implements the same interface.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from .models import TargetType, TaskRecord, TaskStatus


class TaskStore(ABC):
    """Abstract persistence interface for :class:`TaskRecord` objects."""

    async def initialize(self) -> None:
        """Initialise the backend (create tables, open connections, etc.)."""

    async def close(self) -> None:
        """Release resources held by the backend."""

    @abstractmethod
    async def save(self, record: TaskRecord) -> None:
        """Insert or update a task record."""

    @abstractmethod
    async def get(self, task_id: str) -> TaskRecord | None:
        """Return a task record by id, or ``None`` if unknown."""

    @abstractmethod
    async def list(
        self,
        *,
        status: TaskStatus | None = None,
        target_type: TargetType | None = None,
        target: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaskRecord]:
        """List task records (most recent first) with optional filters."""

    @abstractmethod
    async def count(self, *, status: TaskStatus | None = None) -> int:
        """Return the number of stored records, optionally filtered by status."""

    async def delete(self, task_id: str) -> bool:
        """Delete a task record. Returns ``True`` if it existed."""
        return False


class InMemoryTaskStore(TaskStore):
    """In-memory task store with bounded size and TTL-based eviction.

    Args:
        max_records: Maximum number of terminal records to retain. When the
            cap is exceeded the oldest terminal records are evicted first.
        ttl_seconds: Optional lifetime (in seconds) after which terminal
            records become eligible for eviction. ``None`` disables TTL.
    """

    def __init__(self, *, max_records: int = 10_000, ttl_seconds: float | None = 86_400) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._max_records = max_records
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    async def save(self, record: TaskRecord) -> None:
        async with self._lock:
            self._records[record.id] = record
            self._evict()

    async def get(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    async def list(
        self,
        *,
        status: TaskStatus | None = None,
        target_type: TargetType | None = None,
        target: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaskRecord]:
        records = sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
        if status is not None:
            records = [r for r in records if r.status == status]
        if target_type is not None:
            records = [r for r in records if r.target_type == target_type]
        if target is not None:
            records = [r for r in records if r.target == target]
        return records[offset : offset + limit]

    async def count(self, *, status: TaskStatus | None = None) -> int:
        if status is None:
            return len(self._records)
        return sum(1 for r in self._records.values() if r.status == status)

    async def delete(self, task_id: str) -> bool:
        async with self._lock:
            return self._records.pop(task_id, None) is not None

    def _evict(self) -> None:
        """Evict expired and overflow terminal records (caller holds the lock)."""
        now = time.time()
        if self._ttl_seconds is not None:
            expired = [
                tid
                for tid, rec in self._records.items()
                if rec.status.is_terminal
                and rec.finished_at is not None
                and (now - rec.finished_at) > self._ttl_seconds
            ]
            for tid in expired:
                self._records.pop(tid, None)

        if len(self._records) <= self._max_records:
            return

        terminal = sorted(
            (r for r in self._records.values() if r.status.is_terminal),
            key=lambda r: r.finished_at or r.created_at,
        )
        overflow = len(self._records) - self._max_records
        for rec in terminal[:overflow]:
            self._records.pop(rec.id, None)
