"""SQLAlchemy-backed :class:`~agentomatic.tasks.store.TaskStore`.

A durable, multi-worker task store: task records survive process restarts and
are shared across workers/replicas through a common database (SQLite for
development, PostgreSQL/MySQL for production). It implements the exact
:class:`TaskStore` interface, so it is a drop-in replacement for the default
:class:`InMemoryTaskStore`.

This module imports SQLAlchemy lazily — importing :mod:`agentomatic.tasks`
never requires the optional ``db`` extra. The friendly install hint is only
raised when you actually instantiate :class:`SQLAlchemyTaskStore` without the
dependency present.

Usage::

    from agentomatic import AgentPlatform
    from agentomatic.tasks import SQLAlchemyTaskStore

    store = SQLAlchemyTaskStore("postgresql+asyncpg://user:pass@localhost/db")
    platform = AgentPlatform.from_folder("agents/", task_store=store)

Install with::

    pip install "agentomatic[db]"            # SQLite
    pip install "agentomatic[db-postgres]"    # PostgreSQL
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .models import TargetType, TaskRecord, TaskStatus
from .store import TaskStore

try:  # Optional dependency — only needed to *use* this store.
    from sqlalchemy import JSON, Float, String, delete, func, select
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    _SQLALCHEMY_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    _SQLALCHEMY_IMPORT_ERROR = exc


# Cache of (Base, Model) per table name so repeated stores with the same table
# reuse the mapped class (avoids duplicate-table registration errors), while
# different table names get independent metadata.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_model(table_name: str) -> tuple[Any, Any]:
    """Return ``(Base, Model)`` for ``table_name``, building it once."""
    cached = _MODEL_CACHE.get(table_name)
    if cached is not None:
        return cached

    class _Base(DeclarativeBase):
        pass

    class TaskRow(_Base):
        __tablename__ = table_name

        id: Mapped[str] = mapped_column(String(64), primary_key=True)
        target_type: Mapped[str] = mapped_column(String(32), index=True)
        target: Mapped[str] = mapped_column(String(255), index=True)
        mode: Mapped[str] = mapped_column(String(16), default="async")
        status: Mapped[str] = mapped_column(String(16), index=True)
        created_at: Mapped[float] = mapped_column(Float, index=True)
        finished_at: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
        parent_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
        payload: Mapped[dict] = mapped_column(JSON)

    _MODEL_CACHE[table_name] = (_Base, TaskRow)
    return _Base, TaskRow


class SQLAlchemyTaskStore(TaskStore):
    """Durable, multi-worker task store backed by SQLAlchemy (async).

    Supports any SQLAlchemy async driver: SQLite (``aiosqlite``), PostgreSQL
    (``asyncpg``), MySQL (``asyncmy``), etc. Task records are stored as an
    indexed JSON payload so the schema is forward-compatible with changes to
    :class:`TaskRecord`.

    Args:
        url: Database connection URL. Defaults to a local SQLite file.
        table_name: Name of the tasks table.
        pool_size: Persistent connections (ignored for SQLite).
        max_overflow: Temporary connections above ``pool_size`` (non-SQLite).
        pool_recycle: Recycle connections after N seconds (non-SQLite).
        pool_pre_ping: Test connections before use (non-SQLite).
        echo: Log all SQL statements (debug).
        ttl_seconds: Lifetime after which terminal records are eligible for
            eviction. ``None`` disables TTL-based purging.
        max_records: Soft cap on total records; oldest terminal records are
            evicted first when exceeded. ``None`` disables the cap.
        eviction_interval: Run eviction at most once every N ``save`` calls.
        engine: Reuse an existing async engine instead of creating one (e.g.
            share an agent's ``DatabaseConnection``); the engine is then not
            disposed on :meth:`close`.
    """

    def __init__(
        self,
        url: str = "sqlite+aiosqlite:///data/tasks.db",
        *,
        table_name: str = "agentomatic_tasks",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        echo: bool = False,
        ttl_seconds: float | None = 604_800,  # 7 days
        max_records: int | None = 100_000,
        eviction_interval: int = 200,
        engine: Any = None,
    ) -> None:
        if _SQLALCHEMY_IMPORT_ERROR is not None:
            raise ImportError(
                "SQLAlchemyTaskStore requires SQLAlchemy. Install it with: "
                'pip install "agentomatic[db]" (SQLite) or '
                '"agentomatic[db-postgres]" (PostgreSQL).'
            ) from _SQLALCHEMY_IMPORT_ERROR

        self._url = url
        self._table_name = table_name
        self._ttl_seconds = ttl_seconds
        self._max_records = max_records
        self._eviction_interval = max(1, eviction_interval)
        self._save_count = 0
        self._lock = asyncio.Lock()

        self._base, self._model = _get_model(table_name)

        if engine is not None:
            self._owns_engine = False
            self._engine = engine
        else:
            engine_kwargs: dict[str, Any] = {"echo": echo}
            if "sqlite" not in url:
                engine_kwargs.update(
                    {
                        "pool_size": pool_size,
                        "max_overflow": max_overflow,
                        "pool_recycle": pool_recycle,
                        "pool_pre_ping": pool_pre_ping,
                    }
                )
            self._owns_engine = True
            self._engine = create_async_engine(url, **engine_kwargs)

        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )
        safe_url = url.split("@")[-1] if "@" in url else url
        logger.info(f"🧵 SQLAlchemyTaskStore configured: {safe_url} (table={table_name})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the tasks table if it does not exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(self._base.metadata.create_all)
        logger.info("🧵 Task table created/verified")

    async def close(self) -> None:
        """Dispose the engine (only if this store created it)."""
        if self._owns_engine:
            await self._engine.dispose()

    async def health_check(self) -> dict[str, Any]:
        """Verify database connectivity."""
        try:
            async with self._session_factory() as session:
                await session.execute(select(func.count()).select_from(self._model))
            return {"status": "healthy", "backend": "SQLAlchemyTaskStore"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "unhealthy", "backend": "SQLAlchemyTaskStore", "error": str(exc)}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def save(self, record: TaskRecord) -> None:
        columns = self._to_columns(record)
        async with self._session_factory() as session:
            row = await session.get(self._model, record.id)
            if row is None:
                session.add(self._model(**columns))
            else:
                for key, value in columns.items():
                    setattr(row, key, value)
            await session.commit()
        await self._maybe_evict()

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._session_factory() as session:
            row = await session.get(self._model, task_id)
            return self._from_row(row) if row is not None else None

    async def list(
        self,
        *,
        status: TaskStatus | None = None,
        target_type: TargetType | None = None,
        target: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaskRecord]:
        stmt = select(self._model).order_by(self._model.created_at.desc())
        if status is not None:
            stmt = stmt.where(self._model.status == status.value)
        if target_type is not None:
            stmt = stmt.where(self._model.target_type == target_type.value)
        if target is not None:
            stmt = stmt.where(self._model.target == target)
        stmt = stmt.limit(limit).offset(offset)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [self._from_row(row) for row in result.scalars().all()]

    async def count(self, *, status: TaskStatus | None = None) -> int:
        stmt = select(func.count()).select_from(self._model)
        if status is not None:
            stmt = stmt.where(self._model.status == status.value)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return int(result.scalar_one())

    async def delete(self, task_id: str) -> bool:
        async with self._session_factory() as session:
            row = await session.get(self._model, task_id)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    async def purge_expired(self) -> int:
        """Delete TTL-expired terminal records and trim overflow.

        Returns:
            The number of records deleted.
        """
        import time

        terminal = [s.value for s in TaskStatus if s.is_terminal]
        deleted = 0
        async with self._session_factory() as session:
            if self._ttl_seconds is not None:
                cutoff = time.time() - self._ttl_seconds
                stmt = delete(self._model).where(
                    self._model.status.in_(terminal),
                    self._model.finished_at.isnot(None),
                    self._model.finished_at < cutoff,
                )
                result = await session.execute(stmt)
                deleted += getattr(result, "rowcount", 0) or 0

            if self._max_records is not None:
                total = (
                    await session.execute(select(func.count()).select_from(self._model))
                ).scalar_one()
                overflow = int(total) - self._max_records
                if overflow > 0:
                    # Oldest terminal records first.
                    victims = (
                        (
                            await session.execute(
                                select(self._model.id)
                                .where(self._model.status.in_(terminal))
                                .order_by(self._model.created_at.asc())
                                .limit(overflow)
                            )
                        )
                        .scalars()
                        .all()
                    )
                    if victims:
                        await session.execute(
                            delete(self._model).where(self._model.id.in_(victims))
                        )
                        deleted += len(victims)

            await session.commit()
        if deleted:
            logger.debug(f"SQLAlchemyTaskStore: purged {deleted} record(s)")
        return deleted

    async def _maybe_evict(self) -> None:
        """Run eviction at most once every ``eviction_interval`` saves."""
        if self._ttl_seconds is None and self._max_records is None:
            return
        async with self._lock:
            self._save_count += 1
            if self._save_count % self._eviction_interval != 0:
                return
        try:
            await self.purge_expired()
        except Exception as exc:  # noqa: BLE001 - eviction must never break saves
            logger.warning(f"SQLAlchemyTaskStore: eviction failed: {exc}")

    # ------------------------------------------------------------------
    # (De)serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_columns(record: TaskRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "target_type": record.target_type.value,
            "target": record.target,
            "mode": record.mode,
            "status": record.status.value,
            "created_at": record.created_at,
            "finished_at": record.finished_at,
            "parent_id": record.parent_id,
            "payload": record.model_dump(mode="json"),
        }

    @staticmethod
    def _from_row(row: Any) -> TaskRecord:
        return TaskRecord.model_validate(row.payload)
