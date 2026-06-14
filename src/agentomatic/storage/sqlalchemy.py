"""SQLAlchemy-based persistent storage with connection pooling.

Production-ready backend supporting PostgreSQL, SQLite, MySQL, and any
SQLAlchemy-compatible database.  Uses async engines and session pooling.

Usage::

    from agentomatic.storage import SQLAlchemyStore

    # PostgreSQL
    store = SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db")

    # SQLite (development)
    store = SQLAlchemyStore("sqlite+aiosqlite:///data/platform.db")

    await store.initialize()

    # Wire into platform
    platform = AgentPlatform.from_folder("agents/", store=store)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from loguru import logger
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import BaseStore
from .models import (
    Base,
    CheckpointModel,
    FeedbackModel,
    MessageModel,
    SuspendedStateModel,
    ThreadModel,
)


class SQLAlchemyStore(BaseStore):
    """Production async storage with connection pooling.

    Supports:
    - PostgreSQL (``asyncpg``)
    - SQLite (``aiosqlite``)
    - MySQL (``asyncmy``)
    - Any SQLAlchemy async driver

    Args:
        url: Database connection URL.
        pool_size: Number of persistent connections.
        max_overflow: Max temporary connections above pool_size.
        pool_recycle: Recycle connections after N seconds.
        pool_pre_ping: Test connections before use.
        echo: Log all SQL statements (debug).
    """

    def __init__(
        self,
        url: str = "sqlite+aiosqlite:///data/platform.db",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        echo: bool = False,
    ) -> None:
        engine_kwargs: dict[str, Any] = {"echo": echo}
        # SQLite doesn't support pool_size
        if "sqlite" not in url:
            engine_kwargs.update(
                {
                    "pool_size": pool_size,
                    "max_overflow": max_overflow,
                    "pool_recycle": pool_recycle,
                    "pool_pre_ping": pool_pre_ping,
                }
            )

        self._url = url
        self._engine = create_async_engine(url, **engine_kwargs)
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        _safe_url = url.split("@")[-1] if "@" in url else url
        logger.info(f"🗄️ SQLAlchemy store configured: {_safe_url}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create all database tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("🗄️ Database tables created/verified")

    async def close(self) -> None:
        """Dispose the engine and release all pooled connections."""
        await self._engine.dispose()
        logger.info("🗄️ Database connection pool closed")

    async def health_check(self) -> dict[str, Any]:
        """Verify database connectivity."""
        try:
            async with self._session() as session:
                await session.execute(select(func.count(ThreadModel.id)))
            return {
                "status": "healthy",
                "backend": "SQLAlchemyStore",
                "url": self._url.split("@")[-1] if "@" in self._url else self._url,
            }
        except Exception as exc:
            return {"status": "unhealthy", "backend": "SQLAlchemyStore", "error": str(exc)}

    def _session(self) -> AsyncSession:
        """Get a new async session."""
        return self._session_factory()

    # ------------------------------------------------------------------
    # Thread operations
    # ------------------------------------------------------------------

    async def create_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new conversation thread."""
        async with self._session() as session:
            thread = ThreadModel(
                id=thread_id,
                user_id=user_id,
                agent_name=agent_name,
                title=title,
                metadata_json=metadata or {},
            )
            session.add(thread)
            await session.commit()
            await session.refresh(thread)
            return thread.to_dict()

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Get a thread by ID."""
        async with self._session() as session:
            result = await session.execute(select(ThreadModel).where(ThreadModel.id == thread_id))
            thread = result.scalar_one_or_none()
            return thread.to_dict() if thread else None

    async def list_threads(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List threads with optional filtering and pagination."""
        async with self._session() as session:
            stmt = select(ThreadModel).order_by(ThreadModel.updated_at.desc())
            if agent_name:
                stmt = stmt.where(ThreadModel.agent_name == agent_name)
            if user_id:
                stmt = stmt.where(ThreadModel.user_id == user_id)
            stmt = stmt.limit(limit).offset(offset)
            result = await session.execute(stmt)
            return [t.to_dict() for t in result.scalars().all()]

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all its messages (cascading)."""
        async with self._session() as session:
            result = await session.execute(select(ThreadModel).where(ThreadModel.id == thread_id))
            thread = result.scalar_one_or_none()
            if thread:
                await session.delete(thread)
                await session.commit()
                return True
            return False

    async def update_thread(
        self,
        thread_id: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        """Update thread fields."""
        async with self._session() as session:
            result = await session.execute(select(ThreadModel).where(ThreadModel.id == thread_id))
            thread = result.scalar_one_or_none()
            if not thread:
                return None
            for key, val in updates.items():
                if hasattr(thread, key):
                    setattr(thread, key, val)
            thread.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(thread)
            return cast(dict[str, Any], thread.to_dict())

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a message to a thread."""
        async with self._session() as session:
            msg = MessageModel(
                thread_id=thread_id,
                role=role,
                content=content,
                metadata_json=metadata or {},
            )
            session.add(msg)
            # Update thread message count and timestamp
            result = await session.execute(select(ThreadModel).where(ThreadModel.id == thread_id))
            thread = result.scalar_one_or_none()
            if thread:
                thread.message_count += 1
                thread.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(msg)
            return msg.to_dict()

    async def get_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get messages for a thread, chronologically."""
        async with self._session() as session:
            stmt = (
                select(MessageModel)
                .where(MessageModel.thread_id == thread_id)
                .order_by(MessageModel.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return [m.to_dict() for m in result.scalars().all()]

    # ------------------------------------------------------------------
    # Feedback operations
    # ------------------------------------------------------------------

    async def add_feedback(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        rating: int | None = None,
        comment: str | None = None,
        message_id: int | None = None,
        feedback_type: str = "thumbs",
    ) -> dict[str, Any]:
        """Record user feedback."""
        async with self._session() as session:
            fb = FeedbackModel(
                thread_id=thread_id,
                user_id=user_id,
                agent_name=agent_name,
                rating=rating,
                comment=comment,
                message_id=message_id,
                feedback_type=feedback_type,
            )
            session.add(fb)
            await session.commit()
            await session.refresh(fb)
            return fb.to_dict()

    async def get_feedback(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get feedback, optionally filtered."""
        async with self._session() as session:
            stmt = select(FeedbackModel).order_by(FeedbackModel.created_at.desc())
            if agent_name:
                stmt = stmt.where(FeedbackModel.agent_name == agent_name)
            if user_id:
                stmt = stmt.where(FeedbackModel.user_id == user_id)
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return [f.to_dict() for f in result.scalars().all()]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Return storage statistics."""
        async with self._session() as session:
            thread_count = await session.execute(select(func.count(ThreadModel.id)))
            msg_count = await session.execute(select(func.count(MessageModel.id)))
            fb_count = await session.execute(select(func.count(FeedbackModel.id)))
            return {
                "backend": "SQLAlchemyStore",
                "threads": thread_count.scalar_one(),
                "messages": msg_count.scalar_one(),
                "feedback": fb_count.scalar_one(),
            }

    # ------------------------------------------------------------------
    # Suspended State (HITL) operations
    # ------------------------------------------------------------------

    async def save_suspended_state(
        self,
        approval_id: str,
        thread_id: str,
        agent_name: str,
        node_name: str,
        state_json: dict[str, Any],
    ) -> dict[str, Any]:
        """Save suspended state for human approval."""
        async with self._session() as session:
            suspended = SuspendedStateModel(
                id=approval_id,
                thread_id=thread_id,
                agent_name=agent_name,
                node_name=node_name,
                state_json=state_json,
            )
            session.add(suspended)
            await session.commit()
            await session.refresh(suspended)
            return suspended.to_dict()

    async def get_suspended_state(self, approval_id: str) -> dict[str, Any] | None:
        """Retrieve suspended state by approval ID."""
        async with self._session() as session:
            result = await session.execute(
                select(SuspendedStateModel).where(SuspendedStateModel.id == approval_id)
            )
            suspended = result.scalar_one_or_none()
            return suspended.to_dict() if suspended else None

    async def list_suspended_states(
        self,
        *,
        thread_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List suspended states, optionally filtered."""
        async with self._session() as session:
            stmt = select(SuspendedStateModel).order_by(SuspendedStateModel.created_at.desc())
            if thread_id:
                stmt = stmt.where(SuspendedStateModel.thread_id == thread_id)
            if agent_name:
                stmt = stmt.where(SuspendedStateModel.agent_name == agent_name)
            result = await session.execute(stmt)
            return [s.to_dict() for s in result.scalars().all()]

    async def delete_suspended_state(self, approval_id: str) -> bool:
        """Delete suspended state by approval ID."""
        async with self._session() as session:
            result = await session.execute(
                select(SuspendedStateModel).where(SuspendedStateModel.id == approval_id)
            )
            suspended = result.scalar_one_or_none()
            if suspended:
                await session.delete(suspended)
                await session.commit()
                return True
            return False

    async def cleanup_expired_states(self) -> int:
        """Delete expired suspended states from the database."""
        async with self._session() as session:
            now = datetime.now(UTC)
            # Count expired first, then bulk delete
            count_stmt = select(func.count(SuspendedStateModel.id)).where(
                SuspendedStateModel.expires_at.isnot(None),
                SuspendedStateModel.expires_at < now,
            )
            count_result = await session.execute(count_stmt)
            count = count_result.scalar_one()
            if count > 0:
                del_stmt = delete(SuspendedStateModel).where(
                    SuspendedStateModel.expires_at.isnot(None),
                    SuspendedStateModel.expires_at < now,
                )
                await session.execute(del_stmt)
                await session.commit()
            return count

    # ------------------------------------------------------------------
    # Forking operations
    # ------------------------------------------------------------------

    async def fork_thread(
        self,
        parent_thread_id: str,
        message_index: int,
        new_thread_id: str,
        *,
        title: str | None = None,
    ) -> dict[str, Any] | None:
        """Fork a thread at a specific message index."""
        async with self._session() as session:
            # 1. Get parent thread
            result = await session.execute(
                select(ThreadModel).where(ThreadModel.id == parent_thread_id)
            )
            parent = result.scalar_one_or_none()
            if not parent:
                return None

            # 2. Get parent messages chronologically
            msg_result = await session.execute(
                select(MessageModel)
                .where(MessageModel.thread_id == parent_thread_id)
                .order_by(MessageModel.created_at.asc())
            )
            parent_messages = msg_result.scalars().all()

            # 3. Create the new forked thread
            new_title = title or f"Fork of {parent.title or parent_thread_id}"
            parent_meta = parent.metadata_json or {}

            forked_thread = ThreadModel(
                id=new_thread_id,
                user_id=parent.user_id,
                agent_name=parent.agent_name,
                title=new_title,
                metadata_json=parent_meta,
                parent_thread_id=parent_thread_id,
                fork_message_index=message_index,
            )
            session.add(forked_thread)

            # 4. Clone messages up to message_index
            forked_count = 0
            for i, msg in enumerate(parent_messages):
                if i <= message_index:
                    new_msg = MessageModel(
                        thread_id=new_thread_id,
                        role=msg.role,
                        content=msg.content,
                        metadata_json=msg.metadata_json or {},
                    )
                    session.add(new_msg)
                    forked_count += 1

            forked_thread.message_count = forked_count
            await session.commit()
            await session.refresh(forked_thread)
            return forked_thread.to_dict()

    async def get_thread_lineage(self, thread_id: str) -> dict[str, Any]:
        """Get the full lineage tree for a thread."""
        async with self._session() as session:
            # Walk up to find ancestors
            ancestors: list[dict[str, Any]] = []
            current_id = thread_id
            for _ in range(100):  # Guard against cycles
                result = await session.execute(
                    select(ThreadModel).where(ThreadModel.id == current_id)
                )
                current = result.scalar_one_or_none()
                if not current or not current.parent_thread_id:
                    break
                parent_result = await session.execute(
                    select(ThreadModel).where(ThreadModel.id == current.parent_thread_id)
                )
                parent = parent_result.scalar_one_or_none()
                if parent:
                    ancestors.insert(0, parent.to_dict())
                current_id = current.parent_thread_id

            # Find direct descendants
            desc_result = await session.execute(
                select(ThreadModel).where(ThreadModel.parent_thread_id == thread_id)
            )
            descendants = [t.to_dict() for t in desc_result.scalars().all()]

            return {
                "thread_id": thread_id,
                "ancestors": ancestors,
                "descendants": descendants,
            }

    # ------------------------------------------------------------------
    # Checkpointer operations
    # ------------------------------------------------------------------

    async def get_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve a checkpoint."""
        async with self._session() as session:
            if not checkpoint_id:
                stmt = (
                    select(CheckpointModel)
                    .where(
                        CheckpointModel.thread_id == thread_id,
                        CheckpointModel.checkpoint_ns == checkpoint_ns,
                    )
                    .order_by(CheckpointModel.created_at.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                cp = result.scalar_one_or_none()
                return cp.to_dict() if cp else None

            stmt = select(CheckpointModel).where(
                CheckpointModel.thread_id == thread_id,
                CheckpointModel.checkpoint_ns == checkpoint_ns,
                CheckpointModel.checkpoint_id == checkpoint_id,
            )
            result = await session.execute(stmt)
            cp = result.scalar_one_or_none()
            return cp.to_dict() if cp else None

    async def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Save a checkpoint."""
        async with self._session() as session:
            stmt = select(CheckpointModel).where(
                CheckpointModel.thread_id == thread_id,
                CheckpointModel.checkpoint_ns == checkpoint_ns,
                CheckpointModel.checkpoint_id == checkpoint_id,
            )
            result = await session.execute(stmt)
            cp = result.scalar_one_or_none()
            if cp:
                cp.parent_checkpoint_id = parent_checkpoint_id
                cp.checkpoint_json = checkpoint
                cp.metadata_json = metadata
                cp.created_at = datetime.now(UTC)
            else:
                cp = CheckpointModel(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    checkpoint_json=checkpoint,
                    metadata_json=metadata,
                )
                session.add(cp)
            await session.commit()

    async def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List checkpoints."""
        async with self._session() as session:
            stmt = (
                select(CheckpointModel)
                .where(
                    CheckpointModel.thread_id == thread_id,
                    CheckpointModel.checkpoint_ns == checkpoint_ns,
                )
                .order_by(CheckpointModel.created_at.desc())
            )
            if before:
                before_stmt = select(CheckpointModel).where(
                    CheckpointModel.thread_id == thread_id,
                    CheckpointModel.checkpoint_ns == checkpoint_ns,
                    CheckpointModel.checkpoint_id == before,
                )
                before_result = await session.execute(before_stmt)
                before_cp = before_result.scalar_one_or_none()
                if before_cp:
                    stmt = stmt.where(CheckpointModel.created_at < before_cp.created_at)

            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            return [c.to_dict() for c in result.scalars().all()]
