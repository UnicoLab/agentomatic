"""SQLAlchemy-based persistent storage with connection pooling."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base, FeedbackModel, MessageModel, ThreadModel


class SQLAlchemyStore:
    """Production-ready async storage with connection pooling.
    
    Usage:
        store = SQLAlchemyStore(
            url="postgresql+asyncpg://user:pass@localhost:5432/mydb",
            pool_size=10,
        )
        await store.initialize()  # Create tables
        
        # Use in platform:
        platform = AgentPlatform.from_folder("agents/")
        # pass store to router_factory for thread management
    """

    def __init__(
        self,
        url: str = "sqlite+aiosqlite:///data/platform.db",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,
        echo: bool = False,
    ) -> None:
        engine_kwargs: dict[str, Any] = {
            "echo": echo,
        }
        # SQLite doesn't support pool_size
        if "sqlite" not in url:
            engine_kwargs.update({
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_recycle": pool_recycle,
                "pool_pre_ping": True,
            })
        
        self._engine = create_async_engine(url, **engine_kwargs)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession,
        )
        logger.info(f"🗄️ SQLAlchemy store initialized: {url.split('@')[-1] if '@' in url else url}")

    async def initialize(self) -> None:
        """Create all tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("🗄️ Database tables created")

    async def close(self) -> None:
        """Close the engine and connection pool."""
        await self._engine.dispose()
        logger.info("🗄️ Database connection pool closed")

    def _session(self) -> AsyncSession:
        """Get a new session."""
        return self._session_factory()

    # --- Thread Operations ---

    async def create_thread(
        self, thread_id: str, user_id: str, agent_name: str,
        title: str | None = None, metadata: dict | None = None,
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
            result = await session.execute(
                select(ThreadModel).where(ThreadModel.id == thread_id)
            )
            thread = result.scalar_one_or_none()
            return thread.to_dict() if thread else None

    async def list_threads(
        self,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List threads with optional filtering."""
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
        """Delete a thread and all its messages."""
        async with self._session() as session:
            result = await session.execute(
                select(ThreadModel).where(ThreadModel.id == thread_id)
            )
            thread = result.scalar_one_or_none()
            if thread:
                await session.delete(thread)
                await session.commit()
                return True
            return False

    # --- Message Operations ---

    async def add_message(
        self, thread_id: str, role: str, content: str, **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a message to a thread."""
        async with self._session() as session:
            msg = MessageModel(
                thread_id=thread_id,
                role=role,
                content=content,
                metadata_json=kwargs.get("metadata", {}),
            )
            session.add(msg)
            # Update thread message count and timestamp
            result = await session.execute(
                select(ThreadModel).where(ThreadModel.id == thread_id)
            )
            thread = result.scalar_one_or_none()
            if thread:
                thread.message_count += 1
                thread.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(msg)
            return msg.to_dict()

    async def get_messages(
        self, thread_id: str, limit: int = 100, offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get messages for a thread."""
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

    # --- Feedback Operations ---

    async def add_feedback(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        rating: int | None = None,
        comment: str | None = None,
        message_id: int | None = None,
        feedback_type: str = "thumbs",
    ) -> dict[str, Any]:
        """Add user feedback."""
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
        self, agent_name: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get feedback, optionally filtered by agent."""
        async with self._session() as session:
            stmt = select(FeedbackModel).order_by(FeedbackModel.created_at.desc())
            if agent_name:
                stmt = stmt.where(FeedbackModel.agent_name == agent_name)
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return [f.to_dict() for f in result.scalars().all()]

    # --- Stats ---

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        async with self._session() as session:
            thread_count = await session.execute(select(func.count(ThreadModel.id)))
            msg_count = await session.execute(select(func.count(MessageModel.id)))
            fb_count = await session.execute(select(func.count(FeedbackModel.id)))
            return {
                "threads": thread_count.scalar_one(),
                "messages": msg_count.scalar_one(),
                "feedback": fb_count.scalar_one(),
            }
