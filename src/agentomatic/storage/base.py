"""Abstract base class for storage backends.

All storage backends must implement this interface.
This ensures you can swap MemoryStore → SQLAlchemy → Redis → MongoDB
without changing any application code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStore(ABC):
    """Protocol for all agentomatic storage backends.

    Subclass this to create custom backends (Redis, MongoDB, DynamoDB, etc.).
    All methods are async for consistency.

    Example::

        class RedisStore(BaseStore):
            async def initialize(self):
                self._redis = aioredis.from_url("redis://localhost")
            async def create_thread(self, thread_id, user_id, agent_name, **kw):
                await self._redis.hset(f"thread:{thread_id}", ...)
                return {...}
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the backend (create tables, open connections, etc.).

        Override this for backends that require async setup.
        Called automatically during platform startup if ``enable_db=True``.
        """

    async def close(self) -> None:
        """Gracefully close connections and release resources.

        Called automatically during platform shutdown.
        """

    async def health_check(self) -> dict[str, Any]:
        """Return backend health information.

        Returns:
            Dict with at least ``{"status": "healthy"|"unhealthy"}``.
        """
        return {"status": "healthy", "backend": self.__class__.__name__}

    # ------------------------------------------------------------------
    # Thread operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new conversation thread.

        Args:
            thread_id: Unique thread identifier.
            user_id: Owner user identifier.
            agent_name: Name of the agent this thread belongs to.
            title: Optional human-readable title.
            metadata: Optional metadata dict.

        Returns:
            Serialized thread dict.
        """

    @abstractmethod
    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Retrieve a thread by ID. Returns ``None`` if not found."""

    @abstractmethod
    async def list_threads(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List threads with optional filtering and pagination."""

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all associated data.

        Returns:
            ``True`` if the thread was deleted, ``False`` if not found.
        """
        return False

    async def update_thread(
        self, thread_id: str, **updates: Any,
    ) -> dict[str, Any] | None:
        """Update thread fields. Returns updated thread or None."""
        return None

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a message to a thread.

        Args:
            thread_id: Thread to add the message to.
            role: Message role (``user``, ``assistant``, ``system``, ``tool``).
            content: Message text content.
            metadata: Optional metadata.

        Returns:
            Serialized message dict.
        """

    @abstractmethod
    async def get_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get messages for a thread, ordered chronologically."""

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
        """Record user feedback on an agent response.

        Default implementation returns a stub. Override for persistence.
        """
        return {
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "rating": rating,
            "feedback_type": feedback_type,
            "status": "not_persisted",
        }

    async def get_feedback(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve stored feedback. Override for persistence."""
        return []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Return backend statistics."""
        return {"backend": self.__class__.__name__}
