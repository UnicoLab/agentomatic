"""In-memory storage backend for development and testing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import BaseStore


class MemoryStore(BaseStore):
    """Fast in-memory store — perfect for development and unit tests.

    All data is lost when the process stops. For production, use
    :class:`~agentomatic.storage.sqlalchemy.SQLAlchemyStore` or
    implement your own :class:`BaseStore` subclass.
    """

    def __init__(self) -> None:
        self._threads: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._feedback: list[dict[str, Any]] = []

    async def create_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        thread = {
            "id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "title": title,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
        self._threads[thread_id] = thread
        self._messages[thread_id] = []
        return thread

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        return self._threads.get(thread_id)

    async def list_threads(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        threads = list(self._threads.values())
        if agent_name:
            threads = [t for t in threads if t.get("agent_name") == agent_name]
        if user_id:
            threads = [t for t in threads if t.get("user_id") == user_id]
        return threads[offset : offset + limit]

    async def delete_thread(self, thread_id: str) -> bool:
        if thread_id in self._threads:
            del self._threads[thread_id]
            self._messages.pop(thread_id, None)
            return True
        return False

    async def update_thread(
        self, thread_id: str, **updates: Any,
    ) -> dict[str, Any] | None:
        thread = self._threads.get(thread_id)
        if thread:
            thread.update(updates)
            thread["updated_at"] = datetime.now(timezone.utc).isoformat()
            return thread
        return None

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        msg = {
            "id": len(self._messages.get(thread_id, [])) + 1,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if thread_id not in self._messages:
            self._messages[thread_id] = []
        self._messages[thread_id].append(msg)
        if thread_id in self._threads:
            self._threads[thread_id]["message_count"] += 1
            self._threads[thread_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        return msg

    async def get_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        messages = self._messages.get(thread_id, [])
        return messages[offset : offset + limit]

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
        fb = {
            "id": len(self._feedback) + 1,
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "rating": rating,
            "comment": comment,
            "message_id": message_id,
            "feedback_type": feedback_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._feedback.append(fb)
        return fb

    async def get_feedback(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        items = self._feedback
        if agent_name:
            items = [f for f in items if f.get("agent_name") == agent_name]
        if user_id:
            items = [f for f in items if f.get("user_id") == user_id]
        return items[:limit]

    async def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "MemoryStore",
            "threads": len(self._threads),
            "messages": sum(len(msgs) for msgs in self._messages.values()),
            "feedback": len(self._feedback),
        }

    async def health_check(self) -> dict[str, Any]:
        return {"status": "healthy", "backend": "MemoryStore", "threads": len(self._threads)}
