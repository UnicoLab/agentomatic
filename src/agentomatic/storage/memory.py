"""In-memory thread and message store for development."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class MemoryStore:
    """Simple in-memory store for threads and messages."""

    def __init__(self) -> None:
        self._threads: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}

    async def create_thread(
        self, thread_id: str, user_id: str, agent_name: str,
    ) -> dict[str, Any]:
        """Create a new conversation thread."""
        thread = {
            "id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message_count": 0,
        }
        self._threads[thread_id] = thread
        self._messages[thread_id] = []
        return thread

    async def add_message(
        self, thread_id: str, role: str, content: str, **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a message to a thread."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        if thread_id not in self._messages:
            self._messages[thread_id] = []
        self._messages[thread_id].append(msg)
        if thread_id in self._threads:
            self._threads[thread_id]["message_count"] += 1
        return msg

    async def list_threads(
        self, agent_name: str | None = None, user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List threads, optionally filtered by agent or user."""
        threads = list(self._threads.values())
        if agent_name:
            threads = [t for t in threads if t.get("agent_name") == agent_name]
        if user_id:
            threads = [t for t in threads if t.get("user_id") == user_id]
        return threads

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Get a thread by ID."""
        return self._threads.get(thread_id)

    async def get_messages(self, thread_id: str) -> list[dict[str, Any]]:
        """Get all messages for a thread."""
        return self._messages.get(thread_id, [])
