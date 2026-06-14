"""In-memory storage backend for development and testing."""

from __future__ import annotations

from datetime import UTC, datetime
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
        self._suspended_states: dict[str, dict[str, Any]] = {}
        self._checkpoints: dict[tuple[str, str, str], dict[str, Any]] = {}

    async def create_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        thread = {
            "id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "title": title,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "parent_thread_id": None,
            "fork_message_index": None,
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
            # Clean up orphaned suspended states
            orphan_ids = [
                sid for sid, s in self._suspended_states.items() if s.get("thread_id") == thread_id
            ]
            for sid in orphan_ids:
                del self._suspended_states[sid]
            return True
        return False

    async def update_thread(
        self,
        thread_id: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        thread = self._threads.get(thread_id)
        if thread:
            thread.update(updates)
            thread["updated_at"] = datetime.now(UTC).isoformat()
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
        if thread_id not in self._threads:
            raise ValueError(f"Thread '{thread_id}' does not exist")
        msg = {
            "id": len(self._messages.get(thread_id, [])) + 1,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if thread_id not in self._messages:
            self._messages[thread_id] = []
        self._messages[thread_id].append(msg)
        self._threads[thread_id]["message_count"] += 1
        self._threads[thread_id]["updated_at"] = datetime.now(UTC).isoformat()
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
            "created_at": datetime.now(UTC).isoformat(),
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
        from datetime import timedelta

        now = datetime.now(UTC)
        suspended = {
            "id": approval_id,
            "thread_id": thread_id,
            "agent_name": agent_name,
            "node_name": node_name,
            "state_snapshot": state_json,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=7)).isoformat(),
        }
        self._suspended_states[approval_id] = suspended
        return suspended

    async def get_suspended_state(self, approval_id: str) -> dict[str, Any] | None:
        """Retrieve suspended state, returning None if expired."""
        state = self._suspended_states.get(approval_id)
        if state and state.get("expires_at"):
            now = datetime.now(UTC).isoformat()
            if state["expires_at"] < now:
                # Auto-cleanup expired state
                del self._suspended_states[approval_id]
                return None
        return state

    async def list_suspended_states(
        self,
        *,
        thread_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        states = list(self._suspended_states.values())
        if thread_id:
            states = [s for s in states if s.get("thread_id") == thread_id]
        if agent_name:
            states = [s for s in states if s.get("agent_name") == agent_name]
        return states

    async def delete_suspended_state(self, approval_id: str) -> bool:
        if approval_id in self._suspended_states:
            del self._suspended_states[approval_id]
            return True
        return False

    async def cleanup_expired_states(self) -> int:
        """Delete expired suspended states from memory."""
        now = datetime.now(UTC)
        expired_ids = []
        for sid, state in self._suspended_states.items():
            expires_at = state.get("expires_at")
            if expires_at:
                try:
                    exp_dt = datetime.fromisoformat(expires_at)
                    if exp_dt < now:
                        expired_ids.append(sid)
                except (ValueError, TypeError):
                    pass
        for sid in expired_ids:
            del self._suspended_states[sid]
        return len(expired_ids)

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
        parent_thread = self._threads.get(parent_thread_id)
        if not parent_thread:
            return None
        if message_index < 0:
            return None

        now = datetime.now(UTC).isoformat()
        forked_thread = {
            "id": new_thread_id,
            "user_id": parent_thread["user_id"],
            "agent_name": parent_thread["agent_name"],
            "title": title or f"Fork of {parent_thread.get('title') or parent_thread_id}",
            "metadata": {
                **parent_thread.get("metadata", {}),
            },
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "parent_thread_id": parent_thread_id,
            "fork_message_index": message_index,
        }
        self._threads[new_thread_id] = forked_thread

        parent_messages = self._messages.get(parent_thread_id, [])
        forked_messages: list[dict[str, Any]] = []
        for i, msg in enumerate(parent_messages):
            if i <= message_index:
                forked_msg = {
                    **msg,
                    "id": len(forked_messages) + 1,
                    "thread_id": new_thread_id,
                }
                forked_messages.append(forked_msg)

        self._messages[new_thread_id] = forked_messages
        forked_thread["message_count"] = len(forked_messages)
        return forked_thread

    async def get_thread_lineage(self, thread_id: str) -> dict[str, Any]:
        """Get the full lineage tree for a thread."""
        # Walk up to find ancestors
        ancestors: list[dict[str, Any]] = []
        current_id = thread_id
        for _ in range(100):  # Guard against cycles
            current = self._threads.get(current_id)
            if not current:
                break
            parent_id = current.get("parent_thread_id")
            if not parent_id:
                break
            parent = self._threads.get(parent_id)
            if parent:
                ancestors.insert(0, parent)
            current_id = parent_id

        # Find direct descendants
        descendants = [t for t in self._threads.values() if t.get("parent_thread_id") == thread_id]

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
        if not checkpoint_id:
            thread_checkpoints = [
                cp
                for cp in self._checkpoints.values()
                if cp["thread_id"] == thread_id and cp["checkpoint_ns"] == checkpoint_ns
            ]
            if not thread_checkpoints:
                return None
            thread_checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
            return thread_checkpoints[0]
        return self._checkpoints.get((thread_id, checkpoint_ns, checkpoint_id))

    async def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        key = (thread_id, checkpoint_ns, checkpoint_id)
        self._checkpoints[key] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "checkpoint": checkpoint,
            "metadata": metadata,
            "created_at": datetime.now(UTC).isoformat(),
        }

    async def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        cps = [
            cp
            for cp in self._checkpoints.values()
            if cp["thread_id"] == thread_id and cp["checkpoint_ns"] == checkpoint_ns
        ]
        cps.sort(key=lambda x: x["created_at"], reverse=True)
        if before:
            found_idx = -1
            for idx, cp in enumerate(cps):
                if cp["checkpoint_id"] == before:
                    found_idx = idx
                    break
            if found_idx != -1:
                cps = cps[found_idx + 1 :]
        if limit is not None:
            cps = cps[:limit]
        return cps
