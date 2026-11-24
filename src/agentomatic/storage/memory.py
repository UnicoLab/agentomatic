"""In-memory storage backend for development and testing."""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any

from .base import BaseStore

_DEFAULT_CHECKPOINT_LIST_LIMIT = 1000
"""Cap applied to list_checkpoints() when no explicit limit is given."""


def _validate_bound(name: str, value: int | None) -> int | None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def _evict_ordered(store: OrderedDict[Any, Any], limit: int | None, *, keep: Any = None) -> None:
    if limit is None or len(store) <= limit:
        return
    overflow = len(store) - limit
    for key in list(store.keys()):
        if overflow <= 0:
            break
        if key == keep:
            continue
        store.pop(key, None)
        overflow -= 1


class MemoryStore(BaseStore):
    """Fast in-memory store — perfect for development and unit tests.

    All data is lost when the process stops. For production, use
    :class:`~agentomatic.storage.sqlalchemy.SQLAlchemyStore` or
    implement your own :class:`BaseStore` subclass.

    Args:
        max_threads: Optional cap on stored conversation threads. Once it is
            reached, creating or forking a thread evicts the least recently
            used one (with its messages, checkpoints, feedback and suspended
            states), so a long-running process cannot grow without bound.
            ``None`` (the default) keeps every thread unless expired by TTL.
        max_messages_per_thread: Optional cap on stored messages per thread.
        max_feedback: Optional cap on stored feedback entries.
        max_suspended_states: Optional cap on stored HITL suspended states.
        max_checkpoints: Optional cap on stored LangGraph checkpoints.
        max_invocation_logs: Optional cap on stored invocation logs.
        max_log_analyses: Optional cap on stored log analyses.
        max_optimization_runs: Optional cap on stored optimization runs.
        ttl_seconds: Optional expiration age in seconds for stored items
            (defaults to 24h; ``None`` disables TTL eviction).
    """

    def __init__(
        self,
        *,
        max_threads: int | None = None,
        max_messages_per_thread: int | None = 200,
        max_feedback: int | None = 1000,
        max_suspended_states: int | None = 500,
        max_checkpoints: int | None = 1000,
        max_invocation_logs: int | None = 1000,
        max_log_analyses: int | None = 200,
        max_optimization_runs: int | None = 200,
        ttl_seconds: float | None = 86_400.0,
    ) -> None:
        self.max_threads = _validate_bound("max_threads", max_threads)
        self.max_messages_per_thread = _validate_bound(
            "max_messages_per_thread", max_messages_per_thread
        )
        self.max_feedback = _validate_bound("max_feedback", max_feedback)
        self.max_suspended_states = _validate_bound("max_suspended_states", max_suspended_states)
        self.max_checkpoints = _validate_bound("max_checkpoints", max_checkpoints)
        self.max_invocation_logs = _validate_bound("max_invocation_logs", max_invocation_logs)
        self.max_log_analyses = _validate_bound("max_log_analyses", max_log_analyses)
        self.max_optimization_runs = _validate_bound(
            "max_optimization_runs", max_optimization_runs
        )
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds

        self._threads: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._msg_seq: dict[str, int] = {}
        self._feedback: list[dict[str, Any]] = []
        self._feedback_seq: int = 0
        self._suspended_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._checkpoints: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
        self._invocation_logs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._log_analyses: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._optimization_runs: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _is_expired(self, iso_ts: str | None, now: datetime) -> bool:
        if self.ttl_seconds is None:
            return False
        dt = _parse_iso(iso_ts)
        if dt is None:
            return False
        return (now - dt).total_seconds() >= self.ttl_seconds

    def _purge_expired_dict(
        self,
        store: OrderedDict[Any, dict[str, Any]],
        ts_field: str,
        *,
        keep: Any = None,
    ) -> None:
        if self.ttl_seconds is None or not store:
            return
        now = datetime.now(UTC)
        expired = [
            key
            for key, entry in store.items()
            if key != keep and self._is_expired(entry.get(ts_field), now)
        ]
        for key in expired:
            store.pop(key, None)

    async def _purge_expired_threads(self, *, keep: str | None = None) -> None:
        if self.ttl_seconds is None or not self._threads:
            return
        now = datetime.now(UTC)
        expired = [
            tid
            for tid, thread in self._threads.items()
            if tid != keep and self._is_expired(thread.get("updated_at"), now)
        ]
        for tid in expired:
            await self.delete_thread(tid)

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
        self._threads.move_to_end(thread_id)
        self._messages[thread_id] = []
        self._msg_seq[thread_id] = 0
        await self._evict_overflow(keep=thread_id)
        return thread

    async def _evict_overflow(self, *, keep: str) -> None:
        """Purge expired threads and drop LRU threads beyond :attr:`max_threads`."""
        await self._purge_expired_threads(keep=keep)
        if self.max_threads is None or len(self._threads) <= self.max_threads:
            return
        overflow = len(self._threads) - self.max_threads
        for tid in list(self._threads.keys()):
            if overflow <= 0:
                break
            if tid == keep:
                continue
            await self.delete_thread(tid)
            overflow -= 1

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        thread = self._threads.get(thread_id)
        if thread is None:
            return None
        if self._is_expired(thread.get("updated_at"), datetime.now(UTC)):
            await self.delete_thread(thread_id)
            return None
        self._threads.move_to_end(thread_id)
        return thread

    async def list_threads(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        await self._purge_expired_threads()
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
            self._msg_seq.pop(thread_id, None)
            # Clean up orphaned suspended states
            orphan_ids = [
                sid for sid, s in self._suspended_states.items() if s.get("thread_id") == thread_id
            ]
            for sid in orphan_ids:
                del self._suspended_states[sid]
            # Clean up orphaned checkpoints (keyed by (thread_id, ns, checkpoint_id)).
            orphan_cp_keys = [key for key in self._checkpoints if key[0] == thread_id]
            for key in orphan_cp_keys:
                del self._checkpoints[key]
            # Clean up orphaned feedback entries.
            self._feedback = [fb for fb in self._feedback if fb.get("thread_id") != thread_id]
            return True
        return False

    async def update_thread(
        self,
        thread_id: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        thread = await self.get_thread(thread_id)
        if thread:
            thread.update(updates)
            thread["updated_at"] = datetime.now(UTC).isoformat()
            self._threads.move_to_end(thread_id)
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
        thread = await self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Thread '{thread_id}' does not exist")
        seq = self._msg_seq.get(thread_id, len(self._messages.get(thread_id, []))) + 1
        self._msg_seq[thread_id] = seq
        msg = {
            "id": seq,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat(),
        }
        msgs = self._messages.setdefault(thread_id, [])
        msgs.append(msg)
        if self.max_messages_per_thread is not None and len(msgs) > self.max_messages_per_thread:
            del msgs[: len(msgs) - self.max_messages_per_thread]
        thread["message_count"] += 1
        thread["updated_at"] = datetime.now(UTC).isoformat()
        self._threads.move_to_end(thread_id)
        return msg

    async def get_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if await self.get_thread(thread_id) is None:
            return []
        messages = self._messages.get(thread_id, [])
        return messages[offset : offset + limit]

    def _purge_expired_feedback(self) -> None:
        if self.ttl_seconds is None or not self._feedback:
            return
        now = datetime.now(UTC)
        self._feedback = [
            fb for fb in self._feedback if not self._is_expired(fb.get("created_at"), now)
        ]

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
        query: str = "",
        response: str = "",
        correction: str | None = None,
    ) -> dict[str, Any]:
        self._purge_expired_feedback()
        self._feedback_seq += 1
        fb = {
            "id": self._feedback_seq,
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "rating": rating,
            "comment": comment,
            "message_id": message_id,
            "feedback_type": feedback_type,
            "query": query,
            "response": response,
            "correction": correction,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._feedback.append(fb)
        if self.max_feedback is not None and len(self._feedback) > self.max_feedback:
            del self._feedback[: len(self._feedback) - self.max_feedback]
        return fb

    async def get_feedback(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self._purge_expired_feedback()
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
        await self.cleanup_expired_states()
        now = datetime.now(UTC)
        ttl_delta = timedelta(days=7)
        if self.ttl_seconds is not None:
            ttl_delta = min(ttl_delta, timedelta(seconds=self.ttl_seconds))
        suspended = {
            "id": approval_id,
            "thread_id": thread_id,
            "agent_name": agent_name,
            "node_name": node_name,
            "state_snapshot": state_json,
            "created_at": now.isoformat(),
            "expires_at": (now + ttl_delta).isoformat(),
        }
        self._suspended_states[approval_id] = suspended
        self._suspended_states.move_to_end(approval_id)
        _evict_ordered(self._suspended_states, self.max_suspended_states, keep=approval_id)
        return suspended

    async def get_suspended_state(self, approval_id: str) -> dict[str, Any] | None:
        """Retrieve suspended state, returning None if expired."""
        state = self._suspended_states.get(approval_id)
        if state is None:
            return None
        now = datetime.now(UTC)
        exp_dt = _parse_iso(state.get("expires_at"))
        if (exp_dt is not None and exp_dt < now) or self._is_expired(state.get("created_at"), now):
            # Auto-cleanup expired state
            self._suspended_states.pop(approval_id, None)
            return None
        self._suspended_states.move_to_end(approval_id)
        return state

    async def list_suspended_states(
        self,
        *,
        thread_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        await self.cleanup_expired_states()
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
        expired_ids: list[str] = []
        for sid, state in self._suspended_states.items():
            exp_dt = _parse_iso(state.get("expires_at"))
            if (exp_dt is not None and exp_dt < now) or self._is_expired(
                state.get("created_at"), now
            ):
                expired_ids.append(sid)
        for sid in expired_ids:
            self._suspended_states.pop(sid, None)
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
        parent_thread = await self.get_thread(parent_thread_id)
        if not parent_thread or message_index < 0:
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
        self._threads.move_to_end(new_thread_id)

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

        if (
            self.max_messages_per_thread is not None
            and len(forked_messages) > self.max_messages_per_thread
        ):
            forked_messages = forked_messages[-self.max_messages_per_thread :]

        self._messages[new_thread_id] = forked_messages
        self._msg_seq[new_thread_id] = forked_messages[-1]["id"] if forked_messages else 0
        forked_thread["message_count"] = len(forked_messages)
        await self._evict_overflow(keep=new_thread_id)
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
        self._purge_expired_dict(self._checkpoints, "created_at")
        if not checkpoint_id:
            thread_checkpoints = [
                cp
                for cp in self._checkpoints.values()
                if cp["thread_id"] == thread_id and cp["checkpoint_ns"] == checkpoint_ns
            ]
            if not thread_checkpoints:
                return None
            thread_checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
            latest = thread_checkpoints[0]
            latest_key = (
                latest["thread_id"],
                latest["checkpoint_ns"],
                latest["checkpoint_id"],
            )
            if latest_key in self._checkpoints:
                self._checkpoints.move_to_end(latest_key)
            return latest
        key = (thread_id, checkpoint_ns, checkpoint_id)
        cp = self._checkpoints.get(key)
        if cp is not None:
            self._checkpoints.move_to_end(key)
        return cp

    async def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        self._purge_expired_dict(self._checkpoints, "created_at")
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
        self._checkpoints.move_to_end(key)
        _evict_ordered(self._checkpoints, self.max_checkpoints, keep=key)

    async def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._purge_expired_dict(self._checkpoints, "created_at")
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
        # Always cap the result set (see SQLAlchemyStore.list_checkpoints for why).
        cps = cps[: limit if limit is not None else _DEFAULT_CHECKPOINT_LIST_LIMIT]
        return cps

    # ------------------------------------------------------------------
    # Invocation log history
    # ------------------------------------------------------------------

    async def create_invocation_log(
        self,
        *,
        agent_name: str,
        resource_type: str = "agent",
        thread_id: str | None = None,
        run_id: str | None = None,
        endpoint: str = "invoke",
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
        status: str = "ok",
        log_id: str | None = None,
    ) -> dict[str, Any]:
        import uuid

        self._purge_expired_dict(self._invocation_logs, "timestamp")
        lid = log_id or f"invlog_{uuid.uuid4().hex[:16]}"
        rtype = resource_type or "agent"
        entry = {
            "id": lid,
            "resource_type": rtype,
            "resource_name": agent_name,
            "agent_name": agent_name,
            "thread_id": thread_id,
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "input": input_data or {},
            "output": output_data or {},
            "metadata": metadata or {},
            "error": error,
            "duration_ms": duration_ms,
            "status": status,
        }
        self._invocation_logs[lid] = entry
        self._invocation_logs.move_to_end(lid)
        _evict_ordered(self._invocation_logs, self.max_invocation_logs, keep=lid)
        return entry

    async def get_invocation_log(self, log_id: str) -> dict[str, Any] | None:
        self._purge_expired_dict(self._invocation_logs, "timestamp")
        return self._invocation_logs.get(log_id)

    async def list_invocation_logs(
        self,
        *,
        agent_name: str | None = None,
        resource_type: str | None = None,
        thread_id: str | None = None,
        status: str | None = None,
        endpoint: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._purge_expired_dict(self._invocation_logs, "timestamp")
        items = list(self._invocation_logs.values())
        if agent_name:
            items = [i for i in items if i.get("agent_name") == agent_name]
        if resource_type:
            items = [i for i in items if (i.get("resource_type") or "agent") == resource_type]
        if thread_id:
            items = [i for i in items if i.get("thread_id") == thread_id]
        if status:
            items = [i for i in items if i.get("status") == status]
        if endpoint:
            items = [i for i in items if i.get("endpoint") == endpoint]
        items.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        return items[offset : offset + limit]

    async def count_invocation_logs(
        self,
        *,
        agent_name: str | None = None,
        resource_type: str | None = None,
        thread_id: str | None = None,
        status: str | None = None,
        endpoint: str | None = None,
    ) -> int:
        self._purge_expired_dict(self._invocation_logs, "timestamp")
        items = await self.list_invocation_logs(
            agent_name=agent_name,
            resource_type=resource_type,
            thread_id=thread_id,
            status=status,
            endpoint=endpoint,
            limit=10_000_000,
            offset=0,
        )
        return len(items)

    async def save_log_analysis(
        self,
        *,
        agent_name: str,
        resource_type: str = "agent",
        score: float | None,
        summary: str,
        status: str,
        recommendations: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        import uuid

        self._purge_expired_dict(self._log_analyses, "created_at")
        aid = analysis_id or f"logan_{uuid.uuid4().hex[:16]}"
        rtype = resource_type or "agent"
        entry = {
            "id": aid,
            "resource_type": rtype,
            "resource_name": agent_name,
            "agent_name": agent_name,
            "score": score,
            "summary": summary,
            "status": status,
            "recommendations": recommendations or [],
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._log_analyses[aid] = entry
        self._log_analyses.move_to_end(aid)
        _evict_ordered(self._log_analyses, self.max_log_analyses, keep=aid)
        return entry

    async def get_latest_log_analysis(
        self,
        agent_name: str,
        *,
        resource_type: str = "agent",
    ) -> dict[str, Any] | None:
        self._purge_expired_dict(self._log_analyses, "created_at")
        items = [
            a
            for a in self._log_analyses.values()
            if a.get("agent_name") == agent_name
            and (a.get("resource_type") or "agent") == (resource_type or "agent")
        ]
        if not items:
            return None
        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return items[0]

    async def list_log_analyses(
        self,
        *,
        agent_name: str | None = None,
        resource_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._purge_expired_dict(self._log_analyses, "created_at")
        items = list(self._log_analyses.values())
        if agent_name:
            items = [a for a in items if a.get("agent_name") == agent_name]
        if resource_type:
            items = [a for a in items if (a.get("resource_type") or "agent") == resource_type]
        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return items[offset : offset + limit]

    # ------------------------------------------------------------------
    # Optimization / retrain run history
    # ------------------------------------------------------------------

    async def create_optimization_run(
        self,
        *,
        experiment_id: str,
        agent_name: str,
        baseline_score: float | None = None,
        best_score: float | None = None,
        prompt_versions: dict[str, Any] | None = None,
        score_history: list[Any] | None = None,
        learnings: list[Any] | None = None,
        artefacts: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        import uuid

        self._purge_expired_dict(self._optimization_runs, "created_at")
        rid = run_id or f"optrun_{uuid.uuid4().hex[:16]}"
        entry = {
            "id": rid,
            "experiment_id": experiment_id,
            "agent_name": agent_name,
            "baseline_score": baseline_score,
            "best_score": best_score,
            "prompt_versions": prompt_versions or {},
            "score_history": score_history or [],
            "learnings": learnings or [],
            "artefacts": artefacts or {},
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._optimization_runs[rid] = entry
        self._optimization_runs.move_to_end(rid)
        _evict_ordered(self._optimization_runs, self.max_optimization_runs, keep=rid)
        return entry

    async def get_optimization_run(self, run_id: str) -> dict[str, Any] | None:
        self._purge_expired_dict(self._optimization_runs, "created_at")
        return self._optimization_runs.get(run_id)

    async def list_optimization_runs(
        self,
        *,
        agent_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._purge_expired_dict(self._optimization_runs, "created_at")
        items = list(self._optimization_runs.values())
        if agent_name:
            items = [r for r in items if r.get("agent_name") == agent_name]
        if experiment_id:
            items = [r for r in items if r.get("experiment_id") == experiment_id]
        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return items[offset : offset + limit]

    def clear(self) -> None:
        """Clear all in-memory collections."""
        self._threads.clear()
        self._messages.clear()
        self._msg_seq.clear()
        self._feedback.clear()
        self._feedback_seq = 0
        self._suspended_states.clear()
        self._checkpoints.clear()
        self._invocation_logs.clear()
        self._log_analyses.clear()
        self._optimization_runs.clear()

    async def close(self) -> None:
        """Release all in-memory collections."""
        self.clear()
