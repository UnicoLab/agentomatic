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
        self._invocation_logs: dict[str, dict[str, Any]] = {}
        self._log_analyses: dict[str, dict[str, Any]] = {}
        self._optimization_runs: dict[str, dict[str, Any]] = {}

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
        return entry

    async def get_invocation_log(self, log_id: str) -> dict[str, Any] | None:
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
        items = list(self._invocation_logs.values())
        if agent_name:
            items = [i for i in items if i.get("agent_name") == agent_name]
        if resource_type:
            items = [
                i for i in items if (i.get("resource_type") or "agent") == resource_type
            ]
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
        return entry

    async def get_latest_log_analysis(
        self,
        agent_name: str,
        *,
        resource_type: str = "agent",
    ) -> dict[str, Any] | None:
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
        items = list(self._log_analyses.values())
        if agent_name:
            items = [a for a in items if a.get("agent_name") == agent_name]
        if resource_type:
            items = [
                a for a in items if (a.get("resource_type") or "agent") == resource_type
            ]
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
        return entry

    async def get_optimization_run(self, run_id: str) -> dict[str, Any] | None:
        return self._optimization_runs.get(run_id)

    async def list_optimization_runs(
        self,
        *,
        agent_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        items = list(self._optimization_runs.values())
        if agent_name:
            items = [r for r in items if r.get("agent_name") == agent_name]
        if experiment_id:
            items = [r for r in items if r.get("experiment_id") == experiment_id]
        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return items[offset : offset + limit]
