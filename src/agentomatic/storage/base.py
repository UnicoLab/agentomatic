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
        self,
        thread_id: str,
        **updates: Any,
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
        """Save suspended execution state for human approval."""
        return {}

    async def get_suspended_state(self, approval_id: str) -> dict[str, Any] | None:
        """Retrieve suspended state by approval ID."""
        return None

    async def list_suspended_states(
        self,
        *,
        thread_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List suspended states, optionally filtered by thread or agent."""
        return []

    async def delete_suspended_state(self, approval_id: str) -> bool:
        """Delete suspended state by approval ID.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        return False

    async def cleanup_expired_states(self) -> int:
        """Delete expired suspended states.

        Returns:
            Number of states deleted.
        """
        return 0

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
        """Fork a thread at a specific message index (0-indexed relative to parent's chronological messages).

        Copies thread information and all messages up to and including the message_index.
        Returns the new thread dict, or None if the parent thread is not found.
        """
        return None

    async def get_thread_lineage(self, thread_id: str) -> dict[str, Any]:
        """Get the full lineage tree for a thread (ancestors and descendants).

        Returns:
            Dict with ``ancestors`` (list from root to parent) and ``descendants``
            (list of direct child threads).
        """
        return {"thread_id": thread_id, "ancestors": [], "descendants": []}

    # ------------------------------------------------------------------
    # Checkpointer operations
    # ------------------------------------------------------------------

    async def get_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve a LangGraph checkpoint by thread, namespace, and checkpoint ID."""
        return None

    async def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Save a LangGraph execution checkpoint."""
        pass

    async def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List checkpoints for a thread/namespace chronologically."""
        return []

    # ------------------------------------------------------------------
    # Invocation log history
    # ------------------------------------------------------------------

    async def create_invocation_log(
        self,
        *,
        agent_name: str,
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
        """Persist a per-agent invocation log entry.

        Default implementation is a no-op stub. Override for persistence.
        """
        return {
            "id": log_id or "",
            "agent_name": agent_name,
            "thread_id": thread_id,
            "run_id": run_id,
            "endpoint": endpoint,
            "input": input_data or {},
            "output": output_data or {},
            "metadata": metadata or {},
            "error": error,
            "duration_ms": duration_ms,
            "status": status,
            "timestamp": None,
        }

    async def get_invocation_log(self, log_id: str) -> dict[str, Any] | None:
        """Retrieve a single invocation log by ID."""
        return None

    async def list_invocation_logs(
        self,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
        status: str | None = None,
        endpoint: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List invocation logs with optional filters (newest first)."""
        return []

    async def count_invocation_logs(
        self,
        *,
        agent_name: str | None = None,
        thread_id: str | None = None,
        status: str | None = None,
        endpoint: str | None = None,
    ) -> int:
        """Count invocation logs matching optional filters."""
        return 0

    async def save_log_analysis(
        self,
        *,
        agent_name: str,
        score: float | None,
        summary: str,
        status: str,
        recommendations: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist an LLM log-analysis result."""
        return {
            "id": analysis_id or "",
            "agent_name": agent_name,
            "score": score,
            "summary": summary,
            "status": status,
            "recommendations": recommendations or [],
            "metadata": metadata or {},
            "created_at": None,
        }

    async def get_latest_log_analysis(self, agent_name: str) -> dict[str, Any] | None:
        """Return the most recent log analysis for an agent."""
        return None

    async def list_log_analyses(
        self,
        *,
        agent_name: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List stored log analyses (newest first)."""
        return []

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
        """Persist an auditable optimization/retrain run."""
        return {
            "id": run_id or "",
            "experiment_id": experiment_id,
            "agent_name": agent_name,
            "baseline_score": baseline_score,
            "best_score": best_score,
            "prompt_versions": prompt_versions or {},
            "score_history": score_history or [],
            "learnings": learnings or [],
            "artefacts": artefacts or {},
            "metadata": metadata or {},
            "created_at": None,
        }

    async def get_optimization_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a single optimization run by ID."""
        return None

    async def list_optimization_runs(
        self,
        *,
        agent_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List optimization runs (newest first)."""
        return []
