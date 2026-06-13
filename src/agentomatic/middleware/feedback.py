"""Feedback collection middleware and decorator.

Auto-adds feedback endpoints to every agent and provides
a decorator for automatic input/output recording.

Usage::

    # In agent code:
    from agentomatic.middleware.feedback import collect_feedback

    @collect_feedback(store=True)
    async def invoke(state):
        ...

    # Via platform:
    platform = AgentPlatform.from_folder(
        "agents/",
        enable_feedback=True,
        store=MemoryStore(),
    )
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any, TypeVar, cast

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class FeedbackRecord:
    """A feedback entry."""

    feedback_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_name: str = ""
    user_id: str = ""
    thread_id: str | None = None
    message_id: int | None = None
    query: str = ""
    response: str = ""
    rating: int | None = None  # 1 (thumbs down) or 5 (thumbs up)
    comment: str | None = None  # Free-text comment
    correction: str | None = None  # User-provided correct answer
    feedback_type: str = "thumbs"  # thumbs, rating, correction, comment
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != "" and v != {}}


class FeedbackCollector:
    """Async feedback collector with background storage.

    Can be used standalone or auto-attached to agents.
    """

    def __init__(self, store: Any = None, buffer_size: int = 100):
        self._store = store
        self._buffer: list[FeedbackRecord] = []
        self._buffer_size = buffer_size
        self._lock = asyncio.Lock()

    async def record(
        self,
        agent_name: str,
        user_id: str = "",
        *,
        query: str = "",
        response: str = "",
        rating: int | None = None,
        comment: str | None = None,
        correction: str | None = None,
        feedback_type: str = "thumbs",
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        """Record a feedback entry."""
        from datetime import datetime

        record = FeedbackRecord(
            agent_name=agent_name,
            user_id=user_id,
            query=query,
            response=response,
            rating=rating,
            comment=comment,
            correction=correction,
            feedback_type=feedback_type,
            thread_id=thread_id,
            timestamp=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
        )

        # Store via backend
        if self._store and hasattr(self._store, "add_feedback"):
            try:
                await self._store.add_feedback(
                    thread_id=thread_id or "",
                    user_id=user_id,
                    agent_name=agent_name,
                    rating=rating,
                    comment=comment,
                    feedback_type=feedback_type,
                )
            except Exception as exc:
                logger.warning(f"Failed to store feedback: {exc}")

        # Buffer for batch export
        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer[-self._buffer_size :]

        logger.debug(f"📝 Feedback recorded for {agent_name} (rating={rating})")
        return record

    async def get_feedback(
        self,
        agent_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get stored feedback."""
        if self._store and hasattr(self._store, "get_feedback"):
            return cast(
                list[dict[str, Any]],
                await self._store.get_feedback(
                    agent_name=agent_name,
                    limit=limit,
                ),
            )
        # Fall back to buffer
        async with self._lock:
            items = self._buffer
            if agent_name:
                items = [f for f in items if f.agent_name == agent_name]
            return [f.to_dict() for f in items[-limit:]]

    async def export_jsonl(self, agent_name: str | None = None) -> str:
        """Export feedback as JSONL string (for optimization datasets)."""
        import json

        records = await self.get_feedback(agent_name=agent_name, limit=10000)
        lines = []
        for r in records:
            # Convert to optimization-friendly format
            entry = {
                "query": r.get("query", ""),
                "expected_answer": r.get("correction") or r.get("response", ""),
                "metadata": {
                    "rating": r.get("rating"),
                    "comment": r.get("comment"),
                    "feedback_type": r.get("feedback_type"),
                },
            }
            if entry["query"]:  # Only include entries with queries
                lines.append(json.dumps(entry))
        return "\n".join(lines)


def collect_feedback(
    store: bool = True,
    log: bool = True,
) -> Callable[[F], F]:
    """Decorator to auto-record agent inputs/outputs for feedback.

    Records every invocation's query and response.
    Useful for building optimization datasets from production traffic.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(state: dict, *args, **kwargs):
            result = await fn(state, *args, **kwargs)
            if log:
                query = state.get("current_query", "")
                response = result.get("response", "") if isinstance(result, dict) else str(result)
                logger.info(f"📊 [feedback] Q={query[:50]}... A={response[:50]}...")
            return result

        return wrapper  # type: ignore

    return decorator


# Module-level singleton
_collector: FeedbackCollector | None = None


def get_collector() -> FeedbackCollector:
    global _collector
    if _collector is None:
        _collector = FeedbackCollector()
    return _collector


def set_collector(collector: FeedbackCollector) -> None:
    global _collector
    _collector = collector
