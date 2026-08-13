"""Rollout / span / reward models and a lightweight trace store.

Inspired by Agent Lightning's unified data interface (rollouts + spans +
rewards) without adopting LightningStore, client/server, or VERL.

Usage::

    from agentomatic.optimize.rollout import (
        Rollout,
        RewardSignal,
        RolloutTraceStore,
        rollout_from_run_result,
    )

    store = RolloutTraceStore()
    rollout = rollout_from_run_result(run_result, resource_id="v0", reward=0.8)
    store.add(rollout)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agentomatic.optimize.runner import RunResult

RolloutMode = Literal["train", "val", "test"]
RolloutStatus = Literal["running", "succeeded", "failed"]


@dataclass(slots=True)
class RewardSignal:
    """Scalar reward with optional dimensional breakdown.

    Maps Agentomatic metric/judge scores onto a Lightning-style reward.
    """

    value: float
    dimensions: dict[str, float] = field(default_factory=dict)
    source: str = "metric"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "value": self.value,
            "dimensions": dict(self.dimensions),
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardSignal:
        """Deserialise from a plain dict."""
        return cls(
            value=float(data.get("value", 0.0)),
            dimensions=dict(data.get("dimensions") or {}),
            source=str(data.get("source") or "metric"),
            reason=str(data.get("reason") or ""),
        )


@dataclass(slots=True)
class RolloutSpan:
    """One structured span captured during an agent rollout."""

    name: str
    kind: str = "step"
    attributes: dict[str, Any] = field(default_factory=dict)
    start_ms: float | None = None
    end_ms: float | None = None
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutSpan:
        """Deserialise from a plain dict."""
        return cls(
            name=str(data.get("name") or "span"),
            kind=str(data.get("kind") or "step"),
            attributes=dict(data.get("attributes") or {}),
            start_ms=data.get("start_ms"),
            end_ms=data.get("end_ms"),
            parent=data.get("parent"),
        )


@dataclass(slots=True)
class Rollout:
    """One evaluated agent execution with traces and reward."""

    rollout_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str = ""
    response: str = ""
    expected: str | None = None
    mode: RolloutMode = "val"
    status: RolloutStatus = "succeeded"
    resource_id: str | None = None
    reward: RewardSignal | None = None
    spans: list[RolloutSpan] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps_taken: list[str] = field(default_factory=list)
    reasoning: str = ""
    retrieval_context: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "rollout_id": self.rollout_id,
            "query": self.query,
            "response": self.response,
            "expected": self.expected,
            "mode": self.mode,
            "status": self.status,
            "resource_id": self.resource_id,
            "reward": self.reward.to_dict() if self.reward else None,
            "spans": [s.to_dict() for s in self.spans],
            "messages": list(self.messages),
            "tool_calls": list(self.tool_calls),
            "steps_taken": list(self.steps_taken),
            "reasoning": self.reasoning,
            "retrieval_context": list(self.retrieval_context),
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rollout:
        """Deserialise from a plain dict."""
        reward_data = data.get("reward")
        return cls(
            rollout_id=str(data.get("rollout_id") or uuid.uuid4().hex[:12]),
            query=str(data.get("query") or ""),
            response=str(data.get("response") or ""),
            expected=data.get("expected"),
            mode=data.get("mode") or "val",
            status=data.get("status") or "succeeded",
            resource_id=data.get("resource_id"),
            reward=RewardSignal.from_dict(reward_data) if reward_data else None,
            spans=[RolloutSpan.from_dict(s) for s in (data.get("spans") or [])],
            messages=list(data.get("messages") or []),
            tool_calls=list(data.get("tool_calls") or []),
            steps_taken=list(data.get("steps_taken") or []),
            reasoning=str(data.get("reasoning") or ""),
            retrieval_context=list(data.get("retrieval_context") or []),
            duration_ms=float(data.get("duration_ms") or 0.0),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
        )


def spans_from_run_result(rr: RunResult) -> list[RolloutSpan]:
    """Build lightweight spans from an :class:`RunResult`."""
    spans: list[RolloutSpan] = []
    for idx, step in enumerate(rr.steps_taken or []):
        spans.append(
            RolloutSpan(
                name=str(step),
                kind="step",
                attributes={"index": idx},
            )
        )
    for idx, tool in enumerate(rr.tool_calls or []):
        name = str(tool.get("name") or tool.get("tool") or f"tool_{idx}")
        spans.append(
            RolloutSpan(
                name=name,
                kind="tool",
                attributes=dict(tool),
            )
        )
    if rr.reasoning:
        spans.append(
            RolloutSpan(
                name="reasoning",
                kind="reasoning",
                attributes={"text": rr.reasoning[:2000]},
            )
        )
    if rr.retrieval_context:
        spans.append(
            RolloutSpan(
                name="retrieval",
                kind="retrieval",
                attributes={"n_docs": len(rr.retrieval_context)},
            )
        )
    return spans


def rollout_from_run_result(
    rr: RunResult,
    *,
    resource_id: str | None = None,
    reward: float | RewardSignal | None = None,
    mode: RolloutMode = "val",
    messages: list[dict[str, Any]] | None = None,
) -> Rollout:
    """Convert a :class:`RunResult` into a :class:`Rollout`."""
    reward_signal: RewardSignal | None
    if reward is None:
        reward_signal = None
    elif isinstance(reward, RewardSignal):
        reward_signal = reward
    else:
        reward_signal = RewardSignal(value=float(reward))

    status: RolloutStatus = "failed" if rr.error else "succeeded"
    return Rollout(
        query=rr.query,
        response=rr.response,
        expected=rr.expected,
        mode=mode,
        status=status,
        resource_id=resource_id,
        reward=reward_signal,
        spans=spans_from_run_result(rr),
        messages=list(messages or []),
        tool_calls=list(rr.tool_calls or []),
        steps_taken=list(rr.steps_taken or []),
        reasoning=rr.reasoning or "",
        retrieval_context=list(rr.retrieval_context or []),
        duration_ms=rr.duration_ms,
        error=rr.error,
        metadata=dict(rr.metadata or {}),
    )


class RolloutTraceStore:
    """In-memory rollout store with optional SQLite persistence.

    This is intentionally lighter than Agent Lightning's LightningStore:
    it only tracks evaluated rollouts for critique / APO / reporting.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._rollouts: list[Rollout] = []
        self._lock = threading.RLock()
        self._path = Path(path) if path else None
        self._conn: sqlite3.Connection | None = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rollouts (
                    rollout_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()
            self._load_sqlite()

    def _load_sqlite(self) -> None:
        if self._conn is None:
            return
        rows = self._conn.execute(
            "SELECT payload FROM rollouts ORDER BY created_at ASC"
        ).fetchall()
        for (payload,) in rows:
            try:
                self._rollouts.append(Rollout.from_dict(json.loads(payload)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

    def add(self, rollout: Rollout) -> Rollout:
        """Persist a rollout and return it."""
        with self._lock:
            self._rollouts.append(rollout)
            if self._conn is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO rollouts(rollout_id, payload, created_at) "
                    "VALUES (?, ?, ?)",
                    (
                        rollout.rollout_id,
                        json.dumps(rollout.to_dict(), ensure_ascii=False),
                        rollout.created_at,
                    ),
                )
                self._conn.commit()
        return rollout

    def extend(self, rollouts: list[Rollout]) -> None:
        """Add many rollouts."""
        for rollout in rollouts:
            self.add(rollout)

    def list(
        self,
        *,
        resource_id: str | None = None,
        mode: RolloutMode | None = None,
        limit: int | None = None,
    ) -> list[Rollout]:
        """Return rollouts filtered by resource / mode."""
        with self._lock:
            items = list(self._rollouts)
        if resource_id is not None:
            items = [r for r in items if r.resource_id == resource_id]
        if mode is not None:
            items = [r for r in items if r.mode == mode]
        if limit is not None:
            items = items[-limit:]
        return items

    def clear(self) -> None:
        """Drop all rollouts (memory + SQLite table)."""
        with self._lock:
            self._rollouts.clear()
            if self._conn is not None:
                self._conn.execute("DELETE FROM rollouts")
                self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return len(self._rollouts)
