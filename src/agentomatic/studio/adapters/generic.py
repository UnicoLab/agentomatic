"""Generic trace-based adapter for Agentomatic Studio.

Provides a best-effort Studio experience for agents that don't use
LangGraph. Generates synthetic graph topologies, captures execution
traces with timing data, and uses the configured platform store to retain
those traces across API-worker and platform restarts.

This adapter ensures that *every* agent gets useful Studio information
even if the underlying framework doesn't expose graph APIs.
"""

from __future__ import annotations

import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.studio.adapter import StudioAdapter
from agentomatic.studio.models import (
    StudioCheckpoint,
    StudioGraphEdge,
    StudioGraphNode,
    StudioGraphTopology,
    StudioRunEvent,
    StudioStateSnapshot,
)

if TYPE_CHECKING:
    from agentomatic.core.manifest import RegisteredAgent
    from agentomatic.storage.base import BaseStore


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class GenericAdapter(StudioAdapter):
    """Trace-based adapter for non-LangGraph agents.

    Provides the "lite" Studio experience with:

    - **Synthetic graph** — A linear ``__start__ → agent → __end__``
      topology that updates in real-time during execution.
    - **Trace-based SSE** — Captures execution timing, input/output
      payloads, and exceptions as ``StudioRunEvent`` objects.
    - **Captured state** — Stores the last input/output for each thread so
      the State tab always shows useful information. Captured run state is
      durable when Agentomatic has a configured store.
    - **Execution history** — Maintains a per-thread history of all executions
      for the History tab, persisting them when a store is available.

    Args:
        agent: The registered agent with a ``node_fn``.
        store: Optional storage backend (used if available).
    """

    def __init__(
        self,
        agent: RegisteredAgent,
        store: BaseStore | None = None,
    ) -> None:
        super().__init__(agent.name)
        self._agent = agent
        self._store = store
        # Local caches make just-completed traces available without a storage
        # round trip. The configured store is the durable source of truth.
        self._state_store: dict[str, dict[str, Any]] = {}
        self._history_store: dict[str, list[StudioCheckpoint]] = defaultdict(list)
        self._execution_counter: dict[str, int] = defaultdict(int)
        # User-provided graph/state/stream overrides via decorators
        self._custom_graph_fn = getattr(agent, "_studio_graph_fn", None)
        self._custom_state_fn = getattr(agent, "_studio_state_fn", None)
        self._custom_stream_fn = getattr(agent, "_studio_stream_fn", None)

    @property
    def _checkpoint_namespace(self) -> str:
        """Keep generic traces separate from framework-native checkpoints."""
        return f"studio:generic:{self.agent_name}"

    @staticmethod
    def _as_str(value: Any, fallback: str) -> str:
        return value if isinstance(value, str) and value else fallback

    def _checkpoint_from_record(self, record: dict[str, Any]) -> StudioCheckpoint | None:
        """Translate a durable storage record into Studio's public shape."""
        checkpoint = record.get("checkpoint")
        checkpoint_id = record.get("checkpoint_id")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint_id, str):
            return None
        metadata = record.get("metadata")
        parent_id = record.get("parent_checkpoint_id")
        return StudioCheckpoint(
            id=checkpoint_id,
            thread_id=self._as_str(record.get("thread_id"), "default"),
            step=metadata.get("step", 0) if isinstance(metadata, dict) else 0,
            state=checkpoint,
            metadata=metadata if isinstance(metadata, dict) else {},
            parent_id=parent_id if isinstance(parent_id, str) else None,
            timestamp=self._as_str(record.get("created_at"), _now_iso()),
        )

    async def _stored_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> StudioCheckpoint | None:
        """Look up a persisted trace without making storage an execution dependency."""
        if not self._store:
            return None
        try:
            record = await self._store.get_checkpoint(
                thread_id,
                self._checkpoint_namespace,
                checkpoint_id,
            )
            return self._checkpoint_from_record(record) if isinstance(record, dict) else None
        except Exception as exc:  # noqa: BLE001 - debugging remains useful during DB outages.
            logger.warning(
                "Studio could not read generic checkpoint '{}' for '{}': {}",
                checkpoint_id,
                self.agent_name,
                exc,
            )
            return None

    async def _latest_stored_checkpoint(self, thread_id: str) -> StudioCheckpoint | None:
        """Return the latest durable generic trace, if storage is configured."""
        return await self._stored_checkpoint(thread_id, "")

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> list[str]:
        # Generic traces can be durable when a platform store is configured;
        # manual state mutation remains a best-effort adapter capability.
        caps = ["streaming", "traces", "state", "checkpoints"]
        if self._custom_graph_fn is not None:
            caps.append("graph")
        if self._custom_state_fn is not None:
            caps.append("state")
        return caps

    # ------------------------------------------------------------------
    # Graph topology
    # ------------------------------------------------------------------

    async def get_graph(self) -> StudioGraphTopology:
        # If user provided a custom graph via @studio_graph, use it
        if self._custom_graph_fn is not None:
            try:
                result = self._custom_graph_fn()
                return self._parse_user_graph(result)
            except Exception as exc:
                logger.warning(f"Custom graph function failed: {exc}")

        # Default: synthetic linear graph
        return StudioGraphTopology(
            agent_name=self.agent_name,
            nodes=[
                StudioGraphNode(id="__start__", name="Start", type="start"),
                StudioGraphNode(
                    id=self.agent_name,
                    name=self._agent.manifest.name,
                    type="agent",
                    metadata={
                        "framework": self._agent.manifest.framework,
                        "description": self._agent.manifest.description,
                    },
                ),
                StudioGraphNode(id="__end__", name="End", type="end"),
            ],
            edges=[
                StudioGraphEdge(id="edge_0", source="__start__", target=self.agent_name),
                StudioGraphEdge(id="edge_1", source=self.agent_name, target="__end__"),
            ],
            entry_point="__start__",
            end_points=["__end__"],
            metadata={"mode": "synthetic", "framework": self._agent.manifest.framework},
        )

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    async def stream_execution(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
        breakpoints: list[str] | None = None,
        checkpoint_id: str | None = None,
    ) -> AsyncGenerator[StudioRunEvent, None]:
        # If user provided a custom stream function, delegate to it
        if self._custom_stream_fn is not None:
            try:
                async for event in self._custom_stream_fn(state, config, breakpoints):
                    yield event
                return
            except Exception as exc:
                logger.warning(f"Custom stream function failed, falling back: {exc}")

        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

        # Generic agents do not have a framework-native continuation API, but
        # their trace history records the exact input that produced each
        # checkpoint. Replaying a checkpoint must therefore invoke that stored
        # input, rather than silently running the empty request supplied by the
        # Studio replay button. This is a true re-execution (not a claim that
        # an arbitrary node can resume midway through a graph).
        if checkpoint_id:
            checkpoint = next(
                (
                    item
                    for item in self._history_store.get(thread_id, [])
                    if item.id == checkpoint_id
                ),
                None,
            )
            if checkpoint is None:
                checkpoint = await self._stored_checkpoint(thread_id, checkpoint_id)
            checkpoint_input = (
                checkpoint.state.get("input")
                if checkpoint is not None and isinstance(checkpoint.state, dict)
                else None
            )
            if not isinstance(checkpoint_input, dict):
                raise ValueError(
                    f"Checkpoint '{checkpoint_id}' is unavailable for thread '{thread_id}'"
                )
            state = dict(checkpoint_input)

        # Emit node_start for the agent
        yield StudioRunEvent(
            event="node_start",
            run_id="",
            timestamp=_now_iso(),
            node=self.agent_name,
            data={
                "input": state,
                "framework": self._agent.manifest.framework,
            },
        )

        start_time = time.monotonic()
        result: Any = None
        error_info: str | None = None

        try:
            if self._agent.node_fn:
                result = await self._agent.node_fn(state)
            else:
                error_info = f"Agent '{self.agent_name}' has no callable (node_fn or graph_fn)"
        except Exception as exc:
            error_info = str(exc)
            # Emit detailed trace event with stack trace
            yield StudioRunEvent(
                event="trace",
                run_id="",
                timestamp=_now_iso(),
                node=self.agent_name,
                data={
                    "level": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "exception_type": type(exc).__name__,
                },
            )

        duration = round((time.monotonic() - start_time) * 1000, 2)

        if error_info:
            yield StudioRunEvent(
                event="node_end",
                run_id="",
                timestamp=_now_iso(),
                node=self.agent_name,
                data={"error": error_info},
                duration_ms=duration,
            )
            return

        # Normalize result
        output = result if isinstance(result, dict) else {"response": str(result)}

        recorded_at = _now_iso()

        # Store state for immediate inspection.
        self._state_store[thread_id] = {
            "last_input": state,
            "last_output": output,
            "updated_at": recorded_at,
        }

        # Record each finished execution as a UUID-addressed checkpoint. A
        # counter alone collides after a process restart, so the parent is
        # resolved from either the local cache or durable store.
        previous = self._history_store.get(thread_id, [])
        parent = previous[-1] if previous else await self._latest_stored_checkpoint(thread_id)
        step = (parent.step if parent is not None else 0) + 1
        trace = StudioCheckpoint(
            id=f"trace_{uuid.uuid4().hex}",
            thread_id=thread_id,
            step=step,
            state={"input": state, "output": output},
            metadata={
                "duration_ms": duration,
                "framework": self._agent.manifest.framework,
                "step": step,
            },
            parent_id=parent.id if parent is not None else None,
            timestamp=recorded_at,
        )
        self._execution_counter[thread_id] = step
        self._history_store[thread_id].append(trace)
        if self._store:
            try:
                await self._store.save_checkpoint(
                    thread_id,
                    self._checkpoint_namespace,
                    trace.id,
                    trace.parent_id,
                    trace.state,
                    trace.metadata,
                )
            except Exception as exc:  # noqa: BLE001 - do not make tracing break an agent run.
                logger.warning(
                    "Studio could not persist generic trace '{}' for '{}': {}",
                    trace.id,
                    self.agent_name,
                    exc,
                )

        # Emit a trace event with full execution details
        yield StudioRunEvent(
            event="trace",
            run_id="",
            timestamp=_now_iso(),
            node=self.agent_name,
            data={
                "level": "info",
                "message": f"Execution completed in {duration}ms",
                "input_keys": list(state.keys()),
                "output_keys": list(output.keys()),
                "duration_ms": duration,
            },
        )

        # Emit node_end
        yield StudioRunEvent(
            event="node_end",
            run_id="",
            timestamp=_now_iso(),
            node=self.agent_name,
            data={"output": output},
            duration_ms=duration,
        )

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    async def get_state(self, thread_id: str) -> StudioStateSnapshot | None:
        # If user provided a custom state function, use it
        if self._custom_state_fn is not None:
            try:
                import inspect

                if inspect.iscoroutinefunction(self._custom_state_fn):
                    state_data = await self._custom_state_fn(thread_id)
                else:
                    state_data = self._custom_state_fn(thread_id)
                return StudioStateSnapshot(
                    thread_id=thread_id,
                    agent_name=self.agent_name,
                    state=state_data or {},
                    timestamp=_now_iso(),
                )
            except Exception as exc:
                logger.warning(f"Custom state function failed: {exc}")

        # Use the local snapshot first, then rebuild it from the latest durable
        # trace after a worker or platform restart.
        stored = self._state_store.get(thread_id, {})
        if not stored:
            checkpoint = await self._latest_stored_checkpoint(thread_id)
            if checkpoint is not None and isinstance(checkpoint.state, dict):
                stored = {
                    "last_input": checkpoint.state.get("input"),
                    "last_output": checkpoint.state.get("output"),
                    "updated_at": checkpoint.timestamp,
                }
                self._state_store[thread_id] = stored
        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=self.agent_name,
            state=stored,
            timestamp=_now_iso(),
        )

    async def update_state(
        self,
        thread_id: str,
        updates: dict[str, Any],
    ) -> StudioStateSnapshot | None:
        # Generic adapter has limited state mutation support
        # but we can update the in-memory store
        current = self._state_store.get(thread_id, {})
        merged = {**current, **updates}
        self._state_store[thread_id] = merged
        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=self.agent_name,
            state=merged,
            timestamp=_now_iso(),
        )

    # ------------------------------------------------------------------
    # Checkpoint history
    # ------------------------------------------------------------------

    async def get_history(self, thread_id: str) -> list[StudioCheckpoint]:
        local = {item.id: item for item in self._history_store.get(thread_id, [])}
        if self._store:
            try:
                records = await self._store.list_checkpoints(
                    thread_id,
                    self._checkpoint_namespace,
                )
                for record in records:
                    checkpoint = self._checkpoint_from_record(record)
                    if checkpoint is not None:
                        local[checkpoint.id] = checkpoint
            except Exception as exc:  # noqa: BLE001 - history remains useful from the local cache.
                logger.warning(
                    "Studio could not list generic traces for '{}': {}",
                    self.agent_name,
                    exc,
                )
        return sorted(local.values(), key=lambda item: item.timestamp, reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_user_graph(self, data: dict[str, Any]) -> StudioGraphTopology:
        """Parse a user-provided graph dict into a StudioGraphTopology."""
        nodes = []
        for n in data.get("nodes", []):
            nodes.append(
                StudioGraphNode(
                    id=n.get("id", "unknown"),
                    name=n.get("name", n.get("id", "unknown")),
                    type=n.get("type", "agent"),
                    metadata=n.get("metadata", {}),
                )
            )
        edges = []
        for idx, e in enumerate(data.get("edges", [])):
            edges.append(
                StudioGraphEdge(
                    id=e.get("id", f"edge_{idx}"),
                    source=e.get("source", ""),
                    target=e.get("target", ""),
                    condition=e.get("condition"),
                )
            )

        entry_point = None
        end_points = []
        for n in nodes:
            if n.type == "start" or n.id == "__start__":
                entry_point = n.id
            elif n.type == "end" or n.id == "__end__":
                end_points.append(n.id)

        return StudioGraphTopology(
            agent_name=self.agent_name,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            end_points=end_points,
            metadata={"mode": "custom"},
        )
