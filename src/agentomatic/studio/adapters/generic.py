"""Generic trace-based adapter for Agentomatic Studio.

Provides a best-effort Studio experience for agents that don't use
LangGraph. Generates synthetic graph topologies, captures execution
traces with timing data, and maintains an in-memory state/history
store.

This adapter ensures that *every* agent gets useful Studio information
even if the underlying framework doesn't expose graph APIs.
"""

from __future__ import annotations

import time
import traceback
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
    - **In-memory state** — Stores the last input/output for each
      thread so the State tab always shows useful information.
    - **Execution history** — Maintains a per-thread history of all
      executions for the History tab.

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
        # In-memory trace store: thread_id → list of state snapshots
        self._state_store: dict[str, dict[str, Any]] = {}
        self._history_store: dict[str, list[StudioCheckpoint]] = defaultdict(list)
        self._execution_counter: dict[str, int] = defaultdict(int)
        # User-provided graph/state/stream overrides via decorators
        self._custom_graph_fn = getattr(agent, "_studio_graph_fn", None)
        self._custom_state_fn = getattr(agent, "_studio_state_fn", None)
        self._custom_stream_fn = getattr(agent, "_studio_stream_fn", None)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> list[str]:
        caps = ["streaming", "traces"]
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

        # Store state for later inspection
        self._state_store[thread_id] = {
            "last_input": state,
            "last_output": output,
            "updated_at": _now_iso(),
        }

        # Record in history
        self._execution_counter[thread_id] += 1
        step = self._execution_counter[thread_id]
        self._history_store[thread_id].append(
            StudioCheckpoint(
                id=f"trace_{thread_id}_{step}",
                thread_id=thread_id,
                step=step,
                state={"input": state, "output": output},
                metadata={
                    "duration_ms": duration,
                    "framework": self._agent.manifest.framework,
                },
                parent_id=(f"trace_{thread_id}_{step - 1}" if step > 1 else None),
                timestamp=_now_iso(),
            )
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
                import asyncio

                if asyncio.iscoroutinefunction(self._custom_state_fn):
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

        # Use in-memory trace store
        stored = self._state_store.get(thread_id, {})
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
        return list(reversed(self._history_store.get(thread_id, [])))

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
