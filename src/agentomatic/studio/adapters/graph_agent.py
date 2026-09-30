"""Adapter for class-based AgentGraph agents."""

from __future__ import annotations

import uuid
from collections import OrderedDict
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
    from agentomatic.agents.graph import AgentGraph
    from agentomatic.core.manifest import RegisteredAgent
    from agentomatic.storage.base import BaseStore


class GraphAgentAdapter(StudioAdapter):
    """Studio adapter for the native AgentGraph runtime.

    Provides the full Studio experience for class-based agents:
    - Extracts true multi-node topology from AgentGraph
    - Emits fine-grained node_start/node_end SSE events
    """

    #: Local cache bounds (the configured store, when any, is durable).
    _MAX_THREADS = 500
    _MAX_CHECKPOINTS_PER_THREAD = 200

    def __init__(
        self,
        agent: RegisteredAgent,
        store: BaseStore | None = None,
    ) -> None:
        super().__init__(agent.name)
        self._agent = agent
        self._store = store
        # thread_id → latest state / per-node checkpoints, most recent last.
        self._state_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._history_store: OrderedDict[str, list[StudioCheckpoint]] = OrderedDict()

    @property
    def capabilities(self) -> list[str]:
        return ["graph", "streaming", "traces", "state", "checkpoints"]

    @property
    def _checkpoint_namespace(self) -> str:
        return f"studio:graph_agent:{self.agent_name}"

    async def get_graph(self) -> StudioGraphTopology:
        """Extract the true graph topology from AgentGraph.

        ``graph_fn`` is sync and may touch imports/connections — run it in a
        worker thread with a timeout so Studio Connect never wedges the loop.
        """
        import asyncio

        if not self._agent.graph_fn:
            return StudioGraphTopology(agent_name=self.agent_name)

        try:
            graph: AgentGraph = await asyncio.wait_for(
                asyncio.to_thread(self._agent.graph_fn),
                timeout=5.0,
            )
        except TimeoutError:
            return StudioGraphTopology(
                agent_name=self.agent_name,
                metadata={"error": "graph_fn timed out"},
            )
        except Exception as exc:  # noqa: BLE001
            return StudioGraphTopology(
                agent_name=self.agent_name,
                metadata={"error": f"graph_fn failed: {exc}"},
            )

        nodes: list[StudioGraphNode] = []
        # AgentGraph uses END sentinel
        END = "__END__"
        START = "__start__"

        # Add a synthetic start node pointing to the entrypoint if needed
        nodes.append(StudioGraphNode(id=START, name="Start", type="start"))

        for name, node in graph.nodes.items():
            nodes.append(
                StudioGraphNode(
                    id=name,
                    name=name,
                    type="processing",
                    metadata={"description": node.description} if node.description else {},
                )
            )

        # Ensure END node is present
        nodes.append(
            StudioGraphNode(
                id=END,
                name="End",
                type="end",
            )
        )

        edges: list[StudioGraphEdge] = []

        # Connect __start__ to entrypoint
        if graph.entrypoint:
            edges.append(
                StudioGraphEdge(
                    id=f"edge-{START}-to-{graph.entrypoint}", source=START, target=graph.entrypoint
                )
            )

        for source, edge in graph.edges.items():
            if isinstance(edge, str):
                edges.append(
                    StudioGraphEdge(id=f"edge-{source}-to-{edge}", source=source, target=edge)
                )
            else:
                # Conditional edge - we don't know the exact targets without executing,
                # so we point it to a synthetic condition node or directly to END as fallback.
                # In Studio, conditional edges ideally have a condition label.
                edges.append(
                    StudioGraphEdge(
                        id=f"edge-{source}-conditional",
                        source=source,
                        target=END,
                        condition="conditional",
                    )
                )

        # Connect finish node to END
        if graph.finish and graph.finish not in [e.source for e in edges if e.target == END]:
            edges.append(
                StudioGraphEdge(
                    id=f"edge-{graph.finish}-to-{END}", source=graph.finish, target=END
                )
            )

        return StudioGraphTopology(
            agent_name=self.agent_name,
            nodes=nodes,
            edges=edges,
            entry_point=START,
            end_points=[END],
        )

    async def stream_execution(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
        breakpoints: list[str] | None = None,
        checkpoint_id: str | None = None,
    ) -> AsyncGenerator[StudioRunEvent, None]:
        """Stream real execution events using AgentGraph.astream_studio_events."""
        config = config or {}
        run_id = config.get("run_id", "run_local")

        if not self._agent.graph_fn:
            # Fallback if somehow there's no graph
            yield StudioRunEvent(
                event="run_error",
                run_id=run_id,
                timestamp=_now_iso(),
                data={"error": "Agent has no graph_fn"},
            )
            return

        graph = self._agent.graph_fn()

        # Class agents use a dataclass state: convert the incoming raw dict via
        # ``input_to_state`` before streaming, otherwise the graph nodes receive
        # a dict and raise AttributeError (HTTP 500 / run_error in Studio).
        stream_state: Any = state
        instance = getattr(self._agent, "class_instance", None)
        prompt_guard = False
        if instance is not None:
            from agentomatic.agents.base import BaseGraphAgent
            from agentomatic.core.agent_invoke import _input_from_state

            if isinstance(instance, BaseGraphAgent):
                input_data = _input_from_state(state)
                instance._begin_request_prompt(input_data)
                prompt_guard = True
                stream_state = instance.input_to_state(input_data)

        thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")
        try:
            async for evt_dict in graph.astream_studio_events(stream_state, run_id):
                if evt_dict.get("event") == "node_end":
                    await self._record_node(thread_id, run_id, evt_dict)
                yield StudioRunEvent(**evt_dict)
        finally:
            if prompt_guard and instance is not None:
                instance._end_request_prompt()

    # ------------------------------------------------------------------
    # State & history (Studio Debug view)
    # ------------------------------------------------------------------

    async def _record_node(self, thread_id: str, run_id: str, event: dict[str, Any]) -> None:
        """Keep the state after each node as a checkpoint and as the thread state."""
        output = (event.get("data") or {}).get("output")
        state = output if isinstance(output, dict) else {}
        timestamp = str(event.get("timestamp") or _now_iso())
        history = self._history_store.setdefault(thread_id, [])
        self._history_store.move_to_end(thread_id)
        parent = history[-1] if history else None
        step = (parent.step if parent is not None else 0) + 1
        node = str(event.get("node") or "")
        checkpoint = StudioCheckpoint(
            id=f"ckpt_{uuid.uuid4().hex}",
            thread_id=thread_id,
            step=step,
            state=state,
            metadata={
                "node": node,
                "run_id": run_id,
                "duration_ms": event.get("duration_ms"),
                "step": step,
            },
            parent_id=parent.id if parent is not None else None,
            timestamp=timestamp,
        )
        history.append(checkpoint)
        del history[: -self._MAX_CHECKPOINTS_PER_THREAD]
        self._state_store[thread_id] = {
            "state": state,
            "checkpoint_id": checkpoint.id,
            "timestamp": timestamp,
        }
        self._state_store.move_to_end(thread_id)
        for cache in (self._history_store, self._state_store):
            while len(cache) > self._MAX_THREADS:
                cache.popitem(last=False)
        if self._store is not None:
            try:
                await self._store.save_checkpoint(
                    thread_id,
                    self._checkpoint_namespace,
                    checkpoint.id,
                    checkpoint.parent_id,
                    checkpoint.state,
                    checkpoint.metadata,
                )
            except Exception as exc:  # noqa: BLE001 - tracing must never break a run
                logger.warning(
                    "Studio could not persist checkpoint for '{}': {}", self.agent_name, exc
                )

    async def _stored_history(self, thread_id: str) -> list[StudioCheckpoint]:
        if self._store is None:
            return []
        try:
            records = await self._store.list_checkpoints(thread_id, self._checkpoint_namespace)
        except Exception as exc:  # noqa: BLE001 - history stays useful from the cache
            logger.warning("Studio could not list checkpoints for '{}': {}", self.agent_name, exc)
            return []
        out: list[StudioCheckpoint] = []
        for record in records or []:
            checkpoint = record.get("checkpoint") if isinstance(record, dict) else None
            checkpoint_id = record.get("checkpoint_id") if isinstance(record, dict) else None
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint_id, str):
                continue
            metadata = record.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            parent_id = record.get("parent_checkpoint_id")
            out.append(
                StudioCheckpoint(
                    id=checkpoint_id,
                    thread_id=thread_id,
                    step=int(metadata.get("step", 0) or 0),
                    state=checkpoint,
                    metadata=metadata,
                    parent_id=parent_id if isinstance(parent_id, str) else None,
                    timestamp=str(record.get("created_at") or _now_iso()),
                )
            )
        return out

    async def get_state(self, thread_id: str) -> StudioStateSnapshot | None:
        """The state after the thread's most recent node (restored from storage)."""
        cached = self._state_store.get(thread_id)
        if cached is None:
            stored = await self._stored_history(thread_id)
            if not stored:
                return None
            latest = max(stored, key=lambda c: (c.timestamp, c.step))
            cached = {
                "state": latest.state,
                "checkpoint_id": latest.id,
                "timestamp": latest.timestamp,
            }
        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=self.agent_name,
            state=dict(cached["state"]),
            timestamp=cached["timestamp"],
            checkpoint_id=cached.get("checkpoint_id"),
        )

    async def update_state(
        self, thread_id: str, updates: dict[str, Any]
    ) -> StudioStateSnapshot | None:
        """Merge ``updates`` into the inspected state (not replayed into the graph)."""
        current = self._state_store.get(thread_id) or {"state": {}, "checkpoint_id": None}
        merged = {**current.get("state", {}), **updates}
        self._state_store[thread_id] = {
            "state": merged,
            "checkpoint_id": current.get("checkpoint_id"),
            "timestamp": _now_iso(),
        }
        return await self.get_state(thread_id)

    async def get_history(self, thread_id: str) -> list[StudioCheckpoint]:
        """Every node's checkpoint for the thread, newest first."""
        merged = {c.id: c for c in await self._stored_history(thread_id)}
        merged.update({c.id: c for c in self._history_store.get(thread_id, [])})
        return sorted(merged.values(), key=lambda c: (c.timestamp, c.step), reverse=True)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
