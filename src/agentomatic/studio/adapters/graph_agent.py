"""Adapter for class-based AgentGraph agents."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

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

    def __init__(
        self,
        agent: RegisteredAgent,
        store: BaseStore | None = None,
    ) -> None:
        super().__init__(agent.name)
        self._agent = agent
        self._store = store

    @property
    def capabilities(self) -> list[str]:
        return ["graph", "streaming", "traces"]

    async def get_graph(self) -> StudioGraphTopology:
        """Extract the true graph topology from AgentGraph."""
        if not self._agent.graph_fn:
            return StudioGraphTopology(agent_name=self.agent_name)

        graph: AgentGraph = self._agent.graph_fn()

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
            from datetime import UTC, datetime

            def _now_iso() -> str:
                return datetime.now(UTC).isoformat()

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
        if instance is not None:
            from agentomatic.agents.base import BaseGraphAgent
            from agentomatic.core.agent_invoke import _input_from_state

            if isinstance(instance, BaseGraphAgent):
                stream_state = instance.input_to_state(_input_from_state(state))

        async for evt_dict in graph.astream_studio_events(stream_state, run_id):
            yield StudioRunEvent(**evt_dict)

    async def get_state(self, thread_id: str) -> StudioStateSnapshot | None:
        return None

    async def update_state(
        self, thread_id: str, updates: dict[str, Any]
    ) -> StudioStateSnapshot | None:
        return None

    async def get_history(self, thread_id: str) -> list[StudioCheckpoint]:
        return []
