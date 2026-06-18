"""Extract and serialize graph topology from registered agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.studio.models import (
    StudioGraphEdge,
    StudioGraphNode,
    StudioGraphTopology,
)

if TYPE_CHECKING:
    from agentomatic.core.manifest import RegisteredAgent


class GraphInspector:
    """Extract and serialize graph topology from registered agents.

    Supports LangGraph ``CompiledGraph`` objects (via ``get_graph()``) as well
    as simple ``node_fn``-only agents (synthesises a linear start → agent → end
    topology).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self, agent: RegisteredAgent) -> StudioGraphTopology:
        """Inspect an agent and return its graph topology.

        Args:
            agent: The registered agent to inspect.

        Returns:
            A :class:`StudioGraphTopology` describing the agent's execution
            graph.  Returns an empty topology if the agent exposes neither
            ``graph_fn`` nor ``node_fn``.
        """
        if agent.graph_fn:
            try:
                return self._from_compiled_graph(agent)
            except Exception as exc:
                logger.warning(
                    f"Failed to inspect graph for agent '{agent.name}': {exc}"
                )
                # Fall through to node_fn or empty
        if agent.node_fn:
            return self._from_node_fn(agent)
        return StudioGraphTopology(agent_name=agent.name, nodes=[], edges=[])

    def get_capabilities(self, agent: RegisteredAgent) -> list[str]:
        """Determine what debugging capabilities an agent supports.

        Args:
            agent: The registered agent to probe.

        Returns:
            List of capability strings such as ``'graph'``, ``'invoke'``,
            ``'checkpoints'``, ``'streaming'``, ``'threads'``, ``'hitl'``.
        """
        caps: list[str] = []

        if agent.graph_fn:
            caps.append("graph")
            # Probe for a checkpointer on the compiled graph
            try:
                graph = agent.graph_fn()
                if getattr(graph, "checkpointer", None) is not None:
                    caps.append("checkpoints")
            except Exception:
                pass

        if agent.node_fn:
            caps.append("invoke")

        # All agents support streaming via the platform router
        caps.append("streaming")
        # Thread management is always available
        caps.append("threads")

        # HITL support heuristic — LangGraph agents commonly support it
        if agent.manifest.framework == "langgraph":
            caps.append("hitl")

        return caps

    # ------------------------------------------------------------------
    # Internal — CompiledGraph inspection
    # ------------------------------------------------------------------

    def _from_compiled_graph(self, agent: RegisteredAgent) -> StudioGraphTopology:
        """Extract topology from a LangGraph ``CompiledGraph``.

        Uses ``graph.get_graph()`` which returns a ``DrawableGraph`` with
        ``.nodes`` (dict[str, NodeData]) and ``.edges`` (list[Edge]).
        """
        graph = agent.graph_fn()  # type: ignore[misc]
        drawable = graph.get_graph()

        # --- Nodes ---
        nodes: list[StudioGraphNode] = []
        nodes_dict = getattr(drawable, "nodes", {})
        for node_id, node_data in nodes_dict.items():
            node_type = self._classify_node(node_id, node_data)
            node_name = getattr(node_data, "name", None) or str(node_id)
            node_meta = getattr(node_data, "metadata", None) or {}
            if not isinstance(node_meta, dict):
                node_meta = {}
            nodes.append(
                StudioGraphNode(
                    id=str(node_id),
                    name=node_name,
                    type=node_type,
                    metadata=node_meta,
                )
            )

        # --- Edges ---
        edges: list[StudioGraphEdge] = []
        raw_edges = getattr(drawable, "edges", [])
        for idx, edge in enumerate(raw_edges):
            source = getattr(edge, "source", None)
            target = getattr(edge, "target", None)
            if source is None or target is None:
                continue
            # Conditional label (DrawableGraph edges may expose .data or .conditional)
            condition = (
                getattr(edge, "conditional", None)
                or getattr(edge, "data", None)
            )
            if condition is True:
                condition = "conditional"
            elif condition and not isinstance(condition, str):
                condition = str(condition)
            edges.append(
                StudioGraphEdge(
                    id=f"edge_{idx}",
                    source=str(source),
                    target=str(target),
                    condition=condition if condition else None,
                )
            )

        # --- Entry / exit points ---
        entry_point: str | None = None
        end_points: list[str] = []
        for node in nodes:
            if node.type == "start" or node.id == "__start__":
                entry_point = node.id
            elif node.type == "end" or node.id == "__end__":
                end_points.append(node.id)

        return StudioGraphTopology(
            agent_name=agent.name,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            end_points=end_points,
        )

    # ------------------------------------------------------------------
    # Internal — node_fn-only agents
    # ------------------------------------------------------------------

    def _from_node_fn(self, agent: RegisteredAgent) -> StudioGraphTopology:
        """Create a minimal linear graph for simple ``node_fn`` agents."""
        return StudioGraphTopology(
            agent_name=agent.name,
            nodes=[
                StudioGraphNode(id="__start__", name="Start", type="start"),
                StudioGraphNode(
                    id=agent.name,
                    name=agent.manifest.name,
                    type="agent",
                ),
                StudioGraphNode(id="__end__", name="End", type="end"),
            ],
            edges=[
                StudioGraphEdge(id="edge_0", source="__start__", target=agent.name),
                StudioGraphEdge(id="edge_1", source=agent.name, target="__end__"),
            ],
            entry_point="__start__",
            end_points=["__end__"],
        )

    # ------------------------------------------------------------------
    # Internal — node classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_node(node_id: str, node_data: Any) -> str:
        """Classify a node type based on its ID and metadata.

        Applies simple heuristics to determine whether a node is a start,
        end, tool, condition, human-in-the-loop, or regular agent node.
        """
        if node_id == "__start__":
            return "start"
        if node_id == "__end__":
            return "end"

        name = (getattr(node_data, "name", None) or str(node_id)).lower()

        if "tool" in name:
            return "tool"
        if any(kw in name for kw in ("condition", "router", "route", "branch")):
            return "condition"
        if any(kw in name for kw in ("human", "approval", "review")):
            return "human"

        return "agent"
