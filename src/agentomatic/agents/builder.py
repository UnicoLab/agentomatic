"""Fluent builder API for constructing agent graphs.

Provides both **fluent chaining** and **LangGraph-style** APIs::

    # --- LangGraph-style (recommended) ---
    g = GraphBuilder[MyState]()
    g.add_node("extract", self.extract)
    g.add_node("generate", self.generate)
    g.set_entry_point("extract")
    g.add_edge("extract", "generate")
    g.set_finish_point("generate")
    return g.compile()

    # --- Fluent chaining ---
    graph = (
        GraphBuilder[MyState]()
        .node("extract", self.extract)
        .node("generate", self.generate)
        .edge("extract", "generate")
        .entrypoint("extract")
        .finish("generate")
        .build()
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic

from .graph import END, AgentGraph, GraphNode
from .types import StateT


class GraphBuilder(Generic[StateT]):
    """Builder for constructing ``AgentGraph`` instances.

    Provides two equivalent APIs:

    - **LangGraph-style**: ``add_node()``, ``add_edge()``,
      ``set_entry_point()``, ``set_finish_point()``,
      ``add_conditional_edge()``, ``compile()``
    - **Fluent chaining**: ``node()``, ``edge()``,
      ``entrypoint()``, ``finish()``, ``build()``
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode[StateT]] = {}
        self._edges: dict[str, str | Callable[..., str]] = {}
        self._entrypoint: str = ""
        self._finish: str = ""

    # ==============================================================
    # Core API (fluent chaining)
    # ==============================================================

    def node(
        self,
        name: str,
        handler: Callable[[StateT], StateT],
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphBuilder[StateT]:
        """Add a node to the graph.

        Args:
            name: Unique node name.
            handler: Callable that takes state and returns state.
            description: Optional human-readable description.
            metadata: Arbitrary metadata.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If a node with this name already exists.
        """
        if name in self._nodes:
            raise ValueError(f"Duplicate node name: '{name}'")
        self._nodes[name] = GraphNode(
            name=name,
            handler=handler,
            description=description,
            metadata=metadata or {},
        )
        return self

    def edge(
        self,
        source: str,
        target: str,
    ) -> GraphBuilder[StateT]:
        """Add a directed edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name (or ``END``).

        Returns:
            Self for chaining.
        """
        self._edges[source] = target
        return self

    def conditional_edge(
        self,
        source: str,
        condition: Callable[[StateT], str],
        routes: dict[str, str] | None = None,
    ) -> GraphBuilder[StateT]:
        """Add a conditional edge.

        The ``condition`` function receives the state and returns
        the name of the next node. Optionally, ``routes`` maps
        return values to node names.

        Args:
            source: Source node name.
            condition: Function that returns the next node name.
            routes: Optional mapping of condition return values
                to node names.

        Returns:
            Self for chaining.
        """
        if routes:
            original = condition

            def routed_condition(state: StateT) -> str:
                result = original(state)
                return routes.get(str(result), END)

            self._edges[source] = routed_condition
        else:
            self._edges[source] = condition
        return self

    def entrypoint(self, name: str) -> GraphBuilder[StateT]:
        """Set the graph entrypoint.

        Args:
            name: Name of the entry node.

        Returns:
            Self for chaining.
        """
        self._entrypoint = name
        return self

    def finish(self, name: str) -> GraphBuilder[StateT]:
        """Set the finish node.

        Args:
            name: Name of the finish node.

        Returns:
            Self for chaining.
        """
        self._finish = name
        return self

    def build(self) -> AgentGraph[StateT]:
        """Build and validate the graph.

        Returns:
            A validated ``AgentGraph`` instance.

        Raises:
            ValueError: If the graph is invalid.
        """
        graph = AgentGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            entrypoint=self._entrypoint,
            finish=self._finish,
        )

        errors = graph.validate()
        if errors:
            raise ValueError("Graph validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return graph

    # ==============================================================
    # LangGraph-compatible aliases
    # ==============================================================

    def add_node(
        self,
        name: str,
        handler: Callable[[StateT], StateT],
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphBuilder[StateT]:
        """Add a node (LangGraph-compatible alias for ``node()``)."""
        return self.node(
            name,
            handler,
            description=description,
            metadata=metadata,
        )

    def add_edge(
        self,
        source: str,
        target: str,
    ) -> GraphBuilder[StateT]:
        """Add an edge (LangGraph-compatible alias for ``edge()``)."""
        return self.edge(source, target)

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[StateT], str],
        routes: dict[str, str] | None = None,
    ) -> GraphBuilder[StateT]:
        """Add conditional edge (alias for ``conditional_edge()``)."""
        return self.conditional_edge(source, condition, routes)

    def set_entry_point(self, name: str) -> GraphBuilder[StateT]:
        """Set entrypoint (LangGraph-compatible alias)."""
        return self.entrypoint(name)

    def set_finish_point(self, name: str) -> GraphBuilder[StateT]:
        """Set finish node (LangGraph-compatible alias)."""
        return self.finish(name)

    def compile(self) -> AgentGraph[StateT]:
        """Compile the graph (LangGraph-compatible alias for ``build()``)."""
        return self.build()
