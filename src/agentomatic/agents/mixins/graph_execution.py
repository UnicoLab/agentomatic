"""Graph execution mixin — lazy graph creation and invocation.

Provides the core graph lifecycle:
- Lazy ``build_graph()`` / ``graph`` property
- ``invoke_graph()`` / ``ainvoke_graph()`` execution helpers
- ``build_graph_from_decorated_nodes()`` — auto-builds from ``@agent_node``
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic

from loguru import logger

from ..decorators import AGENT_NODE_ATTR
from ..types import StateT

if TYPE_CHECKING:
    from ..decorators import AgentNodeMeta
    from ..graph import AgentGraph


class GraphExecutionMixin(Generic[StateT]):
    """Mixin that owns and manages an ``AgentGraph`` lifecycle.

    Subclasses must implement ``build_graph()`` to define the graph
    topology. The graph is lazily created on first access and can be
    invalidated to force a rebuild.

    Example::

        class MyAgent(GraphExecutionMixin[MyState]):
            def build_graph(self) -> AgentGraph[MyState]:
                return (
                    GraphBuilder[MyState]()
                    .node("step", self.step)
                    .entrypoint("step")
                    .finish("step")
                    .build()
                )
    """

    _graph: AgentGraph[StateT] | None = None

    @abstractmethod
    def build_graph(self) -> AgentGraph[StateT]:
        """Construct the agent graph.

        Must be implemented by subclasses. Called lazily on first
        ``graph`` property access.

        Returns:
            A fully configured ``AgentGraph``.
        """
        ...

    @property
    def graph(self) -> AgentGraph[StateT]:
        """Lazy graph creation — builds on first access.

        Returns:
            The cached ``AgentGraph`` instance.
        """
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    def invalidate_graph(self) -> None:
        """Clear cached graph. Next access rebuilds it."""
        self._graph = None

    def invoke_graph(self, state: StateT) -> StateT:
        """Execute the graph synchronously.

        Args:
            state: Initial state to pass through the graph.

        Returns:
            Final state after graph execution.
        """
        return self.graph.invoke(state)

    async def ainvoke_graph(self, state: StateT) -> StateT:
        """Execute the graph asynchronously.

        Args:
            state: Initial state to pass through the graph.

        Returns:
            Final state after graph execution.
        """
        return await self.graph.ainvoke(state)

    def build_graph_from_decorated_nodes(self) -> AgentGraph[StateT]:
        """Auto-build graph from ``@agent_node`` decorated methods.

        Introspects all methods on ``self`` for the
        ``_agent_node_meta`` attribute (set by ``@agent_node``).
        Builds nodes and edges via ``GraphBuilder``, then returns
        the validated graph.

        Returns:
            A validated ``AgentGraph`` built from decorated methods.

        Raises:
            ValueError: If no entrypoint or finish node is found,
                or if the graph fails validation.
        """
        from ..builder import GraphBuilder as _GraphBuilder

        builder: _GraphBuilder[StateT] = _GraphBuilder()
        entrypoint_name: str | None = None
        finish_name: str | None = None

        # Collect decorated methods
        node_methods: list[tuple[str, object, AgentNodeMeta]] = []

        for attr_name in dir(self):
            try:
                method = getattr(self, attr_name)
            except Exception:  # noqa: BLE001
                continue

            meta: AgentNodeMeta | None = getattr(method, AGENT_NODE_ATTR, None)
            if meta is None:
                continue

            node_name = meta.name or method.__name__
            node_methods.append((node_name, method, meta))

        logger.debug(
            "Found {} decorated nodes: {}",
            len(node_methods),
            [n for n, _, _ in node_methods],
        )

        # Register nodes and track entrypoint / finish
        for node_name, method, meta in node_methods:
            from typing import Callable, Any, cast
            typed_method = cast(Callable[[Any], Any], method)
            builder.node(
                node_name,
                typed_method,
                description=meta.description,
                metadata=meta.metadata,
            )

            if meta.entrypoint:
                if entrypoint_name is not None:
                    raise ValueError(
                        f"Multiple entrypoints: '{entrypoint_name}' and '{node_name}'"
                    )
                entrypoint_name = node_name

            if meta.finish:
                if finish_name is not None:
                    raise ValueError(f"Multiple finish nodes: '{finish_name}' and '{node_name}'")
                finish_name = node_name

        # Register edges from ``after`` declarations
        for node_name, _method, meta in node_methods:
            if meta.after is not None:
                builder.edge(meta.after, node_name)

        # Set entrypoint and finish
        if entrypoint_name is None:
            raise ValueError(
                "No entrypoint found. Decorate one method with @agent_node(entrypoint=True)."
            )
        if finish_name is None:
            raise ValueError(
                "No finish node found. Decorate one method with @agent_node(finish=True)."
            )

        builder.entrypoint(entrypoint_name)
        builder.finish(finish_name)

        return builder.build()
