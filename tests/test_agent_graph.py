"""Tests for the graph builder and runtime (AgentGraph, GraphBuilder)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agentomatic.agents.builder import GraphBuilder
from agentomatic.agents.graph import END, AgentGraph, GraphNode

# ---------------------------------------------------------------------------
# Test state
# ---------------------------------------------------------------------------


@dataclass
class GraphState:
    """Simple dataclass state for graph tests."""

    value: int = 0
    steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_fn(name: str):
    """Create a handler that records its name and increments value."""

    def handler(state: GraphState) -> GraphState:
        state.steps.append(name)
        state.value += 1
        return state

    return handler


def _build_linear_graph() -> AgentGraph[GraphState]:
    """Build a simple a → b → c linear graph."""
    return (
        GraphBuilder[GraphState]()
        .node("a", _node_fn("a"))
        .node("b", _node_fn("b"))
        .node("c", _node_fn("c"))
        .edge("a", "b")
        .edge("b", "c")
        .entrypoint("a")
        .finish("c")
        .build()
    )


# ===========================================================================
# GraphBuilder tests
# ===========================================================================


class TestGraphBuilderNode:
    """Test GraphBuilder.node()."""

    def test_node_adds_node(self):
        """GraphBuilder.node() should register a node."""
        builder = GraphBuilder[GraphState]()
        result = builder.node("step", _node_fn("step"))
        assert "step" in builder._nodes
        assert result is builder  # chaining

    def test_node_returns_self_for_chaining(self):
        """GraphBuilder.node() should return self for fluent API."""
        builder = GraphBuilder[GraphState]()
        result = builder.node("a", _node_fn("a")).node("b", _node_fn("b"))
        assert result is builder

    def test_node_rejects_duplicate_names(self):
        """GraphBuilder.node() should reject duplicate node names."""
        builder = GraphBuilder[GraphState]()
        builder.node("a", _node_fn("a"))
        with pytest.raises(ValueError, match="Duplicate node name"):
            builder.node("a", _node_fn("a2"))

    def test_node_stores_description_and_metadata(self):
        """GraphBuilder.node() should preserve description and metadata."""
        builder = GraphBuilder[GraphState]()
        builder.node(
            "a",
            _node_fn("a"),
            description="My node",
            metadata={"key": "value"},
        )
        node = builder._nodes["a"]
        assert node.description == "My node"
        assert node.metadata == {"key": "value"}


class TestGraphBuilderEdge:
    """Test GraphBuilder.edge()."""

    def test_edge_adds_edge(self):
        """GraphBuilder.edge() should register a directed edge."""
        builder = GraphBuilder[GraphState]()
        builder.node("a", _node_fn("a")).node("b", _node_fn("b"))
        result = builder.edge("a", "b")
        assert builder._edges["a"] == "b"
        assert result is builder

    def test_edge_to_end_sentinel(self):
        """GraphBuilder.edge() should accept END sentinel."""
        builder = GraphBuilder[GraphState]()
        builder.node("a", _node_fn("a"))
        builder.edge("a", END)
        assert builder._edges["a"] == END


class TestGraphBuilderConditionalEdge:
    """Test GraphBuilder.conditional_edge()."""

    def test_conditional_edge_without_routes(self):
        """conditional_edge() without routes stores the callable directly."""
        builder = GraphBuilder[GraphState]()
        builder.node("a", _node_fn("a"))

        def condition(state: GraphState) -> str:
            return "b"

        builder.conditional_edge("a", condition)
        assert callable(builder._edges["a"])

    def test_conditional_edge_with_routes(self):
        """conditional_edge() with routes wraps condition in a router."""
        builder = GraphBuilder[GraphState]()
        builder.node("a", _node_fn("a"))
        builder.node("b", _node_fn("b"))
        builder.node("c", _node_fn("c"))

        def condition(state: GraphState) -> str:
            return "yes" if state.value > 0 else "no"

        builder.conditional_edge(
            "a",
            condition,
            routes={"yes": "b", "no": "c"},
        )
        edge_fn = builder._edges["a"]
        assert callable(edge_fn)

        # Test the router mapping
        state_yes = GraphState(value=1)
        assert edge_fn(state_yes) == "b"
        state_no = GraphState(value=0)
        assert edge_fn(state_no) == "c"


class TestGraphBuilderBuild:
    """Test GraphBuilder.build()."""

    def test_build_returns_agent_graph(self):
        """build() should return a valid AgentGraph."""
        graph = _build_linear_graph()
        assert isinstance(graph, AgentGraph)
        assert graph.entrypoint == "a"
        assert graph.finish == "c"

    def test_build_rejects_missing_entrypoint(self):
        """build() should reject graph with no entrypoint."""
        builder = GraphBuilder[GraphState]().node("a", _node_fn("a")).finish("a")
        with pytest.raises(ValueError, match="entrypoint"):
            builder.build()

    def test_build_rejects_missing_finish(self):
        """build() should reject graph with no finish node."""
        builder = GraphBuilder[GraphState]().node("a", _node_fn("a")).entrypoint("a")
        with pytest.raises(ValueError, match="finish"):
            builder.build()

    def test_build_rejects_edges_to_nonexistent_nodes(self):
        """build() should reject edges pointing to non-existent nodes."""
        builder = (
            GraphBuilder[GraphState]()
            .node("a", _node_fn("a"))
            .edge("a", "nonexistent")
            .entrypoint("a")
            .finish("a")
        )
        with pytest.raises(ValueError):
            builder.build()


# ===========================================================================
# AgentGraph.validate() tests
# ===========================================================================


class TestAgentGraphValidate:
    """Test AgentGraph.validate()."""

    def test_valid_graph_returns_no_errors(self):
        """validate() should return empty list for valid graph."""
        graph = _build_linear_graph()
        errors = graph.validate()
        assert errors == []

    def test_missing_entrypoint(self):
        """validate() should catch missing entrypoint."""
        graph = AgentGraph(
            nodes={"a": GraphNode("a", _node_fn("a"))},
            edges={},
            entrypoint="",
            finish="a",
        )
        errors = graph.validate()
        assert any("entrypoint" in e.lower() for e in errors)

    def test_missing_finish(self):
        """validate() should catch missing finish node."""
        graph = AgentGraph(
            nodes={"a": GraphNode("a", _node_fn("a"))},
            edges={},
            entrypoint="a",
            finish="",
        )
        errors = graph.validate()
        assert any("finish" in e.lower() for e in errors)

    def test_nonexistent_entrypoint_node(self):
        """validate() should catch entrypoint referencing nonexistent node."""
        graph = AgentGraph(
            nodes={"a": GraphNode("a", _node_fn("a"))},
            edges={},
            entrypoint="missing",
            finish="a",
        )
        errors = graph.validate()
        assert any("missing" in e for e in errors)

    def test_edge_source_not_in_nodes(self):
        """validate() should catch edges from non-existent source nodes."""
        graph = AgentGraph(
            nodes={"a": GraphNode("a", _node_fn("a"))},
            edges={"nonexistent": "a"},
            entrypoint="a",
            finish="a",
        )
        errors = graph.validate()
        assert any("nonexistent" in e for e in errors)

    def test_catches_all_error_types_at_once(self):
        """validate() should return multiple errors simultaneously."""
        graph = AgentGraph(
            nodes={},
            edges={},
            entrypoint="",
            finish="",
        )
        errors = graph.validate()
        assert len(errors) >= 2  # at least entrypoint + finish


# ===========================================================================
# AgentGraph.invoke() tests
# ===========================================================================


class TestAgentGraphInvoke:
    """Test AgentGraph.invoke()."""

    def test_executes_nodes_in_correct_order(self):
        """invoke() should execute nodes following edges in order."""
        graph = _build_linear_graph()
        state = GraphState()
        final = graph.invoke(state)
        assert final.steps == ["a", "b", "c"]

    def test_passes_state_between_nodes(self):
        """invoke() should pass state through the chain."""
        graph = _build_linear_graph()
        state = GraphState(value=10)
        final = graph.invoke(state)
        assert final.value == 13  # 10 + 3 nodes

    def test_stops_at_finish_node(self):
        """invoke() should stop execution at the finish node."""

        def failing_handler(state: GraphState) -> GraphState:
            raise RuntimeError("Should not be called")

        graph = (
            GraphBuilder[GraphState]()
            .node("a", _node_fn("a"))
            .node("b", _node_fn("b"))
            .edge("a", "b")
            .entrypoint("a")
            .finish("b")
            .build()
        )
        state = GraphState()
        final = graph.invoke(state)
        assert final.steps == ["a", "b"]
        assert final.value == 2

    def test_raises_on_invalid_graph(self):
        """invoke() should raise ValueError on invalid graph."""
        graph = AgentGraph(nodes={}, edges={}, entrypoint="", finish="")
        with pytest.raises(ValueError, match="Invalid graph"):
            graph.invoke(GraphState())

    def test_raises_on_node_execution_error(self):
        """invoke() should raise RuntimeError on node failure."""

        def bad_handler(state: GraphState) -> GraphState:
            raise ValueError("boom")

        graph = (
            GraphBuilder[GraphState]().node("a", bad_handler).entrypoint("a").finish("a").build()
        )
        with pytest.raises(RuntimeError, match="boom"):
            graph.invoke(GraphState())

    def test_records_trace_events(self):
        """invoke() should populate _last_trace with TraceEvent objects."""
        graph = _build_linear_graph()
        graph.invoke(GraphState())
        trace = graph.last_trace
        assert len(trace) == 3
        assert trace[0].node_name == "a"
        assert trace[1].node_name == "b"
        assert trace[2].node_name == "c"
        for event in trace:
            assert event.status == "success"
            assert event.duration_ms >= 0

    def test_trace_records_error_on_failure(self):
        """invoke() trace should record error status on node failure."""

        def bad_handler(state: GraphState) -> GraphState:
            raise ValueError("failed")

        graph = (
            GraphBuilder[GraphState]().node("a", bad_handler).entrypoint("a").finish("a").build()
        )
        with pytest.raises(RuntimeError):
            graph.invoke(GraphState())

        trace = graph.last_trace
        assert len(trace) == 1
        assert trace[0].status == "error"
        assert trace[0].error == "failed"

    def test_conditional_edge_routes_correctly(self):
        """invoke() should follow conditional edges correctly."""

        def router(state: GraphState) -> str:
            return "high" if state.value >= 5 else "low"

        graph = (
            GraphBuilder[GraphState]()
            .node("start", _node_fn("start"))
            .node("high", _node_fn("high"))
            .node("low", _node_fn("low"))
            .conditional_edge("start", router, {"high": "high", "low": "low"})
            .entrypoint("start")
            .finish("high")
            .build()
        )
        state = GraphState(value=10)
        final = graph.invoke(state)
        assert "high" in final.steps
        assert "low" not in final.steps

    def test_end_sentinel_stops_execution(self):
        """invoke() should stop when a conditional edge returns END."""

        def always_end(state: GraphState) -> str:
            return END

        graph = (
            GraphBuilder[GraphState]()
            .node("a", _node_fn("a"))
            .node("b", _node_fn("b"))
            .conditional_edge("a", always_end)
            .entrypoint("a")
            .finish("b")
            .build()
        )
        state = GraphState()
        final = graph.invoke(state)
        assert final.steps == ["a"]

    def test_single_node_graph(self):
        """invoke() should work with a single-node graph."""
        graph = (
            GraphBuilder[GraphState]()
            .node("only", _node_fn("only"))
            .entrypoint("only")
            .finish("only")
            .build()
        )
        state = GraphState()
        final = graph.invoke(state)
        assert final.steps == ["only"]
        assert final.value == 1


# ===========================================================================
# AgentGraph.ainvoke() tests
# ===========================================================================


class TestAgentGraphAInvoke:
    """Test AgentGraph.ainvoke()."""

    @pytest.mark.asyncio
    async def test_ainvoke_with_sync_handlers(self):
        """ainvoke() should work with synchronous node handlers."""
        graph = _build_linear_graph()
        state = GraphState()
        final = await graph.ainvoke(state)
        assert final.steps == ["a", "b", "c"]
        assert final.value == 3

    @pytest.mark.asyncio
    async def test_ainvoke_with_async_handlers(self):
        """ainvoke() should await async node handlers."""

        async def async_handler(state: GraphState) -> GraphState:
            state.steps.append("async_node")
            state.value += 10
            return state

        graph = (
            GraphBuilder[GraphState]().node("a", async_handler).entrypoint("a").finish("a").build()
        )
        state = GraphState()
        final = await graph.ainvoke(state)
        assert final.steps == ["async_node"]
        assert final.value == 10

    @pytest.mark.asyncio
    async def test_ainvoke_records_trace(self):
        """ainvoke() should record trace events like invoke()."""
        graph = _build_linear_graph()
        await graph.ainvoke(GraphState())
        trace = graph.last_trace
        assert len(trace) == 3


# ===========================================================================
# AgentGraph.visualize() tests
# ===========================================================================


class TestAgentGraphVisualize:
    """Test AgentGraph.visualize()."""

    def test_visualize_returns_mermaid_string(self):
        """visualize() should return a valid Mermaid diagram."""
        graph = _build_linear_graph()
        mermaid = graph.visualize()
        assert "graph TD" in mermaid
        assert "a" in mermaid
        assert "b" in mermaid
        assert "c" in mermaid

    def test_visualize_includes_start_and_done(self):
        """visualize() should include START and DONE markers."""
        graph = _build_linear_graph()
        mermaid = graph.visualize()
        assert "START" in mermaid
        assert "DONE" in mermaid

    def test_visualize_shows_edges(self):
        """visualize() should show edges between nodes."""
        graph = _build_linear_graph()
        mermaid = graph.visualize()
        assert "a --> b" in mermaid
        assert "b --> c" in mermaid


# ===========================================================================
# GraphNode tests
# ===========================================================================


class TestGraphNode:
    """Test GraphNode dataclass."""

    def test_node_callable(self):
        """GraphNode should be directly callable."""
        node = GraphNode("test", _node_fn("test"))
        state = GraphState()
        result = node(state)
        assert result.steps == ["test"]
        assert result.value == 1

    def test_node_is_frozen(self):
        """GraphNode should be immutable (frozen dataclass)."""
        node = GraphNode("test", _node_fn("test"))
        with pytest.raises(AttributeError):
            node.name = "changed"  # type: ignore[misc]

    def test_node_names(self):
        """AgentGraph.node_names should return all node names."""
        graph = _build_linear_graph()
        assert sorted(graph.node_names) == ["a", "b", "c"]
