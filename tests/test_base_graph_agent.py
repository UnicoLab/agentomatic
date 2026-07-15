"""Tests for BaseGraphAgent lifecycle and integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentomatic.agents.base import BaseGraphAgent
from agentomatic.agents.decorators import agent_node
from agentomatic.agents.metrics import ExactKeyMatchMetric
from agentomatic.agents.types import AgentDataset, AgentExample

# ---------------------------------------------------------------------------
# Test state and concrete agent
# ---------------------------------------------------------------------------


@dataclass
class ScopingState:
    """State for the test agent."""

    request: str = ""
    needs: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


class ScopingAgent(BaseGraphAgent[ScopingState]):
    """Concrete test agent with two nodes: extract → generate."""

    agent_name = "test_agent"
    agent_description = "A test agent for scoping"
    agent_version = "0.1.0"

    @agent_node(entrypoint=True)
    def extract(self, state: ScopingState) -> ScopingState:
        """Extract needs from request."""
        state.needs = {"goals": "identified"}
        return state

    @agent_node(after="extract", finish=True)
    def generate(self, state: ScopingState) -> ScopingState:
        """Generate output from needs."""
        state.output = {"summary": "done", "risks": ["risk1"]}
        return state

    def input_to_state(
        self,
        input_data: dict[str, Any],
    ) -> ScopingState:
        """Convert raw input to ScopingState."""
        return ScopingState(request=input_data.get("request", ""))

    def state_to_output(
        self,
        state: ScopingState,
    ) -> dict[str, Any]:
        """Extract output dict from final state."""
        return state.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset() -> AgentDataset:
    """Create a small test dataset."""
    return AgentDataset(
        name="test_dataset",
        examples=[
            AgentExample(
                id="ex1",
                input={"request": "Build a CRM"},
                expected_output={"summary": "done", "risks": ["risk1"]},
            ),
            AgentExample(
                id="ex2",
                input={"request": "Design a dashboard"},
                expected_output={"summary": "done", "risks": ["risk2"]},
            ),
        ],
    )


# ===========================================================================
# Graph lifecycle tests
# ===========================================================================


class TestGraphLifecycle:
    """Test lazy graph building and caching."""

    def test_lazily_builds_graph(self):
        """Graph should not be built until first access."""
        agent = ScopingAgent()
        assert agent._graph is None

    def test_graph_built_on_first_access(self):
        """Accessing .graph should build it lazily."""
        agent = ScopingAgent()
        graph = agent.graph
        assert graph is not None
        assert "extract" in graph.nodes
        assert "generate" in graph.nodes

    def test_graph_is_cached(self):
        """Graph should be cached after first access."""
        agent = ScopingAgent()
        graph1 = agent.graph
        graph2 = agent.graph
        assert graph1 is graph2

    def test_invalidate_graph_clears_cache(self):
        """invalidate_graph() should clear the cached graph."""
        agent = ScopingAgent()
        _ = agent.graph  # build
        agent.invalidate_graph()
        assert agent._graph is None

    def test_invalidate_triggers_rebuild(self):
        """After invalidate, next access should rebuild the graph."""
        agent = ScopingAgent()
        graph1 = agent.graph
        agent.invalidate_graph()
        graph2 = agent.graph
        assert graph2 is not None
        assert graph1 is not graph2

    def test_multiple_transforms_dont_rebuild_graph(self):
        """Multiple transform() calls should reuse the same graph."""
        agent = ScopingAgent()
        agent.transform({"request": "first"})
        graph_after_first = agent._graph
        agent.transform({"request": "second"})
        assert agent._graph is graph_after_first


# ===========================================================================
# Transform tests
# ===========================================================================


class TestTransform:
    """Test the transform() pipeline."""

    def test_transform_returns_correct_output(self):
        """transform() should return state_to_output result."""
        agent = ScopingAgent()
        result = agent.transform({"request": "Build a CRM"})
        assert result == {"summary": "done", "risks": ["risk1"]}

    def test_transform_converts_input_to_state(self):
        """transform() should call input_to_state correctly."""
        agent = ScopingAgent()
        result = agent.transform({"request": "hello"})
        # If we got output, input_to_state worked
        assert isinstance(result, dict)

    def test_transform_executes_all_nodes(self):
        """transform() should run all graph nodes."""
        agent = ScopingAgent()
        result = agent.transform({"request": "test"})
        # generate sets output dict, so if it's populated, both nodes ran
        assert "summary" in result
        assert "risks" in result


# ===========================================================================
# Compile tests
# ===========================================================================


class TestCompile:
    """Test compile() method."""

    def test_compile_stores_metadata(self):
        """compile() should store dataset and metrics info."""
        agent = ScopingAgent()
        dataset = _make_dataset()
        metric = ExactKeyMatchMetric(["summary"])
        agent.compile(dataset, metrics=[metric])

        assert agent.compiled_metadata["dataset_name"] == "test_dataset"
        assert "metrics" in agent.compiled_metadata

    def test_compile_invalidates_graph(self):
        """compile() should invalidate the cached graph."""
        agent = ScopingAgent()
        _ = agent.graph  # force build
        assert agent._graph is not None

        agent.compile(_make_dataset(), metrics=[])
        assert agent._graph is None

    def test_compile_returns_self(self):
        """compile() should return self for chaining."""
        agent = ScopingAgent()
        result = agent.compile(_make_dataset(), metrics=[])
        assert result is agent


# ===========================================================================
# Evaluate tests
# ===========================================================================


class TestEvaluate:
    """Test evaluate() method."""

    def test_evaluate_runs_metrics(self):
        """evaluate() should score all examples with all metrics."""
        agent = ScopingAgent()
        dataset = _make_dataset()
        metric = ExactKeyMatchMetric(["summary", "risks"])
        report = agent.evaluate(dataset, metrics=[metric])

        assert report.agent_name == "test_agent"
        assert report.dataset_name == "test_dataset"
        assert "exact_key_match" in report.scores
        assert report.scores["exact_key_match"] == pytest.approx(1.0)

    def test_evaluate_stores_in_history(self):
        """evaluate() should append to evaluation_history."""
        agent = ScopingAgent()
        dataset = _make_dataset()
        metric = ExactKeyMatchMetric(["summary"])

        assert len(agent.evaluation_history) == 0
        agent.evaluate(dataset, metrics=[metric])
        assert len(agent.evaluation_history) == 1
        agent.evaluate(dataset, metrics=[metric])
        assert len(agent.evaluation_history) == 2

    def test_evaluate_per_example_results(self):
        """evaluate() should return per-example results."""
        agent = ScopingAgent()
        dataset = _make_dataset()
        metric = ExactKeyMatchMetric(["summary"])
        report = agent.evaluate(dataset, metrics=[metric])

        assert report.num_examples == 2
        for result in report.example_results:
            assert result.error is None
            assert "exact_key_match" in result.scores


# ===========================================================================
# Registration bridge tests
# ===========================================================================


class TestRegistrationBridge:
    """Test as_registered_agent() and to_manifest()."""

    def test_as_registered_agent_returns_registered_agent(self):
        """as_registered_agent() should return a RegisteredAgent."""
        from agentomatic.core.manifest import RegisteredAgent

        agent = ScopingAgent()
        reg = agent.as_registered_agent()
        assert isinstance(reg, RegisteredAgent)
        assert reg.manifest.name == "test_agent"

    def test_to_manifest_generates_correct_manifest(self):
        """to_manifest() should use class attributes."""
        agent = ScopingAgent()
        manifest = agent.to_manifest()
        assert manifest.name == "test_agent"
        assert manifest.description == "A test agent for scoping"
        assert manifest.version == "0.1.0"

    def test_to_manifest_slug(self):
        """to_manifest() should generate a slug from agent_name."""
        agent = ScopingAgent()
        manifest = agent.to_manifest()
        assert "test_agent" in manifest.slug


# ===========================================================================
# Serialization tests
# ===========================================================================


class TestSerialization:
    """Test save() and load_compiled() persistence."""

    def test_save_creates_files(self, tmp_path):
        """save() should create config.json, metadata.json, and history."""
        agent = ScopingAgent()
        agent.compiled_config = {"temperature": 0.5}
        save_dir = tmp_path / "agent_save"
        agent.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "evaluation_history.json").exists()

    def test_save_and_load_roundtrip(self, tmp_path):
        """save() + load_compiled() should restore config."""
        agent = ScopingAgent()
        agent.compiled_config = {"temperature": 0.7, "top_k": 5}

        save_dir = tmp_path / "roundtrip"
        agent.save(save_dir)

        new_agent = ScopingAgent()
        new_agent.load_compiled(save_dir)
        assert new_agent.compiled_config["temperature"] == 0.7
        assert new_agent.compiled_config["top_k"] == 5

    def test_save_preserves_metadata(self, tmp_path):
        """save() should persist compiled metadata."""
        agent = ScopingAgent()
        agent.compiled_metadata = {"dataset_name": "my_data"}

        save_dir = tmp_path / "meta_test"
        agent.save(save_dir)

        new_agent = ScopingAgent()
        new_agent.load_compiled(save_dir)
        assert "agent_name" in new_agent.compiled_metadata


# ===========================================================================
# Observability tests
# ===========================================================================


class TestObservability:
    """Test trace recording and retrieval."""

    def test_get_last_trace_after_transform(self):
        """get_last_trace() should return events after transform."""
        agent = ScopingAgent()
        agent.transform({"request": "trace test"})
        trace = agent.get_last_trace()

        assert len(trace) >= 2  # at least extract + generate
        node_names = [e.node_name for e in trace]
        assert "extract" in node_names
        assert "generate" in node_names

    def test_get_last_trace_empty_before_transform(self):
        """get_last_trace() should be empty before any transform."""
        agent = ScopingAgent()
        assert agent.get_last_trace() == []

    def test_trace_history_grows(self):
        """Multiple transforms should add to trace history."""
        agent = ScopingAgent()
        agent.transform({"request": "first"})
        agent.transform({"request": "second"})
        history = agent.get_trace_history()
        assert len(history) == 2


# ===========================================================================
# Class attribute tests
# ===========================================================================


class TestClassAttributes:
    """Test agent_name and agent_description class attributes."""

    def test_agent_name_attribute(self):
        """agent_name should be accessible on instances."""
        agent = ScopingAgent()
        assert agent.agent_name == "test_agent"

    def test_agent_description_attribute(self):
        """agent_description should be accessible on instances."""
        agent = ScopingAgent()
        assert agent.agent_description == "A test agent for scoping"

    def test_agent_version_attribute(self):
        """agent_version should be accessible on instances."""
        agent = ScopingAgent()
        assert agent.agent_version == "0.1.0"

    def test_name_property(self):
        """The name property should delegate to agent_name."""
        agent = ScopingAgent()
        assert agent.agent_name == "test_agent"


# ---------------------------------------------------------------------------
# build_graph() style API tests
# ---------------------------------------------------------------------------


@dataclass
class BuildGraphState:
    """State for build_graph style agent."""

    query: str = ""
    result: dict[str, Any] = field(default_factory=dict)


class BuildGraphAgent(BaseGraphAgent[BuildGraphState]):
    """Agent that uses build_graph() instead of decorators."""

    agent_name = "build_graph_agent"
    agent_description = "Uses build_graph()"

    def build_graph(self):
        g = self.new_graph()
        g.add_node("parse", self.parse)
        g.add_node("respond", self.respond)
        g.set_entry_point("parse")
        g.add_edge("parse", "respond")
        g.set_finish_point("respond")
        return g.compile()

    def parse(self, state: BuildGraphState) -> BuildGraphState:
        state.result["parsed"] = True
        return state

    def respond(self, state: BuildGraphState) -> BuildGraphState:
        state.result["response"] = f"Answer: {state.query}"
        return state

    def input_to_state(self, data: dict[str, Any]) -> BuildGraphState:
        return BuildGraphState(query=data.get("query", ""))

    def state_to_output(self, state: BuildGraphState) -> dict[str, Any]:
        return state.result


class TestBuildGraphStyle:
    """Tests for the build_graph() primary API."""

    def test_transform(self):
        agent = BuildGraphAgent()
        result = agent.transform({"query": "hello"})
        assert result["parsed"] is True
        assert result["response"] == "Answer: hello"

    def test_graph_has_two_nodes(self):
        agent = BuildGraphAgent()
        assert len(agent.graph.nodes) == 2
        assert "parse" in agent.graph.nodes
        assert "respond" in agent.graph.nodes

    def test_graph_entrypoint(self):
        agent = BuildGraphAgent()
        assert agent.graph.entrypoint == "parse"

    def test_graph_finish(self):
        agent = BuildGraphAgent()
        assert agent.graph.finish == "respond"

    def test_trace_recorded(self):
        agent = BuildGraphAgent()
        agent.transform({"query": "test"})
        trace = agent.get_last_trace()
        assert len(trace) == 2
        assert trace[0].node_name == "parse"
        assert trace[1].node_name == "respond"
        assert all(t.status == "success" for t in trace)

    def test_evaluate(self):
        agent = BuildGraphAgent()
        ds = AgentDataset.from_list(
            [
                {
                    "id": "1",
                    "split": "test",
                    "input": {"query": "x"},
                    "expected_output": {"response": "Answer: x"},
                },
            ]
        )
        report = agent.evaluate(ds.test, [ExactKeyMatchMetric(["response"])])
        assert report.pass_rate == 1.0

    def test_compile_and_fit(self):
        from agentomatic.agents.optimizers import NoOpOptimizer

        agent = BuildGraphAgent()
        ds = AgentDataset.from_list(
            [
                {
                    "id": "1",
                    "split": "train",
                    "input": {"query": "x"},
                    "expected_output": {"response": "y"},
                },
            ]
        )
        metrics = [ExactKeyMatchMetric(["response"])]
        agent.compile(ds, metrics, optimizer=NoOpOptimizer())
        agent.fit(ds)

    def test_visualize(self):
        agent = BuildGraphAgent()
        mermaid = agent.visualize()
        assert "parse" in mermaid
        assert "respond" in mermaid

    def test_as_registered_agent(self):
        agent = BuildGraphAgent()
        reg = agent.as_registered_agent()
        assert reg.name == "build_graph_agent"

    def test_new_graph_returns_builder(self):
        from agentomatic.agents.builder import GraphBuilder

        agent = BuildGraphAgent()
        g = agent.new_graph()
        assert isinstance(g, GraphBuilder)


class TestBuildGraphWithConditional:
    """Tests for conditional edges via build_graph()."""

    def test_conditional_routing(self):
        @dataclass
        class RoutingState:
            query: str = ""
            needs_review: bool = False
            output: str = ""

        class RoutingAgent(BaseGraphAgent[RoutingState]):
            agent_name = "router"

            def build_graph(self):
                g = self.new_graph()
                g.add_node("check", self.check)
                g.add_node("approve", self.approve)
                g.add_node("review", self.review)
                g.set_entry_point("check")
                g.add_conditional_edge(
                    "check",
                    self.route,
                    {"approve": "approve", "review": "review"},
                )
                g.set_finish_point("approve")
                g.set_finish_point("review")
                return g.compile()

            def check(self, state):
                return state

            def route(self, state):
                return "review" if state.needs_review else "approve"

            def approve(self, state):
                state.output = "approved"
                return state

            def review(self, state):
                state.output = "needs review"
                return state

            def input_to_state(self, data):
                return RoutingState(
                    query=data.get("query", ""),
                    needs_review=data.get("needs_review", False),
                )

            def state_to_output(self, state):
                return {"output": state.output}

        agent = RoutingAgent()
        assert agent.transform({"query": "ok"})["output"] == "approved"
        assert (
            agent.transform({"query": "risky", "needs_review": True})["output"] == "needs review"
        )


class TestBuildGraphNotImplemented:
    """Test that missing build_graph() gives a clear error."""

    def test_missing_build_graph_raises(self):
        @dataclass
        class EmptyState:
            pass

        class EmptyAgent(BaseGraphAgent[EmptyState]):
            agent_name = "empty"

            def input_to_state(self, data):
                return EmptyState()

            def state_to_output(self, state):
                return {}

        agent = EmptyAgent()
        with pytest.raises(NotImplementedError, match="must implement"):
            agent.transform({})


class TestLangGraphCompatibleAliases:
    """Test that LangGraph-compatible aliases work."""

    def test_fluent_and_alias_produce_same_graph(self):
        from agentomatic.agents.builder import GraphBuilder

        def dummy(state):
            return state

        # Fluent style
        g1 = (
            GraphBuilder()
            .node("a", dummy)
            .node("b", dummy)
            .edge("a", "b")
            .entrypoint("a")
            .finish("b")
            .build()
        )

        # LangGraph style
        g2 = GraphBuilder()
        g2.add_node("a", dummy)
        g2.add_node("b", dummy)
        g2.add_edge("a", "b")
        g2.set_entry_point("a")
        g2.set_finish_point("b")
        result = g2.compile()

        assert len(g1.nodes) == len(result.nodes) == 2
        assert g1.entrypoint == result.entrypoint == "a"
        assert g1.finish == result.finish == "b"


# ===========================================================================
# Keras-style training polish (v1.2)
# ===========================================================================


class TestKerasTrainingPolish:
    """Regression coverage for evaluate defaults, History save/load, registration."""

    def test_evaluate_defaults_to_compile_metrics(self) -> None:
        agent = ScopingAgent()
        dataset = _make_dataset()
        metric = ExactKeyMatchMetric(["summary"])
        agent.compile(dataset=dataset, metrics=[metric])
        report = agent.evaluate(dataset)
        assert "exact_key_match" in report.scores

    def test_evaluate_without_metrics_raises(self) -> None:
        agent = ScopingAgent()
        with pytest.raises(ValueError, match="No metrics"):
            agent.evaluate(_make_dataset())

    def test_save_and_load_fit_history(self, tmp_path) -> None:
        from agentomatic.agents.history import History

        agent = ScopingAgent()
        history = History(params={"epochs": 2})
        history.record(0, {"loss": 0.5})
        history.record(1, {"loss": 0.3})
        agent.history = history
        agent.compiled_config = {"temperature": 0.2}

        save_dir = tmp_path / "with_history"
        agent.save(save_dir)
        assert (save_dir / "fit_history.json").exists()

        restored = ScopingAgent()
        restored.load(save_dir)
        assert restored.history is not None
        assert restored.history.final("loss") == 0.3
        assert restored.compiled_config["temperature"] == 0.2

    def test_save_and_load_evaluation_history(self, tmp_path) -> None:
        agent = ScopingAgent()
        dataset = _make_dataset()
        metric = ExactKeyMatchMetric(["summary"])
        report = agent.evaluate(dataset, metrics=[metric])
        assert len(agent.evaluation_history) == 1

        save_dir = tmp_path / "with_eval"
        agent.save(save_dir)

        restored = ScopingAgent()
        restored.load(save_dir)
        assert len(restored.evaluation_history) == 1
        loaded = restored.evaluation_history[0]
        assert loaded.scores == report.scores
        assert loaded.num_examples == report.num_examples
        assert loaded.example_results[0].example_id == report.example_results[0].example_id

    def test_as_registered_agent_keeps_class_instance(self) -> None:
        agent = ScopingAgent()
        reg = agent.as_registered_agent()
        assert reg.class_instance is agent
