"""Tests for agentomatic.pipelines.flow module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentomatic.pipelines.flow import (
    _ATTR_LISTEN,
    _ATTR_ROUTER,
    _ATTR_START,
    AgentHandle,
    Flow,
    FlowResult,
    _resolve_trigger_name,
    listen,
    router,
    start,
)

# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestStartDecorator:
    """Tests for the @start() decorator."""

    def test_sets_attribute(self) -> None:
        @start()
        def my_method(self, data):  # noqa: ANN001, ANN202
            pass

        assert getattr(my_method, _ATTR_START) is True

    def test_preserves_function_name(self) -> None:
        @start()
        def plan(self, data):  # noqa: ANN001, ANN202
            pass

        assert plan.__name__ == "plan"

    def test_no_other_attrs(self) -> None:
        @start()
        def fn(self, data):  # noqa: ANN001, ANN202
            pass

        assert not hasattr(fn, _ATTR_LISTEN)
        assert not hasattr(fn, _ATTR_ROUTER)


class TestListenDecorator:
    """Tests for the @listen() decorator."""

    def test_with_string_trigger(self) -> None:
        @listen("deep_path")
        def deep(self, data):  # noqa: ANN001, ANN202
            pass

        assert getattr(deep, _ATTR_LISTEN) == "deep_path"

    def test_with_function_trigger(self) -> None:
        def plan():
            pass

        @listen(plan)
        def research(self, data):  # noqa: ANN001, ANN202
            pass

        assert getattr(research, _ATTR_LISTEN) is plan


class TestRouterDecorator:
    """Tests for the @router() decorator."""

    def test_sets_router_attr(self) -> None:
        def research():
            pass

        @router(research)
        def route(self, results):  # noqa: ANN001, ANN202
            pass

        assert getattr(route, _ATTR_ROUTER) is research

    def test_with_string_trigger(self) -> None:
        @router("research")
        def route(self, results):  # noqa: ANN001, ANN202
            pass

        assert getattr(route, _ATTR_ROUTER) == "research"


# ---------------------------------------------------------------------------
# _resolve_trigger_name tests
# ---------------------------------------------------------------------------


class TestResolveTriggerName:
    """Tests for _resolve_trigger_name helper."""

    def test_string_passthrough(self) -> None:
        assert _resolve_trigger_name("deep_path") == "deep_path"

    def test_function_uses_name(self) -> None:
        def plan():
            pass

        assert _resolve_trigger_name(plan) == "plan"

    def test_callable_without_name(self) -> None:
        obj = MagicMock(spec=[])  # no __name__
        result = _resolve_trigger_name(obj)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AgentHandle tests
# ---------------------------------------------------------------------------


class TestAgentHandle:
    """Tests for the AgentHandle class."""

    def test_repr(self) -> None:
        handle = AgentHandle("planner", None)
        assert repr(handle) == "AgentHandle('planner')"

    @pytest.mark.asyncio
    async def test_run_raises_without_registry(self) -> None:
        handle = AgentHandle("planner", None)
        with pytest.raises(ValueError, match="no AgentRegistry bound"):
            await handle.run({"query": "test"})

    @pytest.mark.asyncio
    async def test_run_raises_agent_not_found(self) -> None:
        registry = MagicMock()
        registry.get.return_value = None
        registry.list_names.return_value = ["other"]
        handle = AgentHandle("unknown", registry)
        with pytest.raises(ValueError, match="not found in registry"):
            await handle.run({"query": "test"})

    @pytest.mark.asyncio
    async def test_run_via_graph_fn(self) -> None:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"response": "ok"}
        agent = SimpleNamespace(
            graph_fn=lambda: mock_graph,
            node_fn=None,
        )
        registry = MagicMock()
        registry.get.return_value = agent
        handle = AgentHandle("planner", registry)
        result = await handle.run({"current_query": "test"})
        assert result == {"response": "ok"}

    @pytest.mark.asyncio
    async def test_run_via_node_fn(self) -> None:
        node_fn = AsyncMock(return_value={"response": "from_node"})
        agent = SimpleNamespace(graph_fn=None, node_fn=node_fn)
        registry = MagicMock()
        registry.get.return_value = agent
        handle = AgentHandle("planner", registry)
        result = await handle.run({"current_query": "hi"})
        assert result == {"response": "from_node"}

    @pytest.mark.asyncio
    async def test_run_raises_no_fn(self) -> None:
        agent = SimpleNamespace(graph_fn=None, node_fn=None)
        registry = MagicMock()
        registry.get.return_value = agent
        handle = AgentHandle("planner", registry)
        with pytest.raises(RuntimeError, match="neither graph_fn nor node_fn"):
            await handle.run({"query": "test"})

    def test_build_state_with_dict(self) -> None:
        state = AgentHandle._build_state({"query": "hello", "extra": 42})
        assert state["current_query"] == "hello"
        assert state["query"] == "hello"
        assert state["extra"] == 42

    def test_build_state_with_dict_existing_current_query(self) -> None:
        state = AgentHandle._build_state({"current_query": "already set"})
        assert state["current_query"] == "already set"

    def test_build_state_with_scalar(self) -> None:
        state = AgentHandle._build_state("hello world")
        assert state["current_query"] == "hello world"

    def test_build_state_with_none(self) -> None:
        state = AgentHandle._build_state(None)
        assert state["current_query"] == ""

    def test_normalise_output_dict(self) -> None:
        assert AgentHandle._normalise_output({"a": 1}) == {"a": 1}

    def test_normalise_output_non_dict(self) -> None:
        assert AgentHandle._normalise_output("text") == {"result": "text"}


# ---------------------------------------------------------------------------
# FlowResult tests
# ---------------------------------------------------------------------------


class TestFlowResult:
    """Tests for the FlowResult model."""

    def test_defaults(self) -> None:
        r = FlowResult()
        assert r.output == {}
        assert r.steps == {}
        assert r.duration_ms == 0.0
        assert r.status == "success"

    def test_custom_values(self) -> None:
        r = FlowResult(
            output={"key": "val"},
            steps={"step1": {"a": 1}},
            duration_ms=123.45,
            status="failed",
        )
        assert r.output == {"key": "val"}
        assert r.status == "failed"


# ---------------------------------------------------------------------------
# Flow tests
# ---------------------------------------------------------------------------


class TestFlowBasic:
    """Tests for the Flow base class methods."""

    def test_bind_registry(self) -> None:
        flow = Flow()
        assert flow._registry is None
        reg = MagicMock()
        flow.bind_registry(reg)
        assert flow._registry is reg

    def test_agent_returns_handle(self) -> None:
        flow = Flow()
        handle = flow.agent("planner")
        assert isinstance(handle, AgentHandle)
        assert handle.name == "planner"


class TestFlowIntrospection:
    """Tests for the Flow introspection logic."""

    def test_introspect_finds_start(self) -> None:
        class MyFlow(Flow):
            @start()
            async def entry(self, data: Any) -> dict:
                return {"done": True}

        flow = MyFlow()
        starts, listeners, routers = flow._introspect()
        assert len(starts) == 1
        assert starts[0].__name__ == "entry"
        assert listeners == {}
        assert routers == {}

    def test_introspect_finds_listeners(self) -> None:
        class MyFlow(Flow):
            @start()
            async def entry(self, data: Any) -> dict:
                return {}

            @listen(entry)
            async def step2(self, data: Any) -> dict:
                return {}

        flow = MyFlow()
        starts, listeners, routers = flow._introspect()
        assert len(starts) == 1
        assert "entry" in listeners
        assert len(listeners["entry"]) == 1
        assert listeners["entry"][0].__name__ == "step2"

    def test_introspect_no_start_raises(self) -> None:
        class BadFlow(Flow):
            async def orphan(self, data: Any) -> dict:
                return {}

        flow = BadFlow()
        with pytest.raises(RuntimeError, match="No @start"):
            flow._introspect()

    def test_introspect_finds_routers(self) -> None:
        class MyFlow(Flow):
            @start()
            async def entry(self, data: Any) -> dict:
                return {}

            @router(entry)
            def my_router(self, data: Any) -> str:
                return "path_a"

            @listen("path_a")
            async def path_a_handler(self, data: Any) -> dict:
                return {}

        flow = MyFlow()
        starts, listeners, routers = flow._introspect()
        assert "entry" in routers
        assert "path_a" in listeners


class TestFlowExecution:
    """Tests for the Flow.run() execution engine."""

    @pytest.mark.asyncio
    async def test_simple_linear_flow(self) -> None:
        class LinearFlow(Flow):
            @start()
            async def step1(self, data: Any) -> dict:
                return {"step1": True, "val": data.get("input", "")}

            @listen(step1)
            async def step2(self, data: Any) -> dict:
                return {"step2": True, "from_step1": data}

        flow = LinearFlow()
        result = await flow.run({"input": "hello"})
        assert result.status == "success"
        assert "step1" in result.steps
        assert "step2" in result.steps
        assert result.steps["step1"]["step1"] is True

    @pytest.mark.asyncio
    async def test_router_flow(self) -> None:
        class RouterFlow(Flow):
            @start()
            async def entry(self, data: Any) -> dict:
                return {"count": data.get("count", 0)}

            @router(entry)
            def decide(self, result: Any) -> str:
                if result.get("count", 0) > 2:
                    return "many"
                return "few"

            @listen("many")
            async def handle_many(self, data: Any) -> dict:
                return {"path": "many", "data": data}

            @listen("few")
            async def handle_few(self, data: Any) -> dict:
                return {"path": "few", "data": data}

        # Test "few" path
        flow = RouterFlow()
        result = await flow.run({"count": 1})
        assert result.status == "success"
        assert "handle_few" in result.steps
        assert "handle_many" not in result.steps
        assert result.steps["handle_few"]["path"] == "few"

        # Test "many" path
        flow2 = RouterFlow()
        result2 = await flow2.run({"count": 5})
        assert result2.status == "success"
        assert "handle_many" in result2.steps
        assert "handle_few" not in result2.steps

    @pytest.mark.asyncio
    async def test_sync_method_support(self) -> None:
        class SyncFlow(Flow):
            @start()
            def entry(self, data: Any) -> dict:
                return {"sync": True}

        flow = SyncFlow()
        result = await flow.run({"x": 1})
        assert result.status == "success"
        assert result.steps["entry"]["sync"] is True

    @pytest.mark.asyncio
    async def test_flow_error_handling(self) -> None:
        class BrokenFlow(Flow):
            @start()
            async def entry(self, data: Any) -> dict:
                raise ValueError("boom")

        flow = BrokenFlow()
        result = await flow.run({})
        assert result.status == "failed"
        assert "__error__" in result.steps

    @pytest.mark.asyncio
    async def test_duration_tracked(self) -> None:
        class SlowFlow(Flow):
            @start()
            async def entry(self, data: Any) -> dict:
                return {"done": True}

        flow = SlowFlow()
        result = await flow.run({})
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_chained_listeners(self) -> None:
        class ChainFlow(Flow):
            @start()
            async def a(self, data: Any) -> dict:
                return {"stage": "a"}

            @listen(a)
            async def b(self, data: Any) -> dict:
                return {"stage": "b"}

            @listen(b)
            async def c(self, data: Any) -> dict:
                return {"stage": "c"}

        flow = ChainFlow()
        result = await flow.run({})
        assert result.status == "success"
        assert result.steps["a"]["stage"] == "a"
        assert result.steps["b"]["stage"] == "b"
        assert result.steps["c"]["stage"] == "c"

    @pytest.mark.asyncio
    async def test_final_output_is_last_step(self) -> None:
        class FinalFlow(Flow):
            @start()
            async def first(self, data: Any) -> dict:
                return {"v": 1}

            @listen(first)
            async def second(self, data: Any) -> dict:
                return {"v": 2}

        flow = FinalFlow()
        result = await flow.run({})
        assert result.output == {"v": 2}


class TestFlowParallel:
    """Tests for Flow.parallel()."""

    @pytest.mark.asyncio
    async def test_parallel_all(self) -> None:
        h1 = AgentHandle("a", None)
        h2 = AgentHandle("b", None)
        r1 = {"response": "a"}
        r2 = {"response": "b"}

        with patch.object(AgentHandle, "run", side_effect=[r1, r2]):
            flow = Flow()
            results = await flow.parallel([h1, h2], input={"q": "test"}, strategy="all")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_parallel_first(self) -> None:
        h1 = AgentHandle("a", None)
        h2 = AgentHandle("b", None)
        r1 = {"response": "first!"}

        async def mock_run(data: Any) -> dict:
            return r1

        with patch.object(AgentHandle, "run", side_effect=mock_run):
            flow = Flow()
            results = await flow.parallel([h1, h2], input={"q": "test"}, strategy="first")
        assert len(results) == 1
