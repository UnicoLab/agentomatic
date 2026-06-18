"""Tests for agentomatic core modules."""

from __future__ import annotations

import pytest


class TestAgentManifest:
    """Test AgentManifest creation."""

    def test_basic_creation(self):
        from agentomatic.core.manifest import AgentManifest

        m = AgentManifest(name="test", slug="test-agent")
        assert m.name == "test"
        assert m.slug == "test-agent"
        assert m.version == "1.0.0"
        assert m.framework == "langgraph"
        assert m.is_subagent is True

    def test_frozen(self):
        from agentomatic.core.manifest import AgentManifest

        m = AgentManifest(name="test", slug="test-agent")
        with pytest.raises(AttributeError):
            m.name = "changed"  # type: ignore

    def test_full_creation(self):
        from agentomatic.core.manifest import AgentManifest

        m = AgentManifest(
            name="holidays",
            slug="hr-holidays-agent",
            description="Manages holiday requests",
            intent_keywords=["holiday", "vacation", "leave"],
            version="2.1.0",
            is_subagent=True,
            framework="langchain",
            metadata={"department": "hr"},
        )
        assert m.description == "Manages holiday requests"
        assert len(m.intent_keywords) == 3
        assert m.framework == "langchain"
        assert m.metadata["department"] == "hr"


class TestRegisteredAgent:
    """Test RegisteredAgent."""

    def test_health_check_no_callable(self):
        import asyncio

        from agentomatic.core.manifest import AgentManifest, RegisteredAgent

        m = AgentManifest(name="empty", slug="empty-agent")
        agent = RegisteredAgent(manifest=m)
        result = asyncio.get_event_loop().run_until_complete(agent.health_check())
        assert result["status"] == "degraded"
        assert result["node_fn_ready"] is False

    def test_health_check_with_node_fn(self):
        import asyncio

        from agentomatic.core.manifest import AgentManifest, RegisteredAgent

        async def dummy_fn(state):
            return state

        m = AgentManifest(name="test", slug="test-agent")
        agent = RegisteredAgent(manifest=m, node_fn=dummy_fn)
        result = asyncio.get_event_loop().run_until_complete(agent.health_check())
        assert result["status"] == "healthy"
        assert result["node_fn_ready"] is True


class TestAgentRegistry:
    """Test AgentRegistry."""

    def test_empty_registry(self):
        from agentomatic.core.registry import AgentRegistry

        reg = AgentRegistry()
        assert reg.count == 0
        assert reg.all() == {}
        assert reg.get("nonexistent") is None

    def test_manual_registration(self):
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent
        from agentomatic.core.registry import AgentRegistry

        reg = AgentRegistry()
        m = AgentManifest(name="test", slug="test-agent")
        agent = RegisteredAgent(manifest=m)
        reg._agents["test"] = agent

        assert reg.count == 1
        assert reg.get("test") is not None
        assert reg.get("test").name == "test"

    def test_list_names(self):
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent
        from agentomatic.core.registry import AgentRegistry

        reg = AgentRegistry()
        for name in ["alpha", "beta", "gamma"]:
            m = AgentManifest(name=name, slug=f"{name}-agent")
            reg._agents[name] = RegisteredAgent(manifest=m)

        names = reg.list_names()
        assert sorted(names) == ["alpha", "beta", "gamma"]


class TestBaseAgentState:
    """Test BaseAgentState."""

    def test_state_creation(self):
        from agentomatic.core.state import BaseAgentState

        state: BaseAgentState = {
            "current_query": "hello",
            "user_id": "user-1",
            "response": "",
            "messages": [],
        }
        assert state["current_query"] == "hello"


class TestAPIResponse:
    """Test APIResponse model."""

    def test_default_response(self):
        from agentomatic.protocols.decorators import APIResponse

        resp = APIResponse(data={"key": "value"}, message="OK")
        assert resp.success is True
        assert resp.data == {"key": "value"}
        assert resp.message == "OK"
        assert resp.timestamp is not None

    def test_error_response(self):
        from agentomatic.protocols.decorators import APIResponse

        resp = APIResponse(success=False, error="Something went wrong")
        assert resp.success is False
        assert resp.error == "Something went wrong"


class TestPlatformSettings:
    """Test PlatformSettings."""

    def test_default_settings(self):
        from agentomatic.config.settings import PlatformSettings

        settings = PlatformSettings()
        assert settings.app_name == "Agentomatic Platform"
        assert settings.log_level == "INFO"
        assert settings.llm.provider == "ollama"
        assert settings.features.enable_streaming is True
        assert settings.features.max_concurrent_agents == 10


class TestPromptManager:
    """Test PromptManager."""

    def test_empty_manager(self):
        from agentomatic.prompts.manager import PromptManager

        pm = PromptManager("test")
        assert pm.list_versions() == []
        assert pm.get_prompt("v1") is None

    def test_load_and_get(self, tmp_path):
        import json

        from agentomatic.prompts.manager import PromptManager

        prompts = {
            "v1": {"system": "You are a helpful assistant."},
            "v2": {"system": "You are an expert."},
        }
        f = tmp_path / "prompts.json"
        f.write_text(json.dumps(prompts))

        pm = PromptManager("test")
        pm.load_from_file(f)
        assert pm.list_versions() == ["v1", "v2"]
        assert pm.get_prompt("v1", "system") == "You are a helpful assistant."


class TestCircuitBreaker:
    """Test CircuitBreaker."""

    def test_initial_state(self):
        from agentomatic.observability.concurrency import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        import asyncio

        from agentomatic.observability.concurrency import (
            CircuitBreaker,
            CircuitBreakerOpen,
            CircuitState,
        )

        cb = CircuitBreaker("test", failure_threshold=2)

        async def fail_twice():
            for _ in range(2):
                try:
                    async with cb():
                        raise ValueError("fail")
                except ValueError:
                    pass
            assert cb.state == CircuitState.OPEN
            with pytest.raises(CircuitBreakerOpen):
                async with cb():
                    pass

        asyncio.get_event_loop().run_until_complete(fail_twice())


class TestMemoryStore:
    """Test MemoryStore."""

    def test_thread_lifecycle(self):
        import asyncio

        from agentomatic.storage.memory import MemoryStore

        store = MemoryStore()

        async def run():
            thread = await store.create_thread("t1", "user1", "agent1")
            assert thread["id"] == "t1"

            await store.add_message("t1", "user", "Hello!")
            await store.add_message("t1", "assistant", "Hi there!")

            messages = await store.get_messages("t1")
            assert len(messages) == 2

            threads = await store.list_threads(agent_name="agent1")
            assert len(threads) == 1

        asyncio.get_event_loop().run_until_complete(run())


class TestMetrics:
    """Test metrics module."""

    def test_dummy_metrics(self):
        """Metrics should work even without prometheus_client."""
        from agentomatic.observability.metrics import (
            ACTIVE_REQUESTS,
        )

        # These should not raise regardless of prometheus availability
        ACTIVE_REQUESTS.inc()
        ACTIVE_REQUESTS.dec()


class TestLLMFactory:
    """Test LLM factory."""

    def test_dummy_provider(self):
        from agentomatic.providers.llm import get_llm, reset_llm

        reset_llm()
        llm = get_llm("dummy")
        assert llm is not None
        reset_llm()

    def test_singleton(self):
        from agentomatic.providers.llm import get_llm, reset_llm

        reset_llm()
        llm1 = get_llm("dummy")
        llm2 = get_llm("dummy")
        assert llm1 is llm2
        reset_llm()


class TestVersion:
    """Test version."""

    def test_version_exists(self):
        import re

        from agentomatic import __version__

        assert __version__
        assert re.match(r"^\d+\.\d+\.\d+", __version__)


class TestAgentRegistryDiscovery:
    """Test AgentRegistry folder discovery and dynamic enhancements."""

    def test_discover_agents(self, tmp_path):
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            # Create agent_a (minimal)
            agent_a_dir = tmp_path / "agent_a"
            agent_a_dir.mkdir()
            (agent_a_dir / "__init__.py").write_text("""
from agentomatic.core.manifest import AgentManifest
manifest = AgentManifest(name="agent_a", slug="agent-a", intent_keywords=["query"], is_subagent=False)
async def node_fn(state):
    return {"response": "a"}
""")

            # Create agent_b (with graph, router, config, prompts)
            agent_b_dir = tmp_path / "agent_b"
            agent_b_dir.mkdir()
            (agent_b_dir / "__init__.py").write_text("""
from agentomatic.core.manifest import AgentManifest
manifest = AgentManifest(name="agent_b", slug="agent-b", is_subagent=True)
""")
            (agent_b_dir / "graph.py").write_text("""
def get_graph():
    return "mock_graph"
""")
            (agent_b_dir / "api.py").write_text("""
from fastapi import APIRouter
router = APIRouter()
""")
            (agent_b_dir / "config.py").write_text("""
from pydantic import BaseModel
class AgentBConfig(BaseModel):
    temperature: float = 0.5
""")
            (agent_b_dir / "prompts.json").write_text('{"v1": {"system": "hello"}}')

            from agentomatic.core.registry import AgentRegistry

            reg = AgentRegistry()
            reg.discover(tmp_path)

            assert reg.count == 2
            agent_a = reg.get("agent_a")
            agent_b = reg.get("agent_b")

            assert agent_a is not None
            assert agent_a.manifest.name == "agent_a"
            assert agent_a.node_fn is not None
            assert agent_a.graph_fn is None

            assert agent_b is not None
            assert agent_b.manifest.is_subagent is True
            assert agent_b.graph_fn() == "mock_graph"
            assert agent_b.router is not None
            assert agent_b.config is not None
            assert agent_b.prompt_manager is not None
            assert agent_b.prompt_manager.get_prompt("v1", "system") == "hello"

            assert "agent_b" in reg.get_subagents()
            assert "agent_a" not in reg.get_subagents()

            routers = reg.get_agent_routers()
            assert len(routers) == 1
            assert routers[0][0] == "agent_b"

            keywords = reg.get_intent_keywords()
            assert keywords["agent_a"] == ["query"]

        finally:
            sys.path.remove(str(tmp_path))
            for m in ["agent_a", "agent_b", "agent_b.graph", "agent_b.api", "agent_b.config"]:
                sys.modules.pop(m, None)


class TestStateReducers:
    """Test custom state field reducers."""

    def test_merge_dicts(self):
        from agentomatic.core.state import _merge_dicts

        assert _merge_dicts(None, {"a": 1}) == {"a": 1}
        assert _merge_dicts({"a": 1}, None) == {"a": 1}
        assert _merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        assert _merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}

    def test_last_value(self):
        from agentomatic.core.state import _last_value

        assert _last_value(1, 2) == 2
        assert _last_value(1, None) == 1

    def test_add_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from agentomatic.core.state import add_messages

        msg1 = HumanMessage(content="hello")
        msg2 = AIMessage(content="hi")
        res = add_messages([msg1], [msg2])
        assert len(res) == 2


class TestPromptManagerExtras:
    """Test PromptManager edge cases and reload functionality."""

    def test_instantiation_with_file(self, tmp_path):
        from agentomatic.prompts.manager import PromptManager

        f = tmp_path / "prompts.json"
        f.write_text('{"v1": {"system": "system prompt"}}')

        pm = PromptManager("test", prompts_file=f)
        assert pm.get_prompt("v1", "system") == "system prompt"

    def test_format_prompt_key_error(self):
        from agentomatic.prompts.manager import PromptManager

        pm = PromptManager("test")
        pm._prompts = {"v1": {"user_template": "Hello {name}!"}}

        # Test formatting fallback when key is missing (should return template as-is or format error fallback)
        res = pm.format_prompt("v1", "user_template", age=25)
        assert res == "Hello {name}!"

    def test_format_prompt_missing(self):
        from agentomatic.prompts.manager import PromptManager

        pm = PromptManager("test")
        assert pm.format_prompt("v1", "nonexistent") is None

    def test_reload(self, tmp_path):
        from agentomatic.prompts.manager import PromptManager

        f = tmp_path / "prompts.json"
        f.write_text('{"v1": {"system": "old"}}')

        pm = PromptManager("test", prompts_file=f)
        assert pm.get_prompt("v1", "system") == "old"

        f.write_text('{"v1": {"system": "new"}}')
        pm.reload(f)
        assert pm.get_prompt("v1", "system") == "new"

    def test_invalid_json_handling(self, tmp_path):
        from agentomatic.prompts.manager import PromptManager

        f = tmp_path / "invalid.json"
        f.write_text("invalid json data")

        pm = PromptManager("test")
        pm.load_from_file(f)  # Should log error and not crash
        assert pm.list_versions() == []


class TestHealthCheckErrors:
    """Test health check error paths."""

    @pytest.mark.asyncio
    async def test_health_check_graph_error(self):
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent

        m = AgentManifest(name="test", slug="test-agent")

        def fail_graph():
            raise ValueError("graph error")

        agent = RegisteredAgent(manifest=m, graph_fn=fail_graph)
        result = await agent.health_check()
        assert result["status"] == "degraded"
        assert result["graph_ready"] is False
        assert result["graph_error"] == "graph error"

    @pytest.mark.asyncio
    async def test_health_check_with_prompts(self, tmp_path):
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent
        from agentomatic.prompts.manager import PromptManager

        m = AgentManifest(name="test", slug="test-agent")
        pm = PromptManager("test")
        pm._prompts = {"v1": {"system": "test"}}

        agent = RegisteredAgent(manifest=m, prompt_manager=pm)
        result = await agent.health_check()
        assert result["prompt_versions"] == ["v1"]
