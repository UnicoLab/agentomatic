"""Tests for agentomatic core modules."""
from __future__ import annotations

import pytest
from pathlib import Path


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
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent
        import asyncio

        m = AgentManifest(name="empty", slug="empty-agent")
        agent = RegisteredAgent(manifest=m)
        result = asyncio.get_event_loop().run_until_complete(agent.health_check())
        assert result["status"] == "degraded"
        assert result["node_fn_ready"] is False

    def test_health_check_with_node_fn(self):
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent
        import asyncio

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
            REQUEST_COUNT,
            AGENT_DURATION,
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
        from agentomatic import __version__

        assert __version__ == "0.1.0"
