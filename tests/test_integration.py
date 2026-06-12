"""Integration tests for agentomatic platform with FastAPI TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform


@pytest.fixture
def platform():
    """Create a platform with programmatically registered agents."""
    p = AgentPlatform(
        agents_dir="/tmp/agentomatic_test_agents_empty",
        title="Test Platform",
        version="0.0.1",
    )

    # Register a test agent programmatically
    async def echo_fn(state):
        return {
            "response": f"Echo: {state.get('current_query', '')}",
            "agent_type": "test-echo",
            "suggestions": ["try again"],
        }

    p.register_agent(
        manifest=AgentManifest(
            name="echo",
            slug="test-echo",
            description="Echo agent for testing",
            intent_keywords=["echo", "test"],
        ),
        node_fn=echo_fn,
    )

    # Register a second agent (no callable — degraded)
    p.register_agent(
        manifest=AgentManifest(
            name="empty",
            slug="test-empty",
            description="Empty agent with no callable",
            is_subagent=False,
        ),
    )

    return p


@pytest.fixture
def app(platform):
    """Build the FastAPI app."""
    return platform.build()


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestPlatformEndpoints:
    """Test platform-level endpoints."""

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Platform"
        assert data["version"] == "0.0.1"
        assert "docs" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_count"] >= 1
        assert "echo" in data["agents"]

    def test_readiness(self, client):
        resp = client.get("/readiness")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_list_agents(self, client):
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "echo" in data["agents"]
        assert data["agents"]["echo"]["slug"] == "test-echo"

    def test_a2a_discovery(self, client):
        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "Test Platform"
        assert "echo" in data["agents"]


class TestAgentEndpoints:
    """Test auto-generated agent endpoints."""

    def test_invoke(self, client):
        resp = client.post(
            "/api/v1/echo/invoke",
            json={"query": "Hello world!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Echo: Hello world!" in data["response"]
        assert data["agent_type"] == "test-echo"
        assert data["duration_ms"] > 0

    def test_invoke_with_full_params(self, client):
        resp = client.post(
            "/api/v1/echo/invoke",
            json={
                "query": "Full params test",
                "user_id": "user-42",
                "thread_id": "thread-abc",
                "prompt_version": "v2",
                "temperature": 0.5,
                "max_tokens": 100,
                "context": {"key": "value"},
                "metadata": {"source": "test"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Full params test" in data["response"]

    def test_chat(self, client):
        resp = client.post(
            "/api/v1/echo/chat",
            json={"content": "Chat message", "user_id": "user-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] is not None
        assert "Chat message" in data["response"]

    def test_health(self, client):
        resp = client.get("/api/v1/echo/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["node_fn_ready"] is True

    def test_config(self, client):
        resp = client.get("/api/v1/echo/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "echo"

    def test_prompts(self, client):
        resp = client.get("/api/v1/echo/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "echo"

    def test_card(self, client):
        resp = client.get("/api/v1/echo/card")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-echo"
        assert data["capabilities"]["streaming"] is True
        assert data["capabilities"]["a2a"] is True

    def test_a2a_task(self, client):
        resp = client.post(
            "/api/v1/echo/a2a/tasks",
            json={"message": {"content": "A2A task"}, "metadata": {"from": "agent-x"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "A2A task" in data["result"]

    def test_stream(self, client):
        resp = client.post(
            "/api/v1/echo/invoke/stream",
            json={"query": "Stream test"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/event-stream")

    def test_threads_empty(self, client):
        resp = client.get("/api/v1/echo/threads")
        assert resp.status_code == 200

    def test_nonexistent_agent_404(self, client):
        resp = client.post(
            "/api/v1/nonexistent/invoke",
            json={"query": "test"},
        )
        assert resp.status_code == 404


class TestProgrammaticRegistration:
    """Test programmatic agent registration."""

    def test_register_and_invoke(self):
        p = AgentPlatform(agents_dir="/tmp/agentomatic_empty")

        async def adder(state):
            a = state.get("metadata", {}).get("a", 0)
            b = state.get("metadata", {}).get("b", 0)
            return {"response": str(a + b), "agent_type": "calculator"}

        p.register_agent(
            manifest=AgentManifest(name="calc", slug="calculator"),
            node_fn=adder,
        )

        app = p.build()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/calc/invoke",
            json={"query": "add", "metadata": {"a": 3, "b": 7}},
        )
        assert resp.status_code == 200
        assert resp.json()["response"] == "10"


class TestSubagentFiltering:
    """Test subagent vs non-subagent behavior."""

    def test_non_subagent_no_router(self, platform, app):
        """Non-subagent agents should not get auto-routers."""
        empty_agent = platform.registry.get("empty")
        assert empty_agent is not None
        assert empty_agent.manifest.is_subagent is False


class TestPlatformFactory:
    """Test platform factory methods."""

    def test_from_folder(self):
        p = AgentPlatform.from_folder("/tmp/agentomatic_test_empty")
        assert p.agents_dir.name == "agentomatic_test_empty"

    def test_custom_settings(self):
        p = AgentPlatform(
            agents_dir="/tmp/test",
            title="Custom",
            version="3.0.0",
            api_prefix="/custom/api",
            log_level="DEBUG",
        )
        assert p.title == "Custom"
        assert p.version == "3.0.0"
        assert p.api_prefix == "/custom/api"
