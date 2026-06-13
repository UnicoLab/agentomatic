"""Comprehensive integration tests for agentomatic platform.

Tests the full stack: platform → router → agent → storage → middleware.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.storage.memory import MemoryStore

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def platform(store):
    p = AgentPlatform(
        agents_dir="/tmp/agentomatic_test_empty",
        title="Test Platform",
        version="0.0.1",
        store=store,
    )

    # --- Echo agent (healthy, has node_fn) ---
    async def echo_fn(state):
        return {
            "response": f"Echo: {state.get('current_query', '')}",
            "agent_type": "test-echo",
            "suggestions": ["try again"],
            "steps_taken": ["echo_step"],
            "citations": [{"source": "test"}],
            "metadata": {"echo": True},
        }

    p.register_agent(
        manifest=AgentManifest(
            name="echo",
            slug="test-echo",
            description="Echo agent for testing",
            intent_keywords=["echo", "test"],
            version="1.2.3",
            metadata={"tier": "free"},
        ),
        node_fn=echo_fn,
    )

    # --- Failing agent (raises) ---
    async def fail_fn(state):
        raise ValueError("Intentional failure for testing")

    p.register_agent(
        manifest=AgentManifest(
            name="failer",
            slug="test-failer",
            description="Agent that always fails",
        ),
        node_fn=fail_fn,
    )

    # --- Empty agent (no callable, degraded) ---
    p.register_agent(
        manifest=AgentManifest(
            name="empty",
            slug="test-empty",
            description="Empty agent, no callable",
            is_subagent=False,
        ),
    )

    return p


@pytest.fixture
def app(platform):
    return platform.build()


@pytest.fixture
def client(app):
    return TestClient(app)


# =========================================================================
# Platform-level endpoints
# =========================================================================


class TestPlatformRoot:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Platform"
        assert data["version"] == "0.0.1"
        assert "docs" in data
        assert "health" in data
        assert "a2a" in data

    def test_docs(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


class TestPlatformHealth:
    def test_health_aggregated(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_count"] >= 2
        assert "echo" in data["agents"]
        assert data["agents"]["echo"]["status"] == "healthy"

    def test_readiness(self, client):
        resp = client.get("/readiness")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


class TestAgentListing:
    def test_list_agents(self, client):
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert "echo" in agents
        assert agents["echo"]["slug"] == "test-echo"
        assert agents["echo"]["version"] == "1.2.3"


class TestA2ADiscovery:
    def test_well_known(self, client):
        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "Test Platform"
        assert "echo" in data["agents"]
        echo = data["agents"]["echo"]
        assert echo["name"] == "test-echo"
        assert "invoke" in echo["endpoints"]


# =========================================================================
# Agent invoke endpoints
# =========================================================================


class TestInvoke:
    def test_basic_invoke(self, client):
        resp = client.post("/api/v1/echo/invoke", json={"query": "Hello!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "Echo: Hello!" in data["response"]
        assert data["agent_type"] == "test-echo"
        assert data["duration_ms"] > 0
        assert data["suggestions"] == ["try again"]
        assert data["steps_taken"] == ["echo_step"]
        assert data["citations"] == [{"source": "test"}]

    def test_invoke_with_all_params(self, client):
        resp = client.post(
            "/api/v1/echo/invoke",
            json={
                "query": "Full params",
                "user_id": "user-42",
                "thread_id": "thread-xyz",
                "prompt_version": "v2",
                "temperature": 0.7,
                "max_tokens": 256,
                "context": {"role": "admin"},
                "metadata": {"source": "test"},
            },
        )
        assert resp.status_code == 200
        assert "Full params" in resp.json()["response"]

    def test_invoke_empty_query(self, client):
        resp = client.post("/api/v1/echo/invoke", json={"query": ""})
        assert resp.status_code == 200

    def test_invoke_nonexistent_404(self, client):
        resp = client.post("/api/v1/nonexistent/invoke", json={"query": "x"})
        assert resp.status_code == 404

    def test_invoke_failure_500(self, client):
        resp = client.post("/api/v1/failer/invoke", json={"query": "fail"})
        assert resp.status_code == 500


class TestChat:
    def test_basic_chat(self, client):
        resp = client.post("/api/v1/echo/chat", json={"content": "Hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] is not None
        assert "Hi" in data["response"]
        assert "duration_ms" in data

    def test_chat_with_thread(self, client):
        resp = client.post(
            "/api/v1/echo/chat",
            json={
                "content": "Continue",
                "thread_id": "existing-thread",
                "user_id": "user-1",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["thread_id"] == "existing-thread"

    def test_chat_failure(self, client):
        resp = client.post("/api/v1/failer/chat", json={"content": "fail"})
        assert resp.status_code == 500


class TestStream:
    def test_stream_response(self, client):
        resp = client.post("/api/v1/echo/invoke/stream", json={"query": "Stream"})
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct
        assert resp.headers.get("X-Agent") == "echo"
        body = resp.text
        assert "data:" in body
        assert "[DONE]" in body


# =========================================================================
# Per-agent metadata endpoints
# =========================================================================


class TestAgentMetadata:
    def test_health(self, client):
        resp = client.get("/api/v1/echo/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["node_fn_ready"] is True
        assert data["version"] == "1.2.3"

    def test_config_empty(self, client):
        resp = client.get("/api/v1/echo/config")
        assert resp.status_code == 200
        assert resp.json()["agent"] == "echo"

    def test_prompts_empty(self, client):
        resp = client.get("/api/v1/echo/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "echo"
        assert data["active"] == "v1"

    def test_card(self, client):
        resp = client.get("/api/v1/echo/card")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-echo"
        assert data["version"] == "1.2.3"
        assert data["capabilities"]["streaming"] is True
        assert data["capabilities"]["a2a"] is True
        assert data["metadata"]["tier"] == "free"


# =========================================================================
# A2A tasks
# =========================================================================


class TestA2ATasks:
    def test_submit_task(self, client):
        resp = client.post(
            "/api/v1/echo/a2a/tasks",
            json={
                "message": {"content": "A2A query"},
                "metadata": {"from": "agent-x"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "A2A query" in data["result"]
        assert data["task_id"].startswith("task_")

    def test_get_task_status(self, client):
        resp = client.get("/api/v1/echo/a2a/tasks/task_abc123")
        assert resp.status_code == 200


# =========================================================================
# Threads (no store attached)
# =========================================================================


class TestThreadsWithStore:
    """Threads with MemoryStore attached but no data created."""

    def test_list_threads_empty(self, client):
        resp = client.get("/api/v1/echo/threads")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_get_nonexistent_thread_404(self, client):
        resp = client.get("/api/v1/echo/threads/thread-abc")
        assert resp.status_code == 404

    def test_get_messages_empty(self, client):
        resp = client.get("/api/v1/echo/threads/thread-abc/messages")
        assert resp.status_code == 200


class TestThreadsWithoutStore:
    """Threads with NO store attached."""

    def test_list_no_store(self):
        p = AgentPlatform(agents_dir="/tmp/empty")

        async def fn(state):
            return {"response": "ok"}

        p.register_agent(manifest=AgentManifest(name="x", slug="x"), node_fn=fn)
        app = p.build()
        c = TestClient(app)
        resp = c.get("/api/v1/x/threads")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Thread storage not configured"

    def test_get_thread_no_store(self):
        p = AgentPlatform(agents_dir="/tmp/empty")

        async def fn(state):
            return {"response": "ok"}

        p.register_agent(manifest=AgentManifest(name="y", slug="y"), node_fn=fn)
        app = p.build()
        c = TestClient(app)
        resp = c.get("/api/v1/y/threads/t-xyz")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Thread storage not configured"


# =========================================================================
# Programmatic registration
# =========================================================================


class TestProgrammaticRegistration:
    def test_register_and_invoke(self):
        p = AgentPlatform(agents_dir="/tmp/empty")

        async def adder(state):
            nums = state.get("metadata", {})
            return {"response": str(nums.get("a", 0) + nums.get("b", 0))}

        p.register_agent(
            manifest=AgentManifest(name="calc", slug="calc"),
            node_fn=adder,
        )
        app = p.build()
        c = TestClient(app)
        resp = c.post("/api/v1/calc/invoke", json={"query": "add", "metadata": {"a": 3, "b": 7}})
        assert resp.status_code == 200
        assert resp.json()["response"] == "10"

    def test_multiple_agents(self):
        p = AgentPlatform(agents_dir="/tmp/empty")
        for name in ["agent_a", "agent_b", "agent_c"]:

            async def fn(state, _name=name):
                return {"response": f"I am {_name}"}

            p.register_agent(
                manifest=AgentManifest(name=name, slug=f"test-{name}"),
                node_fn=fn,
            )
        app = p.build()
        c = TestClient(app)
        for name in ["agent_a", "agent_b", "agent_c"]:
            resp = c.post(f"/api/v1/{name}/invoke", json={"query": "who"})
            assert resp.status_code == 200
            assert name in resp.json()["response"]


class TestSubagentFiltering:
    def test_non_subagent_excluded(self, platform, app):
        empty = platform.registry.get("empty")
        assert empty is not None
        assert empty.manifest.is_subagent is False


class TestPlatformFactory:
    def test_from_folder(self):
        p = AgentPlatform.from_folder("/tmp/test")
        assert p.agents_dir.name == "test"

    def test_custom_prefix(self):
        p = AgentPlatform(agents_dir="/tmp/t", api_prefix="/custom/v2")

        async def fn(state):
            return {"response": "ok"}

        p.register_agent(manifest=AgentManifest(name="x", slug="x"), node_fn=fn)
        app = p.build()
        c = TestClient(app)
        resp = c.post("/custom/v2/x/invoke", json={"query": "test"})
        assert resp.status_code == 200

    def test_custom_schemas_route(self):
        import sys
        from types import ModuleType

        from pydantic import BaseModel, Field

        class MockRequest(BaseModel):
            custom_query: str = Field(..., description="custom query field")
            custom_param: int = Field(0)

        class MockResponse(BaseModel):
            custom_answer: str
            custom_score: float

        schemas_mod = ModuleType("tests.test_agent_schemas.schemas")
        schemas_mod.CustomInvokeRequest = MockRequest
        schemas_mod.CustomInvokeResponse = MockResponse
        sys.modules["tests.test_agent_schemas.schemas"] = schemas_mod

        p = AgentPlatform(agents_dir="/tmp/empty")

        async def fn(state):
            return {
                "custom_answer": f"processed: {state.get('current_query')}",
                "custom_score": 0.95,
            }

        p.register_agent(
            manifest=AgentManifest(name="schema_test", slug="schema_test"),
            node_fn=fn,
        )
        agent = p._registry.get("schema_test")
        # Manually inject module path to route builder
        agent.module_path = "tests.test_agent_schemas"

        app = p.build()
        c = TestClient(app)

        # Test validation error (missing required custom_query)
        bad_resp = c.post("/api/v1/schema_test/invoke", json={"query": "test"})
        assert bad_resp.status_code == 422

        # Test valid custom schema payload
        good_resp = c.post(
            "/api/v1/schema_test/invoke", json={"custom_query": "hello", "custom_param": 42}
        )
        assert good_resp.status_code == 200
        data = good_resp.json()
        assert data["custom_answer"] == "processed: hello"
        assert data["custom_score"] == 0.95

        # Clean module reference
        if "tests.test_agent_schemas.schemas" in sys.modules:
            del sys.modules["tests.test_agent_schemas.schemas"]


# =========================================================================
# Lifecycle hooks
# =========================================================================


class TestLifecycleHooks:
    def test_startup_hook(self):
        called = []
        p = AgentPlatform(agents_dir="/tmp/empty")

        @p.on_startup
        async def hook():
            called.append("started")

        async def fn(state):
            return {"response": "ok"}

        p.register_agent(manifest=AgentManifest(name="h", slug="h"), node_fn=fn)
        app = p.build()
        with TestClient(app):
            assert "started" in called

    def test_shutdown_hook(self):
        called = []
        p = AgentPlatform(agents_dir="/tmp/empty")

        @p.on_shutdown
        async def hook():
            called.append("stopped")

        async def fn(state):
            return {"response": "ok"}

        p.register_agent(manifest=AgentManifest(name="h2", slug="h2"), node_fn=fn)
        app = p.build()
        with TestClient(app):
            pass  # triggers shutdown on exit
        assert "stopped" in called


# =========================================================================
# Storage (MemoryStore unit tests within integration context)
# =========================================================================


class TestMemoryStore:
    @pytest.fixture
    def store(self):
        return MemoryStore()

    @pytest.mark.asyncio
    async def test_thread_lifecycle(self, store):
        t = await store.create_thread("t1", "u1", "agent1", title="Test")
        assert t["id"] == "t1"
        assert t["title"] == "Test"

        got = await store.get_thread("t1")
        assert got is not None
        assert got["id"] == "t1"

    @pytest.mark.asyncio
    async def test_messages(self, store):
        await store.create_thread("t2", "u1", "a1")
        await store.add_message("t2", "user", "Hello")
        await store.add_message("t2", "assistant", "Hi there")
        msgs = await store.get_messages("t2")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "Hi there"

    @pytest.mark.asyncio
    async def test_list_filter(self, store):
        await store.create_thread("t3", "u1", "agent_a")
        await store.create_thread("t4", "u2", "agent_b")
        assert len(await store.list_threads(agent_name="agent_a")) == 1
        assert len(await store.list_threads(user_id="u2")) == 1

    @pytest.mark.asyncio
    async def test_delete_thread(self, store):
        await store.create_thread("t5", "u1", "a1")
        assert await store.delete_thread("t5") is True
        assert await store.get_thread("t5") is None
        assert await store.delete_thread("t5") is False

    @pytest.mark.asyncio
    async def test_feedback(self, store):
        fb = await store.add_feedback("t1", "u1", "a1", rating=5, comment="Great")
        assert fb["rating"] == 5
        fbs = await store.get_feedback(agent_name="a1")
        assert len(fbs) == 1

    @pytest.mark.asyncio
    async def test_stats(self, store):
        await store.create_thread("s1", "u1", "a1")
        await store.add_message("s1", "user", "Hi")
        stats = await store.get_stats()
        assert stats["threads"] == 1
        assert stats["messages"] == 1
        assert stats["backend"] == "MemoryStore"

    @pytest.mark.asyncio
    async def test_health(self, store):
        h = await store.health_check()
        assert h["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_pagination(self, store):
        await store.create_thread("p1", "u1", "a1")
        for i in range(10):
            await store.add_message("p1", "user", f"msg-{i}")
        page = await store.get_messages("p1", limit=3, offset=2)
        assert len(page) == 3
        assert page[0]["content"] == "msg-2"


# =========================================================================
# Extra routers
# =========================================================================


class TestExtraRouters:
    def test_include_router(self):
        from fastapi import APIRouter

        p = AgentPlatform(agents_dir="/tmp/empty")
        r = APIRouter()

        @r.get("/ping")
        async def ping():
            return {"pong": True}

        p.include_router(r, prefix="/custom")

        async def fn(state):
            return {"response": "ok"}

        p.register_agent(manifest=AgentManifest(name="x", slug="x"), node_fn=fn)

        app = p.build()
        c = TestClient(app)
        resp = c.get("/custom/ping")
        assert resp.status_code == 200
        assert resp.json()["pong"] is True
