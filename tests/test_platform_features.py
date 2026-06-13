"""Tests for core platform features: HITL, Structured Output, Thread Forking, and A/B Test Routing."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.core.router_factory import AgentSuspendedException
from agentomatic.providers.llm import get_structured_llm
from agentomatic.storage.memory import MemoryStore
from agentomatic.storage.sqlalchemy import SQLAlchemyStore

# =========================================================================
# Schemas for Testing
# =========================================================================


class SampleOutputModel(BaseModel):
    name: str
    age: int
    tags: list[str] = Field(default_factory=list)


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
        title="Features Test Platform",
        version="0.0.1",
        store=store,
    )

    # HITL suspended-execution agent
    async def hitl_fn(state):
        metadata = state.get("metadata") or {}
        if metadata.get("hitl_approved"):
            return {
                "response": f"Approved! Context: {metadata.get('approved_context')}",
                "agent_type": "test-hitl",
                "metadata": metadata,
            }

        raise AgentSuspendedException(
            approval_id="app_123",
            node_name="hitl_node",
            state_snapshot=state,
            message="Wait for human confirmation",
        )

    p.register_agent(
        manifest=AgentManifest(
            name="hitl_agent",
            slug="test-hitl",
            description="HITL agent for testing",
        ),
        node_fn=hitl_fn,
    )

    return p


@pytest.fixture
def app(platform):
    return platform.build()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# =========================================================================
# 1. Structured Output Enforcer Tests
# =========================================================================


def test_structured_llm_fallback():
    """Verify get_structured_llm handles non-native structured models cleanly."""
    structured_llm = get_structured_llm(SampleOutputModel, provider="dummy")
    res = structured_llm.invoke("Return structured name")

    assert isinstance(res, SampleOutputModel)
    assert res.name == "dummy_str"
    assert res.age == 0
    assert res.tags == []


# =========================================================================
# 2. Human-in-the-Loop (HITL) Tests
# =========================================================================


def test_hitl_suspend_and_resume(client):
    """Verify execution suspends, stores state, lists pending, and resumes on approval."""
    # 1. First invocation triggers suspension
    resp = client.post(
        "/api/v1/hitl_agent/invoke",
        json={"query": "test query", "thread_id": "thread_hitl_1"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["detail"]["status"] == "suspended"
    assert data["detail"]["approval_id"] == "app_123"
    assert data["detail"]["node_name"] == "hitl_node"

    # 2. List pending approvals
    pending_resp = client.get("/api/v1/hitl_agent/threads/thread_hitl_1/pending")
    assert pending_resp.status_code == 200
    pending_data = pending_resp.json()
    assert pending_data["count"] == 1
    assert pending_data["pending"][0]["id"] == "app_123"
    assert pending_data["pending"][0]["node_name"] == "hitl_node"

    # 3. Approve and resume execution
    approve_resp = client.post(
        "/api/v1/hitl_agent/threads/thread_hitl_1/approve",
        json={
            "approval_id": "app_123",
            "context": {"approved_context": "human_confirmed"},
        },
    )
    assert approve_resp.status_code == 200
    approve_data = approve_resp.json()
    assert approve_data["response"] == "Approved! Context: human_confirmed"

    # 4. Verify pending is now empty
    pending_resp = client.get("/api/v1/hitl_agent/threads/thread_hitl_1/pending")
    assert pending_resp.json()["count"] == 0


def test_hitl_reject(client):
    """Verify rejecting a suspended execution deletes it and aborts."""
    # 1. Trigger suspension
    resp = client.post(
        "/api/v1/hitl_agent/invoke",
        json={"query": "another test", "thread_id": "thread_hitl_2"},
    )
    assert resp.status_code == 202

    # 2. Reject execution
    reject_resp = client.post(
        "/api/v1/hitl_agent/threads/thread_hitl_2/reject",
        json={"approval_id": "app_123", "reason": "Not allowed"},
    )
    assert reject_resp.status_code == 200
    reject_data = reject_resp.json()
    assert reject_data["status"] == "rejected"
    assert reject_data["reason"] == "Not allowed"

    # 3. Verify pending is empty
    pending_resp = client.get("/api/v1/hitl_agent/threads/thread_hitl_2/pending")
    assert pending_resp.json()["count"] == 0


# =========================================================================
# 3. Thread Forking & Cloning Tests
# =========================================================================


@pytest.mark.asyncio
async def test_thread_forking_memory_and_db(store):
    """Test thread forking logic across memory/SQLAlchemy stores directly."""
    # Create thread and add messages
    thread_id = "parent_thread"
    await store.create_thread(thread_id, "user_1", "hitl_agent")
    await store.add_message(thread_id, "user", "Message 1")
    await store.add_message(thread_id, "assistant", "Response 1")
    await store.add_message(thread_id, "user", "Message 2")

    # Fork at index 1 (Message 1, Response 1)
    forked = await store.fork_thread(thread_id, 1, "forked_thread", title="My Fork")
    assert forked is not None
    assert forked["id"] == "forked_thread"
    assert forked["title"] == "My Fork"
    assert forked["message_count"] == 2

    # Verify messages in fork
    forked_messages = await store.get_messages("forked_thread")
    assert len(forked_messages) == 2
    assert forked_messages[0]["content"] == "Message 1"
    assert forked_messages[1]["content"] == "Response 1"

    # Test database SQLAlchemyStore (SQLite in-memory)
    db_store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await db_store.initialize()
    try:
        await db_store.create_thread(thread_id, "user_1", "hitl_agent")
        await db_store.add_message(thread_id, "user", "Message 1")
        await db_store.add_message(thread_id, "assistant", "Response 1")
        await db_store.add_message(thread_id, "user", "Message 2")

        db_forked = await db_store.fork_thread(thread_id, 1, "db_fork", title="DB Fork")
        assert db_forked is not None
        assert db_forked["id"] == "db_fork"
        assert db_forked["message_count"] == 2

        db_forked_msgs = await db_store.get_messages("db_fork")
        assert len(db_forked_msgs) == 2
        assert db_forked_msgs[0]["content"] == "Message 1"
        assert db_forked_msgs[1]["content"] == "Response 1"
    finally:
        await db_store.close()


@pytest.mark.asyncio
async def test_thread_forking_endpoint(client, store):
    """Test thread forking through the REST endpoint."""
    # Seed messages directly to store
    await store.create_thread("thread_endpoints", "user_1", "hitl_agent")
    await store.add_message("thread_endpoints", "user", "Msg 1")
    await store.add_message("thread_endpoints", "assistant", "Reply 1")

    # Fork thread REST API
    fork_resp = client.post(
        "/api/v1/hitl_agent/threads/thread_endpoints/fork",
        json={"message_index": 0, "new_thread_id": "thread_endpoint_fork", "title": "API Fork"},
    )
    assert fork_resp.status_code == 200
    fork_data = fork_resp.json()
    assert fork_data["id"] == "thread_endpoint_fork"
    assert fork_data["message_count"] == 1

    # Verify messages in forked thread
    msgs_resp = client.get("/api/v1/hitl_agent/threads/thread_endpoint_fork/messages")
    assert msgs_resp.status_code == 200
    msgs_data = msgs_resp.json()
    assert msgs_data["count"] == 1
    assert msgs_data["messages"][0]["content"] == "Msg 1"


# =========================================================================
# 4. A/B Test Prompt Router Tests
# =========================================================================


class MockABConfig:
    prompt_ab_tests = {"v1": 0.0, "v2": 1.0}  # Force 100% v2 version


def test_ab_prompt_router(platform, client):
    """Verify A/B prompt weights are respected and selected version is propagated."""
    agent = platform.registry.get("hitl_agent")
    # Inject MockABConfig to force prompt version selection
    agent.config = MockABConfig()

    # Call invoke without setting prompt_version (default is v1, which triggers A/B routing)
    # Ensure it routes to v2 because of 100% weight configuration
    resp = client.post(
        "/api/v1/hitl_agent/invoke",
        json={"query": "test query", "thread_id": "thread_ab_1"},
    )
    # Catch 202 since this agent triggers HITL
    assert resp.status_code == 202

    # Verify the state snapshot stored in memory has the selected prompt version
    pending_resp = client.get("/api/v1/hitl_agent/threads/thread_ab_1/pending")
    pending_data = pending_resp.json()
    state_snap = pending_data["pending"][0]["state_snapshot"]
    assert state_snap["prompt_version"] == "v2"

    # Call invoke with explicit override (non-v1 default)
    resp_override = client.post(
        "/api/v1/hitl_agent/invoke",
        json={
            "query": "test query",
            "thread_id": "thread_ab_2",
            "prompt_version": "v3",
        },
    )
    assert resp_override.status_code == 202
    pending_resp_override = client.get("/api/v1/hitl_agent/threads/thread_ab_2/pending")
    state_snap_override = pending_resp_override.json()["pending"][0]["state_snapshot"]
    assert state_snap_override["prompt_version"] == "v3"


# =========================================================================
# 5. AgentomaticCheckpointer (LangGraph checkpointer) Tests
# =========================================================================


@pytest.mark.asyncio
async def test_agentomatic_checkpointer_memory(store):
    """Verify checkpointer put/get/list works seamlessly with MemoryStore."""
    from agentomatic.storage.checkpointer import AgentomaticCheckpointer

    checkpointer = AgentomaticCheckpointer(store)

    config = {
        "configurable": {
            "thread_id": "thread_lg_1",
            "checkpoint_ns": "ns1",
            "checkpoint_id": "cp1",
        }
    }
    checkpoint = {"v": 1, "ts": "2026-06-13", "channel_values": {"my_key": "my_val"}}
    metadata = {"source": "input", "step": 1, "parent_checkpoint_id": None}

    # Save checkpoint
    await checkpointer.aput(config, checkpoint, metadata, {})

    # Retrieve checkpoint
    retrieved = await checkpointer.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["channel_values"]["my_key"] == "my_val"
    assert retrieved.metadata["step"] == 1

    # List checkpoints
    cps = await checkpointer._alist_list(config)
    assert len(cps) == 1
    assert cps[0].config["configurable"]["checkpoint_id"] == "cp1"


@pytest.mark.asyncio
async def test_agentomatic_checkpointer_db():
    """Verify checkpointer put/get/list works seamlessly with SQLAlchemyStore."""
    from agentomatic.storage.checkpointer import AgentomaticCheckpointer

    db_store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await db_store.initialize()
    try:
        checkpointer = AgentomaticCheckpointer(db_store)

        config = {
            "configurable": {
                "thread_id": "thread_lg_2",
                "checkpoint_ns": "ns2",
                "checkpoint_id": "cp2",
            }
        }
        checkpoint = {"v": 1, "ts": "2026-06-13", "channel_values": {"my_key": "db_val"}}
        metadata = {"source": "input", "step": 2, "parent_checkpoint_id": None}

        # Save checkpoint
        await checkpointer.aput(config, checkpoint, metadata, {})

        # Retrieve checkpoint
        retrieved = await checkpointer.aget_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["channel_values"]["my_key"] == "db_val"

        # List checkpoints
        cps = await checkpointer._alist_list(config)
        assert len(cps) == 1
        assert cps[0].config["configurable"]["checkpoint_id"] == "cp2"
    finally:
        await db_store.close()


# =========================================================================
# 6. LLM Provider Failover & Fallback Middleware Tests
# =========================================================================


def test_failover_fallback_llm():
    """Verify that get_llm supports fallbacks parameters and chains them."""
    from agentomatic.providers.llm import get_llm

    # We pass 'dummy' as primary, and another model list as fallbacks
    llm = get_llm(provider="dummy", fallbacks=["dummy"])
    # It should have a fallback list (which uses ChatOpenAI/etc. or fallbacks RunnableWithFallbacks)
    assert hasattr(llm, "fallbacks") or llm.__class__.__name__ == "RunnableWithFallbacks"


# =========================================================================
# 7. State-Level Dynamic Middleware (Hook Interceptors) Tests
# =========================================================================


def test_state_level_dynamic_middleware_hooks(platform, client):
    """Verify that startup before_node and after_node hooks execute around node invocations."""
    called_hooks = []

    def before_hook(agent_name, state):
        called_hooks.append(("before", agent_name, state.get("thread_id")))

    def after_hook(agent_name, result):
        called_hooks.append(("after", agent_name, result.get("agent_type")))

    platform.register_before_node_hook(before_hook)
    platform.register_after_node_hook(after_hook)

    # Trigger invocation (will raise suspension but still triggers before hook)
    client.post(
        "/api/v1/hitl_agent/invoke",
        json={"query": "test query", "thread_id": "thread_hook_1"},
    )

    # We expect 'before' hook to have run
    assert len(called_hooks) == 1
    assert called_hooks[0][0] == "before"
    assert called_hooks[0][1] == "hitl_agent"
    assert called_hooks[0][2] == "thread_hook_1"


# =========================================================================
# 8. Extra Edge Cases: Structured Fallbacks, Checkpointers & SQLAlchemy Store
# =========================================================================


class ComplexOutputModel(BaseModel):
    name: str
    num: float
    flag: bool
    items: list[int] = Field(default_factory=list)
    mapping: dict[str, str] = Field(default_factory=dict)
    maybe: str | None = None


def test_structured_output_fallback_parsing_and_types():
    """Verify StructuredOutputFallbackWrapper parses invalid JSON and handles all type annotations."""
    from agentomatic.providers.llm import StructuredOutputFallbackWrapper

    class FakeLLM:
        def __init__(self, content):
            self.content = content

        def invoke(self, *args, **kwargs):
            return self

    # Case 1: Valid JSON response
    wrapper_good = StructuredOutputFallbackWrapper(
        FakeLLM('{"name": "test", "num": 1.2, "flag": true}'), ComplexOutputModel
    )
    res_good = wrapper_good.invoke("dummy query")
    assert isinstance(res_good, ComplexOutputModel)
    assert res_good.name == "test"
    assert res_good.num == 1.2
    assert res_good.flag is True

    # Case 2: Invalid JSON response - triggers fallback parsing logic covering type defaults
    wrapper_bad = StructuredOutputFallbackWrapper(
        FakeLLM("invalid json string"), ComplexOutputModel
    )
    res_bad = wrapper_bad.invoke("dummy query")
    assert isinstance(res_bad, ComplexOutputModel)
    assert res_bad.name == "dummy_str"
    assert res_bad.num == 0.0
    assert res_bad.flag is False
    assert res_bad.items == []
    assert res_bad.mapping == {}
    assert res_bad.maybe is None


@pytest.mark.asyncio
async def test_agentomatic_checkpointer_sync_wrappers_and_errors(store):
    """Test sync wrappers and exception behaviors in AgentomaticCheckpointer."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from agentomatic.storage.checkpointer import AgentomaticCheckpointer

    checkpointer = AgentomaticCheckpointer(store)

    config = {
        "configurable": {
            "thread_id": "thread_lg_sync",
            "checkpoint_ns": "ns_sync",
            "checkpoint_id": "cp_sync_1",
        }
    }
    checkpoint = {"v": 1, "channel_values": {"x": 10}}
    metadata = {"source": "sync", "step": 1}

    # 1. Test aput requires thread_id
    bad_config = {"configurable": {}}
    with pytest.raises(ValueError, match="thread_id is required"):
        await checkpointer.aput(bad_config, checkpoint, metadata, {})

    # 2. Test get_tuple returns None when config is empty or missing thread_id
    res_none = await checkpointer.aget_tuple(bad_config)
    assert res_none is None

    # 3. Test synchronous execution of put/get_tuple/list outside of running event loop
    # We run inside an executor to simulate sync calling context (no active running loop in thread)
    def run_sync_ops():
        checkpointer.put(config, checkpoint, metadata, {})
        ret = checkpointer.get_tuple(config)
        all_cps = list(checkpointer.list(config))
        return ret, all_cps

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        t_res, t_list = await loop.run_in_executor(executor, run_sync_ops)

    assert t_res is not None
    assert t_res.checkpoint["channel_values"]["x"] == 10
    assert len(t_list) == 1
    assert t_list[0].config["configurable"]["checkpoint_id"] == "cp_sync_1"


@pytest.mark.asyncio
async def test_sqlalchemy_store_all_features_edge_cases():
    """Verify SQLAlchemyStore's implementation of feedback, stats, HITL states, checkpoints, and modifications."""
    db_store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await db_store.initialize()
    try:
        # 1. Thread update and delete
        await db_store.create_thread("thread_sqla", "user_sqla", "test_agent", title="Old Title")
        thread = await db_store.get_thread("thread_sqla")
        assert thread["title"] == "Old Title"

        updated = await db_store.update_thread(
            "thread_sqla", title="New Title", metadata_json={"foo": "bar"}
        )
        assert updated is not None
        assert updated["title"] == "New Title"
        assert updated["metadata"]["foo"] == "bar"

        # 2. Feedback operations
        fb = await db_store.add_feedback(
            thread_id="thread_sqla",
            user_id="user_sqla",
            agent_name="test_agent",
            rating=5,
            comment="Awesome!",
            feedback_type="stars",
        )
        assert fb["rating"] == 5
        assert fb["comment"] == "Awesome!"

        fbs = await db_store.get_feedback(agent_name="test_agent", user_id="user_sqla")
        assert len(fbs) == 1
        assert fbs[0]["comment"] == "Awesome!"

        # 3. Stats verification
        stats = await db_store.get_stats()
        assert stats["threads"] == 1
        assert stats["feedback"] == 1

        # 4. Suspended state lifecycle (HITL)
        suspended = await db_store.save_suspended_state(
            approval_id="app_sqla",
            thread_id="thread_sqla",
            agent_name="test_agent",
            node_name="test_node",
            state_json={"val": 42},
        )
        assert suspended["id"] == "app_sqla"

        pending = await db_store.list_suspended_states(
            thread_id="thread_sqla", agent_name="test_agent"
        )
        assert len(pending) == 1
        assert pending[0]["id"] == "app_sqla"

        fetched = await db_store.get_suspended_state("app_sqla")
        assert fetched is not None
        assert fetched["state_snapshot"]["val"] == 42

        deleted = await db_store.delete_suspended_state("app_sqla")
        assert deleted is True
        assert await db_store.get_suspended_state("app_sqla") is None

        # 5. Checkpointer Operations
        # Save first checkpoint
        await db_store.save_checkpoint(
            thread_id="thread_sqla",
            checkpoint_ns="ns_sqla",
            checkpoint_id="cp_sqla_1",
            parent_checkpoint_id=None,
            checkpoint={"step": 1},
            metadata={"source": "test"},
        )

        # Get latest checkpoint (without checkpoint_id)
        latest_cp = await db_store.get_checkpoint("thread_sqla", "ns_sqla", "")
        assert latest_cp is not None
        assert latest_cp["checkpoint_id"] == "cp_sqla_1"

        # Overwrite/update checkpoint
        await db_store.save_checkpoint(
            thread_id="thread_sqla",
            checkpoint_ns="ns_sqla",
            checkpoint_id="cp_sqla_1",
            parent_checkpoint_id="parent_id",
            checkpoint={"step": 1.1},
            metadata={"source": "test_update"},
        )
        updated_cp = await db_store.get_checkpoint("thread_sqla", "ns_sqla", "cp_sqla_1")
        assert updated_cp["parent_checkpoint_id"] == "parent_id"
        assert updated_cp["checkpoint"]["step"] == 1.1

        # List checkpoints with filter limitations
        await db_store.save_checkpoint(
            thread_id="thread_sqla",
            checkpoint_ns="ns_sqla",
            checkpoint_id="cp_sqla_2",
            parent_checkpoint_id="cp_sqla_1",
            checkpoint={"step": 2},
            metadata={"source": "test"},
        )

        cps_all = await db_store.list_checkpoints("thread_sqla", "ns_sqla")
        assert len(cps_all) == 2

        cps_limit = await db_store.list_checkpoints("thread_sqla", "ns_sqla", limit=1)
        assert len(cps_limit) == 1

        cps_before = await db_store.list_checkpoints("thread_sqla", "ns_sqla", before="cp_sqla_2")
        assert len(cps_before) == 1
        assert cps_before[0]["checkpoint_id"] == "cp_sqla_1"

        # Clean up delete thread
        is_deleted = await db_store.delete_thread("thread_sqla")
        assert is_deleted is True
        assert await db_store.get_thread("thread_sqla") is None

    finally:
        await db_store.close()
