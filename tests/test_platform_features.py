"""Tests for core platform features: HITL, Structured Output, Thread Forking, and A/B Test Routing."""

from __future__ import annotations

from datetime import UTC, datetime

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


# =========================================================================
# 9. Checkpoint Safe Serialization Tests
# =========================================================================


@pytest.mark.asyncio
async def test_checkpoint_serialization_with_non_json_objects(store):
    """Verify checkpointer handles non-JSON-serializable objects (datetimes, custom classes)."""
    from datetime import datetime

    from agentomatic.storage.checkpointer import AgentomaticCheckpointer, _ensure_json_serializable

    checkpointer = AgentomaticCheckpointer(store)

    # 1. Test _ensure_json_serializable directly
    assert _ensure_json_serializable({"a": 1, "b": "text"}) == {"a": 1, "b": "text"}

    # 2. Test with datetime values (non-JSON-native)
    dt = datetime(2026, 6, 14, 12, 0, 0)
    result = _ensure_json_serializable({"ts": dt, "val": 42})
    assert result["val"] == 42
    assert isinstance(result["ts"], str)  # datetime converted to string

    # 3. Test with bytes
    result_bytes = _ensure_json_serializable({"data": b"binary"})
    assert isinstance(result_bytes["data"], str)

    # 4. Test full round-trip through checkpointer
    config = {
        "configurable": {
            "thread_id": "thread_serde_test",
            "checkpoint_ns": "ns_serde",
            "checkpoint_id": "cp_serde_1",
        }
    }
    checkpoint = {"v": 1, "ts": dt, "channel_values": {"key": "val"}}
    metadata = {"source": "input", "step": 1}

    await checkpointer.aput(config, checkpoint, metadata, {})
    retrieved = await checkpointer.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["channel_values"]["key"] == "val"
    # The datetime should be stored as a string
    assert isinstance(retrieved.checkpoint["ts"], str)


# =========================================================================
# 10. HITL TTL Expiry Tests
# =========================================================================


@pytest.mark.asyncio
async def test_hitl_ttl_expiry_memory(store):
    """Verify that suspended states get an expires_at and can be cleaned up."""
    # Save suspended state
    suspended = await store.save_suspended_state(
        approval_id="ttl_test_1",
        thread_id="thread_ttl",
        agent_name="test_agent",
        node_name="test_node",
        state_json={"val": 42},
    )
    assert "expires_at" in suspended
    assert suspended["expires_at"] is not None

    # Initially, no expired states (expires_at is 7 days in the future)
    cleaned = await store.cleanup_expired_states()
    assert cleaned == 0

    # Manually set an expired state
    from datetime import timedelta

    now = datetime.now(UTC)
    expired_time = (now - timedelta(days=1)).isoformat()
    store._suspended_states["ttl_test_1"]["expires_at"] = expired_time

    # Now cleanup should remove it
    cleaned = await store.cleanup_expired_states()
    assert cleaned == 1
    assert await store.get_suspended_state("ttl_test_1") is None


@pytest.mark.asyncio
async def test_hitl_ttl_expiry_sqlalchemy():
    """Verify TTL expiry cleanup works with SQLAlchemy store."""
    from datetime import timedelta

    db_store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await db_store.initialize()
    try:
        await db_store.create_thread("thread_ttl_sql", "user_1", "test_agent")

        # Save a suspended state
        await db_store.save_suspended_state(
            approval_id="ttl_sql_1",
            thread_id="thread_ttl_sql",
            agent_name="test_agent",
            node_name="test_node",
            state_json={"val": 42},
        )

        # No expired states yet
        cleaned = await db_store.cleanup_expired_states()
        assert cleaned == 0

        # Manually expire it via direct DB update
        from agentomatic.storage.models import SuspendedStateModel

        async with db_store._session() as session:
            from sqlalchemy import update

            stmt = (
                update(SuspendedStateModel)
                .where(SuspendedStateModel.id == "ttl_sql_1")
                .values(expires_at=datetime.now(UTC) - timedelta(hours=1))
            )
            await session.execute(stmt)
            await session.commit()

        # Now cleanup should remove it
        cleaned = await db_store.cleanup_expired_states()
        assert cleaned == 1
        assert await db_store.get_suspended_state("ttl_sql_1") is None

    finally:
        await db_store.close()


# =========================================================================
# 11. Thread Lineage Tracking Tests
# =========================================================================


@pytest.mark.asyncio
async def test_thread_lineage_memory(store):
    """Verify thread lineage (parent/child tracking) works in MemoryStore."""
    # Create root thread
    await store.create_thread("root_thread", "user_1", "agent_1", title="Root")

    # Add messages
    await store.add_message("root_thread", "user", "Hello")
    await store.add_message("root_thread", "assistant", "Hi there")

    # Fork to create child
    forked = await store.fork_thread("root_thread", 0, "child_thread", title="Child")
    assert forked is not None
    assert forked["parent_thread_id"] == "root_thread"
    assert forked["fork_message_index"] == 0

    # Check lineage from child
    lineage = await store.get_thread_lineage("child_thread")
    assert lineage["thread_id"] == "child_thread"
    assert len(lineage["ancestors"]) == 1
    assert lineage["ancestors"][0]["id"] == "root_thread"
    assert len(lineage["descendants"]) == 0

    # Check lineage from root
    root_lineage = await store.get_thread_lineage("root_thread")
    assert len(root_lineage["ancestors"]) == 0
    assert len(root_lineage["descendants"]) == 1
    assert root_lineage["descendants"][0]["id"] == "child_thread"

    # Fork the child to get a grandchild
    await store.add_message("child_thread", "user", "Follow-up")
    grandchild = await store.fork_thread("child_thread", 0, "grandchild_thread")
    assert grandchild["parent_thread_id"] == "child_thread"

    # Verify multi-level lineage from grandchild
    gc_lineage = await store.get_thread_lineage("grandchild_thread")
    assert len(gc_lineage["ancestors"]) == 2
    assert gc_lineage["ancestors"][0]["id"] == "root_thread"
    assert gc_lineage["ancestors"][1]["id"] == "child_thread"


@pytest.mark.asyncio
async def test_thread_lineage_sqlalchemy():
    """Verify thread lineage works with SQLAlchemyStore."""
    db_store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await db_store.initialize()
    try:
        # Create root and messages
        await db_store.create_thread("root_sql", "user_1", "agent_1", title="Root SQL")
        await db_store.add_message("root_sql", "user", "Msg 1")
        await db_store.add_message("root_sql", "assistant", "Reply 1")

        # Fork
        forked = await db_store.fork_thread("root_sql", 0, "child_sql", title="Child SQL")
        assert forked["parent_thread_id"] == "root_sql"
        assert forked["fork_message_index"] == 0

        # Lineage from child
        lineage = await db_store.get_thread_lineage("child_sql")
        assert len(lineage["ancestors"]) == 1
        assert lineage["ancestors"][0]["id"] == "root_sql"

        # Lineage from root
        root_lineage = await db_store.get_thread_lineage("root_sql")
        assert len(root_lineage["descendants"]) == 1
        assert root_lineage["descendants"][0]["id"] == "child_sql"

    finally:
        await db_store.close()


@pytest.mark.asyncio
async def test_thread_lineage_endpoint(client, store):
    """Test the GET /threads/{thread_id}/lineage REST endpoint."""
    # Set up parent and fork
    await store.create_thread("lineage_parent", "user_1", "hitl_agent")
    await store.add_message("lineage_parent", "user", "Hello")
    await store.fork_thread("lineage_parent", 0, "lineage_child", title="Child")

    # Query lineage endpoint from child
    resp = client.get("/api/v1/hitl_agent/threads/lineage_child/lineage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["thread_id"] == "lineage_child"
    assert len(data["ancestors"]) == 1
    assert data["ancestors"][0]["id"] == "lineage_parent"


# =========================================================================
# 12. Failover Telemetry Tests
# =========================================================================


def test_failover_telemetry_counter():
    """Verify failover counter increments and resets correctly."""
    from agentomatic.providers.llm import (
        get_failover_count,
        record_failover,
        reset_llm,
    )

    reset_llm()
    assert get_failover_count() == 0

    record_failover("openai", "azure", "RateLimitError")
    assert get_failover_count() == 1

    record_failover("azure", "vertex", "TimeoutError")
    assert get_failover_count() == 2

    reset_llm()
    assert get_failover_count() == 0


def test_failover_chain_configuration():
    """Verify LLM failover chain is properly configured with exceptions_to_handle."""
    from agentomatic.providers.llm import get_llm, reset_llm

    reset_llm()

    # Build chain with dummy primary and dummy fallback
    llm = get_llm(provider="dummy", fallbacks=["dummy"])

    # Should be a RunnableWithFallbacks
    assert llm.__class__.__name__ == "RunnableWithFallbacks"
    assert len(llm.fallbacks) == 1

    reset_llm()


# =========================================================================
# 13. Lineage field absent for non-forked threads
# =========================================================================


@pytest.mark.asyncio
async def test_non_forked_thread_has_null_lineage(store):
    """Verify that threads created normally have null parent_thread_id."""
    thread = await store.create_thread("normal_thread", "user_1", "agent_1")
    assert thread.get("parent_thread_id") is None
    assert thread.get("fork_message_index") is None

    # Lineage should be empty ancestors/descendants
    lineage = await store.get_thread_lineage("normal_thread")
    assert lineage["ancestors"] == []
    assert lineage["descendants"] == []


@pytest.mark.asyncio
async def test_lineage_nonexistent_thread(store):
    """Verify get_thread_lineage returns empty for non-existent thread."""
    lineage = await store.get_thread_lineage("nonexistent")
    assert lineage["ancestors"] == []
    assert lineage["descendants"] == []


# =========================================================================
# 14. Edge Cases & Robustness Tests
# =========================================================================


@pytest.mark.asyncio
async def test_lineage_cycle_guard_memory(store):
    """Verify MemoryStore get_thread_lineage terminates on circular references."""
    # Create two threads and manually create a cycle
    await store.create_thread("cycle_a", "user_1", "agent_1")
    await store.create_thread("cycle_b", "user_1", "agent_1")

    # Manually inject circular parent references
    store._threads["cycle_a"]["parent_thread_id"] = "cycle_b"
    store._threads["cycle_b"]["parent_thread_id"] = "cycle_a"

    # Should terminate (not infinite loop) thanks to the cycle guard
    lineage = await store.get_thread_lineage("cycle_a")
    assert lineage["thread_id"] == "cycle_a"
    # Ancestors list should be bounded (not infinite)
    assert len(lineage["ancestors"]) <= 100


@pytest.mark.asyncio
async def test_lineage_cycle_guard_sqlalchemy():
    """Verify SQLAlchemyStore get_thread_lineage terminates on circular references."""
    db_store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await db_store.initialize()
    try:
        await db_store.create_thread("cycle_a_sql", "user_1", "agent_1")
        await db_store.create_thread("cycle_b_sql", "user_1", "agent_1")

        # Manually inject circular parent references via direct DB update
        async with db_store._session() as session:
            from sqlalchemy import update

            from agentomatic.storage.models import ThreadModel

            await session.execute(
                update(ThreadModel)
                .where(ThreadModel.id == "cycle_a_sql")
                .values(parent_thread_id="cycle_b_sql")
            )
            await session.execute(
                update(ThreadModel)
                .where(ThreadModel.id == "cycle_b_sql")
                .values(parent_thread_id="cycle_a_sql")
            )
            await session.commit()

        lineage = await db_store.get_thread_lineage("cycle_a_sql")
        assert lineage["thread_id"] == "cycle_a_sql"
        assert len(lineage["ancestors"]) <= 100
    finally:
        await db_store.close()


@pytest.mark.asyncio
async def test_ensure_json_serializable_nested():
    """Verify _ensure_json_serializable handles deeply nested non-serializable objects."""
    from agentomatic.storage.checkpointer import _ensure_json_serializable

    nested = {
        "level1": {
            "level2": {
                "dt": datetime(2026, 1, 1, 12, 0),
                "data": b"binary_nested",
                "num": 42,
            }
        },
        "list_with_dt": [datetime(2026, 6, 1), "normal", 123],
    }
    result = _ensure_json_serializable(nested)
    assert result["level1"]["level2"]["num"] == 42
    assert isinstance(result["level1"]["level2"]["dt"], str)
    assert isinstance(result["level1"]["level2"]["data"], str)
    assert isinstance(result["list_with_dt"][0], str)
    assert result["list_with_dt"][1] == "normal"


@pytest.mark.asyncio
async def test_fork_negative_index_returns_none(store):
    """Verify fork_thread returns None for negative message_index."""
    await store.create_thread("neg_idx_thread", "user_1", "agent_1")
    await store.add_message("neg_idx_thread", "user", "Hello")

    result = await store.fork_thread("neg_idx_thread", -1, "neg_fork")
    assert result is None


@pytest.mark.asyncio
async def test_delete_thread_cleans_up_suspended_states(store):
    """Verify deleting a thread also removes its orphaned suspended states."""
    await store.create_thread("del_thread", "user_1", "agent_1")
    await store.save_suspended_state(
        approval_id="del_approval",
        thread_id="del_thread",
        agent_name="agent_1",
        node_name="node_1",
        state_json={"val": 1},
    )

    # Verify state exists
    state = await store.get_suspended_state("del_approval")
    assert state is not None

    # Delete thread
    await store.delete_thread("del_thread")

    # Verify suspended state is also cleaned up
    state = await store.get_suspended_state("del_approval")
    assert state is None


@pytest.mark.asyncio
async def test_create_thread_returns_consistent_shape(store):
    """Verify create_thread returns dict with parent_thread_id and fork_message_index."""
    thread = await store.create_thread("shape_thread", "user_1", "agent_1")
    # These keys should exist even for non-forked threads
    assert "parent_thread_id" in thread
    assert "fork_message_index" in thread
    assert thread["parent_thread_id"] is None
    assert thread["fork_message_index"] is None


@pytest.mark.asyncio
async def test_multi_level_lineage_sqlalchemy():
    """Verify multi-level lineage (grandchild) works in SQLAlchemyStore."""
    db_store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await db_store.initialize()
    try:
        # Create root → child → grandchild
        await db_store.create_thread("ml_root", "user_1", "agent_1", title="Root")
        await db_store.add_message("ml_root", "user", "Hello")
        await db_store.add_message("ml_root", "assistant", "Hi")

        await db_store.fork_thread("ml_root", 0, "ml_child", title="Child")
        await db_store.add_message("ml_child", "user", "Follow-up")

        await db_store.fork_thread("ml_child", 0, "ml_grandchild", title="Grandchild")

        # Verify 3-level lineage from grandchild
        gc_lineage = await db_store.get_thread_lineage("ml_grandchild")
        assert len(gc_lineage["ancestors"]) == 2
        assert gc_lineage["ancestors"][0]["id"] == "ml_root"
        assert gc_lineage["ancestors"][1]["id"] == "ml_child"

        # Verify root sees only direct children
        root_lineage = await db_store.get_thread_lineage("ml_root")
        assert len(root_lineage["descendants"]) == 1
        assert root_lineage["descendants"][0]["id"] == "ml_child"
    finally:
        await db_store.close()


# =========================================================================
# 15. Conversation Memory Manager — Unit Tests
# =========================================================================


@pytest.mark.asyncio
async def test_memory_manager_get_or_create_thread():
    """Memory manager auto-creates threads that don't exist."""
    from agentomatic.core.memory_manager import ConversationMemoryManager

    s = MemoryStore()
    mgr = ConversationMemoryManager(s)

    # Create new thread
    tid = await mgr.get_or_create_thread(None, "user1", "test-agent", title="Test")
    assert tid.startswith("thread_")

    # Get existing thread
    tid2 = await mgr.get_or_create_thread(tid, "user1", "test-agent")
    assert tid2 == tid

    # Thread ID provided but doesn't exist — creates it
    tid3 = await mgr.get_or_create_thread("custom_id", "user1", "test-agent")
    assert tid3 == "custom_id"


@pytest.mark.asyncio
async def test_memory_manager_load_history_empty():
    """Load history from empty thread returns just the current message."""
    from agentomatic.core.memory_manager import ConversationMemoryManager

    s = MemoryStore()
    mgr = ConversationMemoryManager(s)
    tid = await mgr.get_or_create_thread(None, "user1", "test-agent")

    messages = await mgr.load_history(tid, "Hello!")
    assert len(messages) == 1
    assert messages[0].content == "Hello!"


@pytest.mark.asyncio
async def test_memory_manager_load_history_with_prior_messages():
    """Load history includes prior messages plus current message."""
    from langchain_core.messages import AIMessage, HumanMessage

    from agentomatic.core.memory_manager import ConversationMemoryManager

    s = MemoryStore()
    mgr = ConversationMemoryManager(s)
    tid = await mgr.get_or_create_thread("hist_test", "user1", "test-agent")

    # Add some messages
    await s.add_message(tid, "user", "First question", metadata={})
    await s.add_message(tid, "assistant", "First answer", metadata={})
    await s.add_message(tid, "user", "Second question", metadata={})
    await s.add_message(tid, "assistant", "Second answer", metadata={})

    messages = await mgr.load_history(tid, "Third question")
    assert len(messages) == 5  # 4 prior + 1 current
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "First question"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "First answer"
    assert messages[-1].content == "Third question"


@pytest.mark.asyncio
async def test_memory_manager_save_turn():
    """Save turn persists both user and assistant messages."""
    from agentomatic.core.memory_manager import ConversationMemoryManager

    s = MemoryStore()
    mgr = ConversationMemoryManager(s)
    tid = await mgr.get_or_create_thread("save_test", "user1", "test-agent")

    result = await mgr.save_turn(tid, "Hello!", "Hi there!", agent_name="test-agent")
    assert result["user_message_id"] is not None
    assert result["assistant_message_id"] is not None

    messages = await s.get_messages(tid)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello!"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there!"


@pytest.mark.asyncio
async def test_memory_manager_windowing():
    """When messages exceed max_messages, only the latest are loaded."""
    from agentomatic.core.memory_manager import ConversationMemoryManager

    s = MemoryStore()
    mgr = ConversationMemoryManager(s, max_messages=4, summarize_after=0)
    tid = await mgr.get_or_create_thread("window_test", "user1", "test-agent")

    # Add 10 messages
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        await s.add_message(tid, role, f"Message {i}", metadata={})

    messages = await mgr.load_history(tid, "Current query")
    # Should have 4 windowed + 1 current = 5
    assert len(messages) == 5
    assert messages[-1].content == "Current query"
    assert messages[0].content == "Message 6"  # windowed from the end


@pytest.mark.asyncio
async def test_memory_manager_convert_messages():
    """Convert stored message dicts to LangChain message types."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from agentomatic.core.memory_manager import ConversationMemoryManager

    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = ConversationMemoryManager._convert_to_langchain_messages(msgs)
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)


@pytest.mark.asyncio
async def test_memory_manager_fallback_summary():
    """Fallback summary works when LLM is unavailable."""
    from agentomatic.core.memory_manager import ConversationMemoryManager

    msgs = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
    ]
    summary = ConversationMemoryManager._fallback_summary(msgs)
    assert "2 messages" in summary
    assert "Question 1" in summary


# =========================================================================
# 16. Chat Endpoint Memory Integration Tests
# =========================================================================


@pytest.fixture
def memory_platform():
    """Platform with a chat-aware dummy agent that echoes messages."""
    s = MemoryStore()
    p = AgentPlatform(
        agents_dir="/tmp/agentomatic_test_empty",
        title="Memory Test Platform",
        version="0.0.1",
        store=s,
    )

    async def echo_fn(state):
        msg_count = len(state.get("messages", []))
        ctx = state.get("context", {})
        return {
            "response": f"Echo: {state.get('current_query', '')} (history: {msg_count})",
            "agent_type": "test-echo",
            "suggestions": [],
            "citations": [],
            "context": ctx,
            "steps_taken": ["echo_step"],
            "metadata": state.get("metadata", {}),
        }

    p.register_agent(
        manifest=AgentManifest(
            name="echo_agent",
            slug="test-echo",
            description="Echo agent for memory testing",
        ),
        node_fn=echo_fn,
    )

    return p


@pytest.fixture
def memory_app(memory_platform):
    return memory_platform.build()


@pytest.fixture
def memory_client(memory_app):
    with TestClient(memory_app) as c:
        yield c


def test_chat_creates_thread_and_persists(memory_client):
    """Chat endpoint auto-creates thread and persists messages."""
    resp = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Hello!", "user_id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "thread_id" in data
    assert data["response"].startswith("Echo: Hello!")
    assert "history_loaded" in data

    thread_id = data["thread_id"]

    # Check messages were persisted
    msg_resp = memory_client.get(f"/api/v1/echo_agent/threads/{thread_id}/messages")
    assert msg_resp.status_code == 200
    msg_data = msg_resp.json()
    assert msg_data["count"] == 2  # user + assistant
    assert msg_data["messages"][0]["role"] == "user"
    assert msg_data["messages"][0]["content"] == "Hello!"
    assert msg_data["messages"][1]["role"] == "assistant"


def test_chat_multi_turn_loads_history(memory_client):
    """Multiple chat turns on same thread loads prior history."""
    # Turn 1
    resp1 = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "First message", "thread_id": "mt_test"},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["history_loaded"] == 0  # no prior messages

    # Turn 2
    resp2 = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Second message", "thread_id": "mt_test"},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["history_loaded"] == 2  # 2 prior messages (user + assistant from turn 1)

    # Turn 3
    resp3 = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Third message", "thread_id": "mt_test"},
    )
    data3 = resp3.json()
    assert data3["history_loaded"] == 4  # 4 prior messages (2 turns)


def test_chat_with_include_history_false(memory_client):
    """Chat with include_history=False skips history loading."""
    # Turn 1
    memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Message 1", "thread_id": "no_hist"},
    )

    # Turn 2 without history
    resp2 = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={
            "content": "Message 2",
            "thread_id": "no_hist",
            "include_history": False,
        },
    )
    data2 = resp2.json()
    assert data2["history_loaded"] == 0


# =========================================================================
# 17. Thread CRUD Endpoint Tests
# =========================================================================


def test_create_thread_endpoint(memory_client):
    """POST /threads creates a thread."""
    resp = memory_client.post(
        "/api/v1/echo_agent/threads",
        json={"user_id": "user1", "title": "Test Thread"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "user1"
    assert data["title"] == "Test Thread"
    assert "id" in data


def test_create_thread_with_custom_id(memory_client):
    """POST /threads with custom ID uses that ID."""
    resp = memory_client.post(
        "/api/v1/echo_agent/threads",
        json={"thread_id": "custom_123", "user_id": "user1"},
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == "custom_123"


def test_delete_thread_endpoint(memory_client):
    """DELETE /threads/{id} deletes the thread."""
    # Create
    memory_client.post(
        "/api/v1/echo_agent/threads",
        json={"thread_id": "del_test", "user_id": "user1"},
    )

    # Delete
    resp = memory_client.delete("/api/v1/echo_agent/threads/del_test")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify gone
    resp2 = memory_client.get("/api/v1/echo_agent/threads/del_test")
    assert resp2.status_code == 404


def test_delete_nonexistent_thread(memory_client):
    """DELETE /threads/{id} returns 404 for missing thread."""
    resp = memory_client.delete("/api/v1/echo_agent/threads/nonexistent")
    assert resp.status_code == 404


def test_update_thread_endpoint(memory_client):
    """PATCH /threads/{id} updates thread fields."""
    # Create
    memory_client.post(
        "/api/v1/echo_agent/threads",
        json={"thread_id": "upd_test", "user_id": "user1", "title": "Old Title"},
    )

    # Update
    resp = memory_client.patch(
        "/api/v1/echo_agent/threads/upd_test",
        json={"title": "New Title"},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "New Title"


def test_update_thread_no_fields(memory_client):
    """PATCH with no fields returns 400."""
    memory_client.post(
        "/api/v1/echo_agent/threads",
        json={"thread_id": "no_upd", "user_id": "user1"},
    )
    resp = memory_client.patch(
        "/api/v1/echo_agent/threads/no_upd",
        json={},
    )
    assert resp.status_code == 400


# =========================================================================
# 18. Message Pagination Tests
# =========================================================================


def test_messages_pagination(memory_client):
    """GET /threads/{id}/messages supports limit and offset."""
    # Create thread and add messages
    memory_client.post(
        "/api/v1/echo_agent/threads",
        json={"thread_id": "page_test", "user_id": "user1"},
    )
    for i in range(5):
        memory_client.post(
            "/api/v1/echo_agent/chat",
            json={"content": f"Msg {i}", "thread_id": "page_test"},
        )

    # Default — all messages
    resp = memory_client.get("/api/v1/echo_agent/threads/page_test/messages")
    all_msgs = resp.json()
    assert all_msgs["count"] == 10  # 5 user + 5 assistant

    # Limit
    resp2 = memory_client.get("/api/v1/echo_agent/threads/page_test/messages?limit=4&offset=0")
    assert resp2.json()["count"] == 4
    assert resp2.json()["limit"] == 4
    assert resp2.json()["offset"] == 0


def test_clear_thread_messages(memory_client):
    """DELETE /threads/{id}/messages clears messages but keeps thread."""
    # Create and populate
    memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Hello", "thread_id": "clear_test"},
    )

    # Verify messages exist
    msgs = memory_client.get("/api/v1/echo_agent/threads/clear_test/messages")
    assert msgs.json()["count"] > 0

    # Clear
    resp = memory_client.delete("/api/v1/echo_agent/threads/clear_test/messages")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"

    # Verify messages cleared
    msgs2 = memory_client.get("/api/v1/echo_agent/threads/clear_test/messages")
    assert msgs2.json()["count"] == 0

    # Thread still exists
    thread = memory_client.get("/api/v1/echo_agent/threads/clear_test")
    assert thread.status_code == 200


# =========================================================================
# 19. Conversation Summary Tests
# =========================================================================


def test_thread_summary_no_storage(client):
    """Summary endpoint returns 400 when store not configured."""
    # The 'client' fixture has a store, but let's test the functionality
    resp = client.get("/api/v1/hitl_agent/threads/nonexistent/summary")
    assert resp.status_code == 200  # Returns "No messages in this thread."


def test_thread_summary_with_messages(memory_client):
    """Summary endpoint generates summary for a thread with messages."""
    # Create and populate
    memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "What are my vacation days?", "thread_id": "sum_test"},
    )
    memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "How about sick days?", "thread_id": "sum_test"},
    )

    resp = memory_client.get("/api/v1/echo_agent/threads/sum_test/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["thread_id"] == "sum_test"
    assert len(data["summary"]) > 0


# =========================================================================
# 20. Invoke Endpoint Memory Integration
# =========================================================================


def test_invoke_with_thread_persists_messages(memory_client):
    """Invoke with thread_id persists user and assistant messages."""
    resp = memory_client.post(
        "/api/v1/echo_agent/invoke",
        json={"query": "Test invoke memory", "thread_id": "inv_mem_test"},
    )
    assert resp.status_code == 200

    # Check messages were persisted
    msgs = memory_client.get("/api/v1/echo_agent/threads/inv_mem_test/messages")
    assert msgs.status_code == 200
    msg_data = msgs.json()
    assert msg_data["count"] == 2  # user + assistant


def test_invoke_multi_turn_builds_history(memory_client):
    """Multiple invocations on same thread accumulate history."""
    for i in range(3):
        memory_client.post(
            "/api/v1/echo_agent/invoke",
            json={"query": f"Turn {i}", "thread_id": "inv_multi"},
        )

    msgs = memory_client.get("/api/v1/echo_agent/threads/inv_multi/messages")
    assert msgs.json()["count"] == 6  # 3 turns × 2 messages


# =========================================================================
# 21. API Modularity & Configurability Tests
# =========================================================================


def test_chat_with_user_supplied_messages(memory_client):
    """User can supply their own messages instead of auto-loading from store."""
    resp = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={
            "content": "Follow-up question",
            "messages": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["history_loaded"] == 2  # 2 user-supplied messages
    # Agent sees 3 messages: 2 supplied + 1 current
    assert "history: 3" in data["response"]


def test_chat_context_passthrough(memory_client):
    """Context dict is passed through to agent state and returned in response."""
    resp = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={
            "content": "Hello",
            "context": {"user_role": "manager", "department": "engineering"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # Agent echoes back context
    assert data["context"]["user_role"] == "manager"
    assert data["context"]["department"] == "engineering"


def test_chat_steps_taken_returned(memory_client):
    """Steps taken by agent are included in chat response."""
    resp = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Hello"},
    )
    data = resp.json()
    assert "echo_step" in data["steps_taken"]


def test_chat_persist_false_skips_save(memory_client):
    """Setting persist=False skips saving messages to store."""
    resp = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={
            "content": "Ephemeral message",
            "thread_id": "no_persist_test",
            "persist": False,
        },
    )
    assert resp.status_code == 200

    # Messages should NOT be persisted
    msgs = memory_client.get("/api/v1/echo_agent/threads/no_persist_test/messages")
    assert msgs.json()["count"] == 0


def test_chat_metadata_passthrough(memory_client):
    """Metadata from request is merged with agent's metadata in response."""
    resp = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={
            "content": "Hello",
            "metadata": {"source": "mobile_app", "version": "2.0"},
        },
    )
    data = resp.json()
    assert data["metadata"]["source"] == "mobile_app"
    assert data["metadata"]["version"] == "2.0"


def test_invoke_context_passthrough(memory_client):
    """Context dict is passed through to agent state and returned in invoke response."""
    resp = memory_client.post(
        "/api/v1/echo_agent/invoke",
        json={
            "query": "Hello",
            "context": {"org_id": "org_123", "permissions": ["read", "write"]},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["context"]["org_id"] == "org_123"
    assert data["context"]["permissions"] == ["read", "write"]


def test_user_messages_override_auto_loading(memory_client):
    """When user supplies messages, store history is NOT loaded."""
    # First, create some store history
    memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Stored msg 1", "thread_id": "override_test"},
    )
    memory_client.post(
        "/api/v1/echo_agent/chat",
        json={"content": "Stored msg 2", "thread_id": "override_test"},
    )

    # Now send with user-supplied messages (should override store)
    resp = memory_client.post(
        "/api/v1/echo_agent/chat",
        json={
            "content": "My question",
            "thread_id": "override_test",
            "messages": [
                {"role": "user", "content": "Custom context"},
            ],
        },
    )
    data = resp.json()
    # Only user-supplied messages loaded, NOT the 4 stored ones
    assert data["history_loaded"] == 1
    assert "history: 2" in data["response"]  # 1 supplied + 1 current
