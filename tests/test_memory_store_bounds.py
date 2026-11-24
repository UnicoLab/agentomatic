"""Tests for bounded capacity and TTL eviction across all 8 MemoryStore collections."""

from __future__ import annotations

import asyncio

import pytest

from agentomatic.storage.memory import MemoryStore


@pytest.mark.asyncio
async def test_memory_store_bounds_all_eight_stores_and_fork_thread() -> None:
    store = MemoryStore(
        max_threads=10,
        max_messages_per_thread=5,
        max_feedback=12,
        max_suspended_states=8,
        max_checkpoints=15,
        max_invocation_logs=10,
        max_log_analyses=6,
        max_optimization_runs=6,
        ttl_seconds=60.0,
    )

    for i in range(40):
        tid = f"thread-{i}"
        await store.create_thread(tid, "user-1", "agent-a")
        for m in range(8):
            await store.add_message(tid, "user", f"msg-{m}")
        if i % 4 == 0:
            await store.fork_thread(tid, 6, f"fork-{i}")
        await store.add_feedback(tid, "user-1", "agent-a", rating=5)
        await store.save_suspended_state(f"appr-{i}", tid, "agent-a", "node", {"i": i})
        await store.save_checkpoint(f"cp-thread-{i}", "ns", f"cp-{i}", None, {"i": i}, {})
        await store.create_invocation_log(agent_name="agent-a", thread_id=tid, log_id=f"log-{i}")
        await store.save_log_analysis(
            agent_name="agent-a", score=0.8, summary="ok", status="ok", analysis_id=f"ana-{i}"
        )
        await store.create_optimization_run(
            experiment_id="exp", agent_name="agent-a", run_id=f"run-{i}"
        )

    assert len(store._threads) <= 10
    assert max(len(msgs) for msgs in store._messages.values()) <= 5
    assert len(store._feedback) <= 12
    assert len(store._suspended_states) <= 8
    assert len(store._checkpoints) <= 15
    assert len(store._invocation_logs) <= 10
    assert len(store._log_analyses) <= 6
    assert len(store._optimization_runs) <= 6


@pytest.mark.asyncio
async def test_memory_store_ttl_expiration() -> None:
    store = MemoryStore(max_threads=50, ttl_seconds=0.04)
    await store.create_thread("t1", "u1", "a1")
    await store.add_feedback("t1", "u1", "a1", rating=4)
    await store.save_checkpoint("t1", "ns", "c1", None, {}, {})
    await store.create_invocation_log(agent_name="a1", log_id="l1")
    await store.save_log_analysis(
        agent_name="a1", score=1.0, summary="s", status="ok", analysis_id="an1"
    )
    await store.create_optimization_run(experiment_id="e1", agent_name="a1", run_id="r1")

    await asyncio.sleep(0.08)

    assert await store.get_thread("t1") is None
    assert await store.get_feedback() == []
    assert await store.get_checkpoint("t1", "ns", "c1") is None
    assert await store.get_invocation_log("l1") is None
    assert await store.get_latest_log_analysis("a1") is None
    assert await store.get_optimization_run("r1") is None
