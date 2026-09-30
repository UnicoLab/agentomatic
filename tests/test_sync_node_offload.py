# pyright: reportMissingParameterType=none
"""Sync graph nodes must not block the event loop.

Template nodes call ``self.llm.invoke(...)`` — a blocking network call. The
async execution paths ran such nodes inline on the event loop, so while a model
answered the whole server stood still: an ``async`` task submission only
returned once the LLM had finished, and task polling, SSE progress and health
checks stalled with it. Sync nodes now run in a worker thread.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.agents import BaseGraphAgent
from agentomatic.storage import MemoryStore


@dataclass
class _State:
    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class _SlowSyncAgent(BaseGraphAgent[_State]):
    """A sync node that blocks like an LLM call and reports task progress."""

    agent_name = "slow_sync"
    agent_description = "blocking sync node"
    delay = 0.6

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.set_finish_point("work")
        return g.compile()

    def work(self, state: _State) -> _State:
        from agentomatic.tasks import report_stage_sync

        report_stage_sync("thinking", percent=50.0)
        time.sleep(self.delay)
        state.output = {
            "response": f"done: {state.request}",
            "prompt": self.resolve_system_prompt(default="default prompt"),
            "thread": threading.current_thread().name,
        }
        return state

    def input_to_state(self, data: dict[str, Any]) -> _State:
        return _State(request=data.get("current_query", ""))

    def state_to_output(self, state: _State) -> dict[str, Any]:
        return state.output


@pytest.fixture
def client(tmp_path):
    platform = AgentPlatform(agents_dir=tmp_path / "agents", enable_studio=False)
    reg = _SlowSyncAgent().as_registered_agent()
    platform.register_agent(
        manifest=reg.manifest,
        node_fn=reg.node_fn,
        graph_fn=reg.graph_fn,
        class_instance=reg.class_instance,
    )
    with TestClient(platform.build()) as c:
        yield c


class TestEventLoopStaysResponsive:
    def test_async_task_submission_returns_before_the_node_finishes(self, client) -> None:
        started = time.perf_counter()
        response = client.post(
            "/api/v1/tasks",
            json={"target_type": "agent", "target": "slow_sync", "input": {"query": "x"}},
        )
        elapsed = time.perf_counter() - started
        assert response.status_code == 202
        assert elapsed < _SlowSyncAgent.delay / 2, f"submission blocked for {elapsed:.2f}s"

        # While the node sleeps, other requests are still served.
        health_started = time.perf_counter()
        assert client.get("/health").status_code == 200
        assert time.perf_counter() - health_started < _SlowSyncAgent.delay / 2

        task_id = response.json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            record = client.get(f"/api/v1/tasks/{task_id}").json()
            if record["status"] == "succeeded":
                break
            time.sleep(0.05)
        assert record["status"] == "succeeded", record
        assert record["result"]["response"] == "done: x"

    def test_progress_from_an_offloaded_sync_node_reaches_the_task(self, client) -> None:
        response = client.post(
            "/api/v1/tasks",
            json={
                "target_type": "agent",
                "target": "slow_sync",
                "input": {"query": "x"},
                "mode": "sync",
                "wait": True,
            },
        )
        task_id = response.json()["id"]
        with client.stream("GET", f"/api/v1/tasks/{task_id}/events?since=1") as stream:
            body = "".join(stream.iter_text())
        assert '"stage": "thinking"' in body or '"stage":"thinking"' in body, body

    def test_sync_node_runs_off_the_loop_thread(self, client) -> None:
        response = client.post("/api/v1/slow_sync/invoke", json={"query": "x"})
        assert response.status_code == 200
        thread_name = response.json()["output"]["thread"]
        assert thread_name != "MainThread"


class TestRequestPromptIsolation:
    async def test_concurrent_runs_keep_their_own_prompt_override(self) -> None:
        """Candidates scored concurrently must each see their own prompt."""
        agent = _SlowSyncAgent()
        agent.delay = 0.2

        async def run(prompt: str) -> str:
            out = await agent.atransform({"current_query": "q", "system_prompt_override": prompt})
            return out["prompt"]

        results = await asyncio.gather(*(run(f"prompt {i}") for i in range(4)))
        assert results == [f"prompt {i}" for i in range(4)]
        # Nothing leaks once the runs are over.
        assert agent._request_system_prompt is None

    def test_override_is_cleared_after_a_sync_transform(self) -> None:
        agent = _SlowSyncAgent()
        agent.delay = 0
        out = agent.transform({"current_query": "q", "system_prompt_override": "custom"})
        assert out["prompt"] == "custom"
        assert agent.transform({"current_query": "q"})["prompt"] == "default prompt"


class TestBoundedMemoryStore:
    async def test_oldest_threads_are_evicted_past_the_cap(self) -> None:
        store = MemoryStore(max_threads=2)
        for i in range(3):
            await store.create_thread(f"t{i}", "u", "a")
            await store.add_message(f"t{i}", "user", f"m{i}")
            await asyncio.sleep(0.001)
        assert await store.get_thread("t0") is None
        assert await store.get_messages("t0") == []
        assert await store.get_thread("t1") is not None
        assert await store.get_thread("t2") is not None

    async def test_recently_updated_thread_survives_eviction(self) -> None:
        store = MemoryStore(max_threads=2)
        await store.create_thread("old", "u", "a")
        await asyncio.sleep(0.001)
        await store.create_thread("mid", "u", "a")
        await asyncio.sleep(0.001)
        await store.add_message("old", "user", "still talking")
        await asyncio.sleep(0.001)
        await store.create_thread("new", "u", "a")
        assert await store.get_thread("old") is not None
        assert await store.get_thread("mid") is None

    def test_unbounded_by_default_and_validates_the_cap(self) -> None:
        assert MemoryStore().max_threads is None
        with pytest.raises(ValueError):
            MemoryStore(max_threads=0)


def test_build_invoke_state_does_not_mutate_the_payload() -> None:
    from agentomatic.core.agent_invoke import build_invoke_state

    payload: dict[str, Any] = {"query": "q", "temperature": 0.3, "metadata": {}, "context": {}}
    state = build_invoke_state(payload)
    assert state["metadata"] == {"temperature": 0.3}
    assert payload["metadata"] == {}
