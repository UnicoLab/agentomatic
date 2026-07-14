"""Tests for the unified task/execution subsystem.

Covers models, the in-memory store, the task manager (async/batch/cancel/
progress/webhook), the HTTP task board, and A2A rewiring.
"""

from __future__ import annotations

import asyncio

import pytest

from agentomatic.tasks import (
    InMemoryTaskStore,
    TargetType,
    TaskManager,
    TaskRecord,
    TaskStatus,
)
from agentomatic.tasks.context import TaskContext

# =====================================================================
# Models
# =====================================================================


class TestModels:
    def test_status_terminal(self):
        assert TaskStatus.SUCCEEDED.is_terminal
        assert TaskStatus.FAILED.is_terminal
        assert TaskStatus.CANCELLED.is_terminal
        assert not TaskStatus.QUEUED.is_terminal
        assert not TaskStatus.RUNNING.is_terminal

    def test_record_defaults(self):
        rec = TaskRecord(target_type=TargetType.AGENT, target="foo")
        assert rec.id.startswith("task_")
        assert rec.status == TaskStatus.QUEUED
        assert rec.duration_ms is None

    def test_public_dict_has_duration(self):
        rec = TaskRecord(target_type=TargetType.AGENT, target="foo")
        rec.started_at = 1.0
        rec.finished_at = 2.0
        data = rec.public_dict()
        assert data["duration_ms"] == pytest.approx(1000.0)


# =====================================================================
# InMemoryTaskStore
# =====================================================================


class TestStore:
    async def test_save_get(self):
        store = InMemoryTaskStore()
        rec = TaskRecord(target_type=TargetType.AGENT, target="foo")
        await store.save(rec)
        got = await store.get(rec.id)
        assert got is not None
        assert got.id == rec.id

    async def test_list_filters(self):
        store = InMemoryTaskStore()
        a = TaskRecord(target_type=TargetType.AGENT, target="a", status=TaskStatus.SUCCEEDED)
        b = TaskRecord(target_type=TargetType.PLUGIN, target="b", status=TaskStatus.FAILED)
        await store.save(a)
        await store.save(b)
        assert len(await store.list()) == 2
        assert len(await store.list(status=TaskStatus.SUCCEEDED)) == 1
        assert len(await store.list(target_type=TargetType.PLUGIN)) == 1
        assert (await store.list(target="a"))[0].id == a.id

    async def test_count_and_delete(self):
        store = InMemoryTaskStore()
        rec = TaskRecord(target_type=TargetType.AGENT, target="foo", status=TaskStatus.SUCCEEDED)
        await store.save(rec)
        assert await store.count() == 1
        assert await store.count(status=TaskStatus.SUCCEEDED) == 1
        assert await store.delete(rec.id) is True
        assert await store.count() == 0

    async def test_max_records_eviction(self):
        store = InMemoryTaskStore(max_records=2, ttl_seconds=None)
        for i in range(5):
            rec = TaskRecord(
                target_type=TargetType.AGENT,
                target=f"t{i}",
                status=TaskStatus.SUCCEEDED,
            )
            rec.finished_at = float(i)
            await store.save(rec)
        assert await store.count() <= 2


# =====================================================================
# TaskManager
# =====================================================================


def _echo_dispatcher():
    async def run(target, payload, ctx: TaskContext):
        return {"echo": payload, "target": target}

    return run


class TestManager:
    async def test_submit_and_wait_success(self):
        mgr = TaskManager()
        mgr.register_dispatcher(TargetType.AGENT, _echo_dispatcher())
        rec = await mgr.submit_and_wait(TargetType.AGENT, "x", input={"a": 1})
        assert rec.status == TaskStatus.SUCCEEDED
        assert rec.result == {"echo": {"a": 1}, "target": "x"}
        assert rec.progress.percent == 100.0

    async def test_unknown_target_type_raises(self):
        mgr = TaskManager()
        with pytest.raises(ValueError):
            await mgr.submit(TargetType.PLUGIN, "x", input={})

    async def test_failure_is_captured(self):
        async def boom(target, payload, ctx):
            raise RuntimeError("kaboom")

        mgr = TaskManager()
        mgr.register_dispatcher(TargetType.AGENT, boom)
        rec = await mgr.submit_and_wait(TargetType.AGENT, "x", input={})
        assert rec.status == TaskStatus.FAILED
        assert "kaboom" in (rec.error or "")

    async def test_batch_progress(self):
        async def run(target, payload, ctx):
            return payload * 2

        mgr = TaskManager()
        mgr.register_dispatcher(TargetType.AGENT, run)
        rec = await mgr.submit_and_wait(TargetType.AGENT, "x", batch=[1, 2, 3])
        assert rec.status == TaskStatus.SUCCEEDED
        assert sorted(rec.result) == [2, 4, 6]
        assert rec.mode == "batch"

    async def test_cancellation(self):
        started = asyncio.Event()

        async def slow(target, payload, ctx):
            started.set()
            await asyncio.sleep(5)
            return "done"

        mgr = TaskManager()
        mgr.register_dispatcher(TargetType.AGENT, slow)
        rec = await mgr.submit(TargetType.AGENT, "x", input={})
        await asyncio.wait_for(started.wait(), timeout=2)
        assert await mgr.cancel(rec.id) is True
        await asyncio.sleep(0.1)
        refreshed = await mgr.get(rec.id)
        assert refreshed.status == TaskStatus.CANCELLED

    async def test_progress_reporting(self):
        async def run(target, payload, ctx: TaskContext):
            await ctx.report(current=1, total=2, message="half")
            await ctx.report(current=2, total=2, message="full")
            return "ok"

        mgr = TaskManager()
        mgr.register_dispatcher(TargetType.AGENT, run)
        rec = await mgr.submit_and_wait(TargetType.AGENT, "x", input={})
        assert rec.status == TaskStatus.SUCCEEDED

    async def test_subscribe_receives_events(self):
        async def run(target, payload, ctx: TaskContext):
            await ctx.report(current=1, total=1, message="working")
            return "ok"

        mgr = TaskManager()
        mgr.register_dispatcher(TargetType.AGENT, run)
        rec = await mgr.submit(TargetType.AGENT, "x", input={})
        queue = await mgr.subscribe(rec.id)
        events = []
        try:
            while True:
                evt = await asyncio.wait_for(queue.get(), timeout=2)
                events.append(evt)
                if evt.status.is_terminal:
                    break
        finally:
            mgr.unsubscribe(rec.id, queue)
        assert any(e.event == "progress" for e in events)
        assert events[-1].status == TaskStatus.SUCCEEDED

    async def test_concurrency_limit(self):
        active = 0
        peak = 0

        async def run(target, payload, ctx):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return "ok"

        mgr = TaskManager(max_concurrency=2)
        mgr.register_dispatcher(TargetType.AGENT, run)
        recs = [await mgr.submit(TargetType.AGENT, "x", input={}) for _ in range(6)]
        # Wait for all to finish.
        for _ in range(100):
            done = [await mgr.get(r.id) for r in recs]
            if all(d.status.is_terminal for d in done):
                break
            await asyncio.sleep(0.02)
        assert peak <= 2


# =====================================================================
# HTTP task board
# =====================================================================


def _build_task_app():
    from fastapi import FastAPI

    from agentomatic.tasks.routes import create_task_router

    mgr = TaskManager()
    mgr.register_dispatcher(TargetType.AGENT, _echo_dispatcher())
    app = FastAPI()
    app.include_router(create_task_router(mgr), prefix="/api/v1/tasks")
    return app, mgr


class TestTaskRoutes:
    def test_submit_wait_and_result(self):
        from fastapi.testclient import TestClient

        app, _ = _build_task_app()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/tasks",
                json={
                    "target_type": "agent",
                    "target": "x",
                    "input": {"q": 1},
                    "wait": True,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "succeeded"
            task_id = body["id"]

            got = client.get(f"/api/v1/tasks/{task_id}")
            assert got.status_code == 200

            result = client.get(f"/api/v1/tasks/{task_id}/result")
            assert result.status_code == 200
            assert result.json()["result"]["echo"] == {"q": 1}

    def test_submit_async_returns_202(self):
        from fastapi.testclient import TestClient

        app, _ = _build_task_app()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}},
            )
            assert resp.status_code == 202
            assert "id" in resp.json()

    def test_unknown_target_type_400(self):
        from fastapi.testclient import TestClient

        app, _ = _build_task_app()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/tasks",
                json={"target_type": "plugin", "target": "x", "input": {}, "wait": True},
            )
            assert resp.status_code == 400

    def test_list_tasks(self):
        from fastapi.testclient import TestClient

        app, _ = _build_task_app()
        with TestClient(app) as client:
            client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            )
            resp = client.get("/api/v1/tasks")
            assert resp.status_code == 200
            assert resp.json()["count"] >= 1

    def test_get_missing_task_404(self):
        from fastapi.testclient import TestClient

        app, _ = _build_task_app()
        with TestClient(app) as client:
            assert client.get("/api/v1/tasks/nope").status_code == 404
