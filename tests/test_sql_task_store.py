"""Tests for the durable SQLAlchemy-backed TaskStore."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sqlalchemy")

from agentomatic.tasks import SQLAlchemyTaskStore
from agentomatic.tasks.models import TargetType, TaskRecord, TaskStatus


def _record(
    task_id: str,
    *,
    status: TaskStatus = TaskStatus.QUEUED,
    target: str = "researcher",
    target_type: TargetType = TargetType.AGENT,
    finished_at: float | None = None,
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        target_type=target_type,
        target=target,
        status=status,
        input={"query": "hi"},
        finished_at=finished_at,
    )


async def _store(tmp_path, **kwargs) -> SQLAlchemyTaskStore:
    store = SQLAlchemyTaskStore(f"sqlite+aiosqlite:///{tmp_path}/tasks.db", **kwargs)
    await store.initialize()
    return store


class TestCrud:
    @pytest.mark.asyncio
    async def test_save_and_get(self, tmp_path):
        store = await _store(tmp_path)
        rec = _record("task_1")
        await store.save(rec)

        loaded = await store.get("task_1")
        assert loaded is not None
        assert loaded.id == "task_1"
        assert loaded.target == "researcher"
        assert loaded.status == TaskStatus.QUEUED
        assert loaded.input == {"query": "hi"}
        await store.close()

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, tmp_path):
        store = await _store(tmp_path)
        assert await store.get("nope") is None
        await store.close()

    @pytest.mark.asyncio
    async def test_save_is_upsert(self, tmp_path):
        store = await _store(tmp_path)
        await store.save(_record("t", status=TaskStatus.QUEUED))
        rec = _record("t", status=TaskStatus.SUCCEEDED)
        rec.result = {"answer": 42}
        await store.save(rec)

        loaded = await store.get("t")
        assert loaded.status == TaskStatus.SUCCEEDED
        assert loaded.result == {"answer": 42}
        assert await store.count() == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        store = await _store(tmp_path)
        await store.save(_record("t"))
        assert await store.delete("t") is True
        assert await store.delete("t") is False
        assert await store.get("t") is None
        await store.close()


class TestListAndCount:
    @pytest.mark.asyncio
    async def test_list_filters(self, tmp_path):
        store = await _store(tmp_path)
        await store.save(_record("a", status=TaskStatus.RUNNING, target="r1"))
        await store.save(_record("b", status=TaskStatus.SUCCEEDED, target="r2"))
        await store.save(
            _record("c", status=TaskStatus.SUCCEEDED, target_type=TargetType.PIPELINE)
        )

        assert len(await store.list()) == 3
        succeeded = await store.list(status=TaskStatus.SUCCEEDED)
        assert {r.id for r in succeeded} == {"b", "c"}
        pipelines = await store.list(target_type=TargetType.PIPELINE)
        assert {r.id for r in pipelines} == {"c"}
        by_target = await store.list(target="r1")
        assert {r.id for r in by_target} == {"a"}
        await store.close()

    @pytest.mark.asyncio
    async def test_count(self, tmp_path):
        store = await _store(tmp_path)
        await store.save(_record("a", status=TaskStatus.RUNNING))
        await store.save(_record("b", status=TaskStatus.SUCCEEDED))
        assert await store.count() == 2
        assert await store.count(status=TaskStatus.SUCCEEDED) == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_list_pagination_order(self, tmp_path):
        store = await _store(tmp_path)
        for i in range(5):
            rec = _record(f"t{i}")
            rec.created_at = 1000.0 + i
            await store.save(rec)
        page = await store.list(limit=2, offset=0)
        # Newest first
        assert [r.id for r in page] == ["t4", "t3"]
        await store.close()


class TestDurability:
    @pytest.mark.asyncio
    async def test_survives_new_store_instance(self, tmp_path):
        store = await _store(tmp_path)
        await store.save(_record("persisted", status=TaskStatus.SUCCEEDED))
        await store.close()

        # A brand-new store pointed at the same DB file sees the record.
        store2 = await _store(tmp_path)
        loaded = await store2.get("persisted")
        assert loaded is not None
        assert loaded.status == TaskStatus.SUCCEEDED
        await store2.close()


class TestEviction:
    @pytest.mark.asyncio
    async def test_purge_expired_ttl(self, tmp_path):
        store = await _store(tmp_path, ttl_seconds=10)
        old = _record("old", status=TaskStatus.SUCCEEDED, finished_at=time.time() - 100)
        fresh = _record("fresh", status=TaskStatus.SUCCEEDED, finished_at=time.time())
        running = _record("running", status=TaskStatus.RUNNING)
        await store.save(old)
        await store.save(fresh)
        await store.save(running)

        deleted = await store.purge_expired()
        assert deleted == 1
        assert await store.get("old") is None
        assert await store.get("fresh") is not None
        assert await store.get("running") is not None
        await store.close()

    @pytest.mark.asyncio
    async def test_purge_overflow(self, tmp_path):
        store = await _store(tmp_path, ttl_seconds=None, max_records=2)
        for i in range(4):
            rec = _record(f"t{i}", status=TaskStatus.SUCCEEDED, finished_at=time.time())
            rec.created_at = 1000.0 + i
            await store.save(rec)
        await store.purge_expired()
        assert await store.count() == 2
        # Oldest evicted first
        assert await store.get("t0") is None
        assert await store.get("t3") is not None
        await store.close()


class TestManagerIntegration:
    @pytest.mark.asyncio
    async def test_task_manager_with_sql_store(self, tmp_path):
        from agentomatic.tasks import TaskManager
        from agentomatic.tasks.context import TaskContext

        store = await _store(tmp_path)
        manager = TaskManager(store=store)

        async def dispatcher(target: str, payload, ctx: TaskContext):
            return {"echo": payload}

        manager.register_dispatcher(TargetType.AGENT, dispatcher)

        record = await manager.submit(TargetType.AGENT, "echo", input={"x": 1})
        # Poll until terminal
        for _ in range(50):
            current = await manager.get(record.id)
            if current and current.status.is_terminal:
                break
            import asyncio

            await asyncio.sleep(0.02)

        final = await manager.get(record.id)
        assert final is not None
        assert final.status == TaskStatus.SUCCEEDED
        assert final.result == {"echo": {"x": 1}}
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_health_check(self, tmp_path):
        store = await _store(tmp_path)
        health = await store.health_check()
        assert health["status"] == "healthy"
        assert health["backend"] == "SQLAlchemyTaskStore"
        await store.close()
