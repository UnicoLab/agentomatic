# pyright: reportMissingParameterType=none
"""Tests for resumable task progress streams.

Task events used to exist only while somebody was listening: ``_emit``
returned early when a task had no subscribers, so everything emitted while a
client was disconnected was never created, let alone retained. A dropped
connection could therefore only be recovered as far as the task's *current*
state — every intermediate step was gone.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentomatic.tasks.event_log import (
    InMemoryTaskEventLog,
    NullTaskEventLog,
    available_event_log_providers,
    create_event_log,
    estimate_event_size,
    event_log_from_env,
    parse_last_event_id,
    register_event_log_provider,
    resolve_resumption,
)
from agentomatic.tasks.manager import TaskManager
from agentomatic.tasks.models import TargetType, TaskEvent, TaskStatus
from agentomatic.tasks.routes import create_task_router


def _stepping_dispatcher(steps: int = 5, delay: float = 0.02):
    """Return a dispatcher that reports ``steps`` progress events."""

    async def dispatch(target: str, payload: Any, ctx: Any) -> dict[str, Any]:
        for i in range(1, steps + 1):
            await ctx.report(
                percent=i * (100 / steps), message=f"step {i}", current=i, total=steps
            )
            await asyncio.sleep(delay)
        return {"steps": steps}

    return dispatch


def _build_app(manager: TaskManager) -> FastAPI:
    """Mount the task router for ``manager`` on a bare app."""
    app = FastAPI()
    app.include_router(create_task_router(manager), prefix="/api/v1/tasks")
    return app


def _frames(body: str) -> list[dict[str, Any]]:
    """Parse an SSE response body into its JSON data frames."""
    out: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[len("data: ") :]))
    return out


def _ids(body: str) -> list[int]:
    """Extract the SSE ``id:`` values from a response body."""
    return [int(line[len("id: ") :]) for line in body.splitlines() if line.startswith("id: ")]


# =====================================================================
# Sequencing
# =====================================================================


class TestEventSequencing:
    @pytest.mark.asyncio
    async def test_events_are_recorded_without_any_subscriber(self) -> None:
        """The whole point: nobody listening must not mean nothing retained."""
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(3))

        record = await manager.submit(TargetType.AGENT, "x", input={}, mode="async")
        await asyncio.sleep(0.4)

        events = await manager.event_log.replay(record.id)
        assert events, "no events retained for an unobserved task"
        assert events[-1].status.is_terminal

    @pytest.mark.asyncio
    async def test_sequences_are_gapless_and_start_at_one(self) -> None:
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(4))

        record = await manager.submit(TargetType.AGENT, "x", input={}, mode="async")
        await asyncio.sleep(0.5)

        sequences = [e.sequence for e in await manager.event_log.replay(record.id)]
        assert sequences == list(range(1, len(sequences) + 1))

    @pytest.mark.asyncio
    async def test_sequences_are_independent_per_task(self) -> None:
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(2))

        first = await manager.submit(TargetType.AGENT, "x", input={}, mode="async")
        second = await manager.submit(TargetType.AGENT, "y", input={}, mode="async")
        await asyncio.sleep(0.5)

        for record in (first, second):
            sequences = [e.sequence for e in await manager.event_log.replay(record.id)]
            assert sequences[0] == 1, "each task numbers its own events from 1"

    @pytest.mark.asyncio
    async def test_reconnect_replays_exactly_the_gap(self) -> None:
        """No missed event, and nothing delivered twice."""
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(6, delay=0.03))

        record = await manager.submit(TargetType.AGENT, "x", input={}, mode="async")
        await asyncio.sleep(0.1)

        seen = [e.sequence for e in await manager.event_log.replay(record.id)]
        cursor = seen[-1]
        await asyncio.sleep(0.6)

        resumed = await manager.resume(record.id, after=cursor)
        replayed = [e.sequence for e in resumed.replay]
        everything = [e.sequence for e in await manager.event_log.replay(record.id)]

        assert seen + replayed == everything
        assert not resumed.truncated


# =====================================================================
# The SSE surface
# =====================================================================


class TestResumableSSE:
    def test_new_subscriber_gets_a_snapshot_carrying_a_cursor(self) -> None:
        """Without an id on the snapshot, a client that drops straight after it
        has nothing to resume from and must restart the whole stream."""
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(3))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            ).json()
            task_id = submitted["id"]
            body = client.get(f"/api/v1/tasks/{task_id}/events").text

        ids = _ids(body)
        assert ids, "no id: lines — EventSource cannot resume without them"
        assert ids[-1] == asyncio.run(manager.event_log.latest_sequence(task_id))
        assert _frames(body)[-1]["id"] == task_id, "a new client gets the full record"

    def test_since_replays_exactly_the_missed_events(self) -> None:
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(4))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            ).json()
            task_id = submitted["id"]
            resumed = client.get(f"/api/v1/tasks/{task_id}/events?since=2").text

        recorded = [e.sequence for e in asyncio.run(manager.event_log.replay(task_id))]
        assert _ids(resumed) == [s for s in recorded if s > 2]
        assert all(seq > 2 for seq in _ids(resumed)), "re-sent already-seen events"

    def test_last_event_id_header_matches_the_since_parameter(self) -> None:
        """A browser EventSource resumes with this header and no query param."""
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(4))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            ).json()
            task_id = submitted["id"]

            by_query = _ids(client.get(f"/api/v1/tasks/{task_id}/events?since=2").text)
            by_header = _ids(
                client.get(
                    f"/api/v1/tasks/{task_id}/events",
                    headers={"Last-Event-ID": "2"},
                ).text
            )

        assert by_header == by_query
        assert by_header, "the header was ignored"

    def test_malformed_last_event_id_is_treated_as_a_new_subscriber(self) -> None:
        """It must never be read as a valid cursor that silently skips events."""
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(2))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            ).json()
            task_id = submitted["id"]
            body = client.get(
                f"/api/v1/tasks/{task_id}/events",
                headers={"Last-Event-ID": "not-a-number"},
            ).text
            fresh = client.get(f"/api/v1/tasks/{task_id}/events").text

        assert _ids(body) == _ids(fresh)
        # A new subscriber's snapshot is the complete current state, so nothing
        # is lost by discarding the unusable cursor.
        assert _frames(body)[-1]["status"] == "succeeded"

    def test_truncated_history_is_announced(self) -> None:
        """Silently serving a partial history would be the dangerous failure."""
        manager = TaskManager(event_log=InMemoryTaskEventLog(max_events_per_task=2))
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(5))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            ).json()
            body = client.get(f"/api/v1/tasks/{submitted['id']}/events?since=1").text

        assert any(f.get("event") == "truncated" for f in _frames(body))

    def test_terminal_task_stream_closes_with_done(self) -> None:
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(2))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            ).json()
            body = client.get(f"/api/v1/tasks/{submitted['id']}/events").text

        assert body.rstrip().endswith("data: [DONE]")

    def test_unknown_task_still_404s(self) -> None:
        manager = TaskManager()
        with TestClient(_build_app(manager)) as client:
            assert client.get("/api/v1/tasks/nope/events").status_code == 404

    def test_deleting_a_task_forgets_its_events(self) -> None:
        manager = TaskManager()
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(2))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={"target_type": "agent", "target": "x", "input": {}, "wait": True},
            ).json()
            task_id = submitted["id"]
            assert client.delete(f"/api/v1/tasks/{task_id}").status_code == 200

            assert client.get(f"/api/v1/tasks/{task_id}/events").status_code == 404


# =====================================================================
# The log itself
# =====================================================================


def _event(task_id: str, sequence: int) -> TaskEvent:
    """Build a minimal event for log-level tests."""
    return TaskEvent(
        task_id=task_id, sequence=sequence, event="progress", status=TaskStatus.RUNNING
    )


class TestInMemoryEventLog:
    @pytest.mark.asyncio
    async def test_replay_filters_by_sequence(self) -> None:
        log = InMemoryTaskEventLog()
        for seq in range(1, 5):
            await log.append(_event("t", seq))
        assert [e.sequence for e in await log.replay("t", after=2)] == [3, 4]

    @pytest.mark.asyncio
    async def test_per_task_retention_is_bounded(self) -> None:
        log = InMemoryTaskEventLog(max_events_per_task=3)
        for seq in range(1, 11):
            await log.append(_event("t", seq))
        assert [e.sequence for e in await log.replay("t")] == [8, 9, 10]

    @pytest.mark.asyncio
    async def test_task_count_is_bounded(self) -> None:
        log = InMemoryTaskEventLog(max_tasks=2)
        for name in ("a", "b", "c"):
            await log.append(_event(name, 1))
        assert await log.replay("a") == [], "oldest task should have been evicted"
        assert await log.replay("c")

    @pytest.mark.asyncio
    async def test_drop_forgets_one_task_only(self) -> None:
        log = InMemoryTaskEventLog()
        await log.append(_event("a", 1))
        await log.append(_event("b", 1))
        await log.drop("a")
        assert await log.replay("a") == []
        assert await log.replay("b")

    @pytest.mark.asyncio
    async def test_earliest_sequence_reports_retention_floor(self) -> None:
        log = InMemoryTaskEventLog(max_events_per_task=2)
        for seq in range(1, 6):
            await log.append(_event("t", seq))
        assert await log.earliest_sequence("t") == 4
        assert await log.earliest_sequence("missing") == 0


class TestResumptionResolution:
    @pytest.mark.asyncio
    async def test_new_client_gets_no_replay(self) -> None:
        log = InMemoryTaskEventLog()
        await log.append(_event("t", 1))
        resumed = await resolve_resumption(log, "t", after=0)
        assert resumed.replay == []
        assert not resumed.truncated

    @pytest.mark.asyncio
    async def test_evicted_gap_is_flagged_truncated(self) -> None:
        log = InMemoryTaskEventLog(max_events_per_task=2)
        for seq in range(1, 6):
            await log.append(_event("t", seq))
        resumed = await resolve_resumption(log, "t", after=1)
        assert resumed.truncated

    @pytest.mark.asyncio
    async def test_contiguous_history_is_not_truncated(self) -> None:
        log = InMemoryTaskEventLog()
        for seq in range(1, 6):
            await log.append(_event("t", seq))
        resumed = await resolve_resumption(log, "t", after=3)
        assert not resumed.truncated
        assert resumed.cursor == 5


class TestNullEventLog:
    @pytest.mark.asyncio
    async def test_retains_nothing_but_never_fails(self) -> None:
        manager = TaskManager(event_log=NullTaskEventLog())
        manager.register_dispatcher(TargetType.AGENT, _stepping_dispatcher(2))
        record = await manager.submit(TargetType.AGENT, "x", input={}, mode="async")
        await asyncio.sleep(0.3)
        assert await manager.event_log.replay(record.id) == []


class TestProviderRegistry:
    def test_builtin_providers_resolve(self) -> None:
        assert isinstance(create_event_log("memory"), InMemoryTaskEventLog)
        assert isinstance(create_event_log("null"), NullTaskEventLog)

    def test_unknown_provider_names_the_registry(self) -> None:
        with pytest.raises(ValueError, match="register_event_log_provider"):
            create_event_log("redis")

    def test_custom_provider_can_be_registered(self) -> None:
        register_event_log_provider("test-shared", lambda **_: NullTaskEventLog())
        assert "test-shared" in available_event_log_providers()
        assert isinstance(create_event_log("test-shared"), NullTaskEventLog)


class TestLastEventIdParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("5", 5), ("  7 ", 7), (None, 0), ("", 0), ("abc", 0), ("-3", 0), ("0", 0)],
    )
    def test_parses_defensively(self, raw: str | None, expected: int) -> None:
        assert parse_last_event_id(raw) == expected


# =====================================================================
# Coverage across every resource type
# =====================================================================


class TestEveryTargetTypeResumes:
    """Agents, pipelines, plugins, endpoints and ingestion share one task
    manager, so the resumable stream must work identically for all of them —
    a pipeline's per-step progress is exactly the case where losing the
    connection hurts most."""

    @pytest.mark.parametrize("target_type", list(TargetType))
    def test_stream_is_resumable(self, target_type: TargetType) -> None:
        manager = TaskManager()
        manager.register_dispatcher(target_type, _stepping_dispatcher(4))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={
                    "target_type": target_type.value,
                    "target": f"{target_type.value}-x",
                    "input": {},
                    "wait": True,
                },
            ).json()
            task_id = submitted["id"]
            resumed = client.get(f"/api/v1/tasks/{task_id}/events?since=2").text

        recorded = [e.sequence for e in asyncio.run(manager.event_log.replay(task_id))]
        assert recorded == list(range(1, len(recorded) + 1)), "gapless sequence required"
        assert _ids(resumed) == [s for s in recorded if s > 2]

    @pytest.mark.parametrize("target_type", list(TargetType))
    def test_events_survive_having_no_listener(self, target_type: TargetType) -> None:
        manager = TaskManager()
        manager.register_dispatcher(target_type, _stepping_dispatcher(3))
        with TestClient(_build_app(manager)) as client:
            submitted = client.post(
                "/api/v1/tasks",
                json={
                    "target_type": target_type.value,
                    "target": f"{target_type.value}-x",
                    "input": {},
                    "wait": True,
                },
            ).json()

        # Nothing ever subscribed, yet the history is there to replay.
        assert asyncio.run(manager.event_log.replay(submitted["id"]))


class TestEventLogMemoryIsBounded:
    """A progress event can carry a pipeline checkpoint's ``sub_result``, so
    capping the number of events is not the same as capping memory."""

    @pytest.mark.asyncio
    async def test_total_bytes_stay_under_budget(self) -> None:
        log = InMemoryTaskEventLog(max_bytes_total=200_000)
        payload = {"blob": "x" * 10_000}
        for task in range(40):
            for seq in range(1, 21):
                await log.append(
                    TaskEvent(
                        task_id=f"t{task}",
                        sequence=seq,
                        event="progress",
                        status=TaskStatus.RUNNING,
                        data=payload,
                    )
                )

        assert (await log.stats())["bytes"] <= 200_000

    @pytest.mark.asyncio
    async def test_eviction_keeps_the_accounting_honest(self) -> None:
        log = InMemoryTaskEventLog(max_events_per_task=5)
        for seq in range(1, 201):
            await log.append(_event("t", seq))

        stats = await log.stats()
        assert stats["events"] == 5
        assert stats["bytes"] == sum(estimate_event_size(evt) for evt in await log.replay("t"))

    @pytest.mark.asyncio
    async def test_dropping_a_task_reclaims_its_bytes(self) -> None:
        log = InMemoryTaskEventLog()
        await log.append(_event("a", 1))
        await log.append(_event("b", 1))
        before = (await log.stats())["bytes"]

        await log.drop("a")

        assert (await log.stats())["bytes"] < before

    @pytest.mark.asyncio
    async def test_sequences_stay_gapless_through_eviction(self) -> None:
        log = InMemoryTaskEventLog(max_events_per_task=4)
        for seq in range(1, 21):
            await log.append(_event("t", seq))

        assert [e.sequence for e in await log.replay("t")] == [17, 18, 19, 20]

    @pytest.mark.asyncio
    async def test_empty_data_costs_only_the_fixed_overhead(self) -> None:
        """The common case must not pay for JSON serialisation."""
        assert estimate_event_size(_event("t", 1)) == 512

    @pytest.mark.asyncio
    async def test_unserialisable_data_does_not_raise(self) -> None:
        event = _event("t", 1)
        event.data = {"obj": object()}
        assert estimate_event_size(event) >= 512


class TestEnvironmentConfiguration:
    """Retention is an operational concern: an operator sizing a container has
    to be able to cap or disable it without editing Python."""

    def test_defaults_when_nothing_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "AGENTOMATIC_TASK_EVENT_LOG",
            "AGENTOMATIC_TASK_EVENTS_PER_TASK",
            "AGENTOMATIC_TASK_EVENT_TASKS",
            "AGENTOMATIC_TASK_EVENT_MAX_MB",
        ):
            monkeypatch.delenv(var, raising=False)
        assert isinstance(event_log_from_env(), InMemoryTaskEventLog)

    def test_replay_can_be_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTOMATIC_TASK_EVENT_LOG", "none")
        assert isinstance(event_log_from_env(), NullTaskEventLog)

    @pytest.mark.asyncio
    async def test_limits_are_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTOMATIC_TASK_EVENT_LOG", "memory")
        monkeypatch.setenv("AGENTOMATIC_TASK_EVENTS_PER_TASK", "3")
        log = event_log_from_env()
        for seq in range(1, 11):
            await log.append(_event("t", seq))
        assert [e.sequence for e in await log.replay("t")] == [8, 9, 10]

    def test_a_bad_value_falls_back_instead_of_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo in a container's env must not stop the platform booting."""
        monkeypatch.setenv("AGENTOMATIC_TASK_EVENTS_PER_TASK", "not-a-number")
        assert isinstance(event_log_from_env(), InMemoryTaskEventLog)

    def test_an_unknown_provider_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTOMATIC_TASK_EVENT_LOG", "redis-typo")
        assert isinstance(event_log_from_env(), InMemoryTaskEventLog)
