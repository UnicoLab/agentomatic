# pyright: reportMissingParameterType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for replaying a dropped ``/invoke/stream`` response.

Frames used to be written straight to the socket and kept nowhere, so a
client that lost the connection mid-answer lost the partial answer with it.
They are now numbered and retained under the ``X-Stream-Id`` the response
carries, so the produced frames can be collected afterwards.

Replay returns what was *produced*; it does not restart a run cancelled with
the request. The durable path is a task — see ``test_task_event_resume.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.core.manifest import AgentManifest
from agentomatic.streaming import (
    StreamReplayBuffer,
    buffer_from_env,
    get_replay_buffer,
    new_stream_id,
    set_replay_buffer,
    sse_frame,
)

BASE = "/api/v1"


async def _echo_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Return a canned response for the incoming query."""
    return {"response": f"echo: {state.get('current_query', '')}"}


@pytest.fixture(autouse=True)
def _fresh_buffer():
    """Isolate each test from the process-wide buffer."""
    set_replay_buffer(StreamReplayBuffer())
    yield
    set_replay_buffer(None)


@pytest.fixture
def client() -> Any:
    """A platform exposing one streaming agent."""
    platform = AgentPlatform(
        agents_dir="/tmp/agentomatic_stream_replay_test",
        title="Stream Replay Test",
        version="0.0.1",
    )
    platform.register_agent(
        manifest=AgentManifest(name="echo", slug="fn-echo", description="Echo"),
        node_fn=_echo_fn,
    )
    with TestClient(platform.build()) as test_client:
        yield test_client


def _ids(body: str) -> list[int]:
    """Extract SSE ``id:`` values from a body."""
    return [int(line[len("id: ") :]) for line in body.splitlines() if line.startswith("id: ")]


def _data(body: str) -> list[str]:
    """Extract SSE ``data:`` payloads from a body."""
    return [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]


class TestInvokeStreamIsReplayable:
    def test_response_advertises_a_stream_id(self, client: Any) -> None:
        response = client.post(f"{BASE}/echo/invoke/stream", json={"query": "hi"})
        assert response.status_code == 200
        assert response.headers.get("X-Stream-Id", "").startswith("stream_")

    def test_frames_are_numbered(self, client: Any) -> None:
        response = client.post(f"{BASE}/echo/invoke/stream", json={"query": "hi"})
        sequences = _ids(response.text)
        assert sequences == list(range(1, len(sequences) + 1)), "gapless numbering required"

    def test_replay_returns_the_produced_frames(self, client: Any) -> None:
        original = client.post(f"{BASE}/echo/invoke/stream", json={"query": "hi"})
        stream_id = original.headers["X-Stream-Id"]

        replayed = client.get(f"{BASE}/echo/invoke/stream/{stream_id}")

        assert replayed.status_code == 200
        assert _data(replayed.text) == _data(original.text)
        assert _ids(replayed.text) == _ids(original.text)

    def test_replay_honours_since(self, client: Any) -> None:
        original = client.post(f"{BASE}/echo/invoke/stream", json={"query": "hi"})
        stream_id = original.headers["X-Stream-Id"]

        replayed = client.get(f"{BASE}/echo/invoke/stream/{stream_id}?since=1")

        assert all(seq > 1 for seq in _ids(replayed.text))
        assert _ids(replayed.text) == [s for s in _ids(original.text) if s > 1]

    def test_replay_honours_last_event_id(self, client: Any) -> None:
        original = client.post(f"{BASE}/echo/invoke/stream", json={"query": "hi"})
        stream_id = original.headers["X-Stream-Id"]

        by_query = client.get(f"{BASE}/echo/invoke/stream/{stream_id}?since=1").text
        by_header = client.get(
            f"{BASE}/echo/invoke/stream/{stream_id}",
            headers={"Last-Event-ID": "1"},
        ).text

        assert _ids(by_header) == _ids(by_query)

    def test_terminal_marker_is_retained(self, client: Any) -> None:
        """The [DONE] marker must survive replay, or a client never stops."""
        original = client.post(f"{BASE}/echo/invoke/stream", json={"query": "hi"})
        stream_id = original.headers["X-Stream-Id"]
        replayed = client.get(f"{BASE}/echo/invoke/stream/{stream_id}").text
        assert "[DONE]" in replayed

    def test_unknown_stream_404s(self, client: Any) -> None:
        response = client.get(f"{BASE}/echo/invoke/stream/stream_deadbeef")
        assert response.status_code == 404

    def test_each_invocation_gets_its_own_stream(self, client: Any) -> None:
        first = client.post(f"{BASE}/echo/invoke/stream", json={"query": "one"})
        second = client.post(f"{BASE}/echo/invoke/stream", json={"query": "two"})
        assert first.headers["X-Stream-Id"] != second.headers["X-Stream-Id"]

        replayed = client.get(f"{BASE}/echo/invoke/stream/{first.headers['X-Stream-Id']}").text
        assert "one" in replayed
        assert "two" not in replayed


class TestReplayBuffer:
    @pytest.mark.asyncio
    async def test_record_assigns_gapless_sequences(self) -> None:
        buffer = StreamReplayBuffer()
        sequences = [await buffer.record("s", f"frame-{i}") for i in range(5)]
        assert sequences == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_replay_filters_by_sequence(self) -> None:
        buffer = StreamReplayBuffer()
        for i in range(1, 5):
            await buffer.record("s", f"f{i}")
        assert [f.sequence for f in await buffer.replay("s", after=2)] == [3, 4]

    @pytest.mark.asyncio
    async def test_frames_per_stream_are_bounded(self) -> None:
        buffer = StreamReplayBuffer(max_frames_per_stream=3)
        for i in range(1, 8):
            await buffer.record("s", f"f{i}")
        assert [f.sequence for f in await buffer.replay("s")] == [5, 6, 7]

    @pytest.mark.asyncio
    async def test_stream_count_is_bounded(self) -> None:
        buffer = StreamReplayBuffer(max_streams=2)
        for name in ("a", "b", "c"):
            await buffer.record(name, "f")
        assert not await buffer.knows("a"), "oldest stream should be evicted"
        assert await buffer.knows("c")

    @pytest.mark.asyncio
    async def test_drop_forgets_one_stream_only(self) -> None:
        buffer = StreamReplayBuffer()
        await buffer.record("a", "f")
        await buffer.record("b", "f")
        await buffer.drop("a")
        assert not await buffer.knows("a")
        assert await buffer.knows("b")

    @pytest.mark.asyncio
    async def test_earliest_sequence_reports_the_retention_floor(self) -> None:
        buffer = StreamReplayBuffer(max_frames_per_stream=2)
        for i in range(1, 6):
            await buffer.record("s", f"f{i}")
        assert await buffer.earliest_sequence("s") == 4
        assert await buffer.earliest_sequence("missing") == 0

    @pytest.mark.asyncio
    async def test_streams_number_independently(self) -> None:
        buffer = StreamReplayBuffer()
        await buffer.record("a", "f")
        assert await buffer.record("b", "f") == 1, "streams must not share a counter"

    @pytest.mark.asyncio
    async def test_concurrent_records_do_not_collide(self) -> None:
        buffer = StreamReplayBuffer()
        sequences = await asyncio.gather(*(buffer.record("s", f"f{i}") for i in range(50)))
        assert sorted(sequences) == list(range(1, 51))


class TestSseEncoding:
    def test_omits_id_when_unnumbered(self) -> None:
        assert sse_frame({"a": 1}) == 'data: {"a": 1}\n\n'

    def test_includes_id_when_numbered(self) -> None:
        assert sse_frame({"a": 1}, event_id=7).startswith("id: 7\n")

    def test_passes_through_pre_rendered_strings(self) -> None:
        assert sse_frame("[DONE]") == "data: [DONE]\n\n"

    def test_stream_ids_are_unique(self) -> None:
        assert len({new_stream_id() for _ in range(100)}) == 100

    def test_default_buffer_is_shared(self) -> None:
        assert get_replay_buffer() is get_replay_buffer()


class TestPublicStreamingHelper:
    """``create_streaming_response`` is exported for user-built endpoints, so
    it should be able to opt into the same numbering and retention."""

    @pytest.mark.asyncio
    async def test_without_a_stream_id_nothing_is_retained(self) -> None:
        from agentomatic.protocols import create_streaming_response

        async def frames():
            yield "data: one\n\n"

        response = create_streaming_response(frames(), agent_name="a")
        assert "X-Stream-Id" not in response.headers
        body = "".join([chunk async for chunk in response.body_iterator])
        assert "id:" not in body

    @pytest.mark.asyncio
    async def test_a_stream_id_numbers_and_retains(self) -> None:
        from agentomatic.protocols import create_streaming_response

        async def frames():
            yield "data: one\n\n"
            yield "data: two\n\n"

        stream_id = new_stream_id()
        response = create_streaming_response(frames(), agent_name="a", stream_id=stream_id)
        assert response.headers["X-Stream-Id"] == stream_id

        body = "".join([chunk async for chunk in response.body_iterator])
        assert "id: 1" in body
        assert "id: 2" in body

        retained = await get_replay_buffer().replay(stream_id)
        assert [frame.data for frame in retained] == ["one", "two"]

    @pytest.mark.asyncio
    async def test_non_data_chunks_pass_through_unnumbered(self) -> None:
        """Keep-alive comments must not be numbered or retained as frames."""
        from agentomatic.protocols import create_streaming_response

        async def frames():
            yield ": keep-alive\n\n"
            yield "data: real\n\n"

        stream_id = new_stream_id()
        response = create_streaming_response(frames(), stream_id=stream_id)
        body = "".join([chunk async for chunk in response.body_iterator])

        assert ": keep-alive" in body
        retained = await get_replay_buffer().replay(stream_id)
        assert [frame.data for frame in retained] == ["real"]


class TestStudioStreamsAreNumbered:
    """Studio builds its frames inside ``run_tracker``, so the numbering is
    applied by wrapping the generator at the router. Worth asserting directly:
    a refactor that stops wrapping would silently drop the ids."""

    @pytest.fixture
    def studio_client(self) -> Any:
        platform = AgentPlatform(
            agents_dir="/tmp/agentomatic_studio_numbering_test",
            title="Studio Numbering",
            version="0.0.1",
            enable_studio=True,
        )
        platform.register_agent(
            manifest=AgentManifest(name="echo", slug="fn-echo", description="Echo"),
            node_fn=_echo_fn,
        )
        with TestClient(platform.build()) as test_client:
            yield test_client

    def test_run_stream_frames_carry_ids(self, studio_client: Any) -> None:
        response = studio_client.post(
            "/studio/agents/echo/runs/stream", json={"input": {"query": "hi"}}
        )
        assert response.status_code == 200
        sequences = _ids(response.text)
        assert sequences == list(range(1, len(sequences) + 1))

    def test_run_stream_frames_are_retained(self, studio_client: Any) -> None:
        response = studio_client.post(
            "/studio/agents/echo/runs/stream", json={"input": {"query": "hi"}}
        )
        run_id = response.headers["X-Studio-Run-Id"]
        retained = asyncio.run(get_replay_buffer().replay(run_id))
        assert retained, "studio frames were not retained under the run id"
        assert [frame.sequence for frame in retained] == _ids(response.text)


class TestNewRoutesRequireAuth:
    """Auth is middleware with a skip-list, so a new route is protected by
    default — but only until someone adds a path to that list. A replay route
    that stops requiring a key hands out other callers' responses."""

    @pytest.fixture
    def secured(self) -> Any:
        platform = AgentPlatform(
            agents_dir="/tmp/agentomatic_stream_auth_test",
            title="Secured",
            version="0.0.1",
            enable_auth=True,
            auth_api_key="secret",
            enable_tasks=True,
        )
        platform.register_agent(
            manifest=AgentManifest(name="echo", slug="fn-echo", description="Echo"),
            node_fn=_echo_fn,
        )
        with TestClient(platform.build()) as test_client:
            yield test_client

    def test_stream_replay_requires_a_key(self, secured: Any) -> None:
        header = {"X-API-Key": "secret"}
        stream_id = secured.post(
            f"{BASE}/echo/invoke/stream", json={"query": "hi"}, headers=header
        ).headers["X-Stream-Id"]

        assert secured.get(f"{BASE}/echo/invoke/stream/{stream_id}").status_code == 401
        assert (
            secured.get(f"{BASE}/echo/invoke/stream/{stream_id}", headers=header).status_code
            == 200
        )

    def test_a2a_event_stream_requires_a_key(self, secured: Any) -> None:
        header = {"X-API-Key": "secret"}
        task_id = secured.post(
            f"{BASE}/echo/a2a/tasks",
            json={"message": {"content": "hi"}},
            headers=header,
        ).json()["task_id"]

        assert secured.get(f"{BASE}/echo/a2a/tasks/{task_id}/events").status_code == 401
        assert (
            secured.get(f"{BASE}/echo/a2a/tasks/{task_id}/events", headers=header).status_code
            == 200
        )

    def test_task_event_stream_requires_a_key(self, secured: Any) -> None:
        header = {"X-API-Key": "secret"}
        task_id = secured.post(
            f"{BASE}/echo/a2a/tasks",
            json={"message": {"content": "hi"}},
            headers=header,
        ).json()["task_id"]

        assert secured.get(f"{BASE}/tasks/{task_id}/events").status_code == 401

    def test_stream_ids_are_not_guessable(self) -> None:
        """The id is the only thing standing between a caller and a retained
        response, so it must carry real entropy."""
        ids = {new_stream_id() for _ in range(500)}
        assert len(ids) == 500
        # uuid4 hex[:16] — 64 bits.
        assert all(len(sid.removeprefix("stream_")) == 16 for sid in ids)


class TestMemoryIsBoundedByBytes:
    """Count limits are not memory limits. 512 frames across 256 streams is a
    few hundred megabytes of perfectly ordinary traffic, and one long
    generated answer per frame pushes it past a container's limit."""

    @pytest.mark.asyncio
    async def test_total_bytes_stay_under_budget(self) -> None:
        buffer = StreamReplayBuffer(max_bytes_total=200_000)
        big = "x" * 10_000
        for stream in range(40):
            for _ in range(20):
                await buffer.record(f"s{stream}", big)

        stats = await buffer.stats()
        assert stats["bytes"] <= 200_000, "byte budget exceeded"

    @pytest.mark.asyncio
    async def test_per_stream_bytes_stay_under_budget(self) -> None:
        buffer = StreamReplayBuffer(max_bytes_per_stream=50_000)
        for _ in range(200):
            await buffer.record("s", "y" * 5_000)

        stats = await buffer.stats()
        assert stats["bytes"] <= 50_000

    @pytest.mark.asyncio
    async def test_newest_frame_survives_even_when_oversized(self) -> None:
        """A frame bigger than the whole budget must still be replayable —
        dropping it would silently lose the answer it carries."""
        buffer = StreamReplayBuffer(max_bytes_per_stream=100, max_bytes_total=100)
        await buffer.record("s", "z" * 50_000)

        retained = await buffer.replay("s")
        assert len(retained) == 1
        assert retained[0].data == "z" * 50_000

    @pytest.mark.asyncio
    async def test_eviction_keeps_the_accounting_honest(self) -> None:
        """If evicted bytes were not subtracted, the total would drift up
        until the buffer refused to retain anything at all."""
        buffer = StreamReplayBuffer(max_frames_per_stream=5)
        for _ in range(500):
            await buffer.record("s", "a" * 1_000)

        stats = await buffer.stats()
        retained = await buffer.replay("s")
        assert stats["frames"] == len(retained) == 5
        assert stats["bytes"] == sum(frame.size for frame in retained)

    @pytest.mark.asyncio
    async def test_dropping_a_stream_reclaims_its_bytes(self) -> None:
        buffer = StreamReplayBuffer()
        await buffer.record("a", "x" * 1_000)
        await buffer.record("b", "x" * 1_000)
        before = (await buffer.stats())["bytes"]

        await buffer.drop("a")
        after = (await buffer.stats())["bytes"]

        assert after < before
        assert after == sum(frame.size for frame in await buffer.replay("b"))

    @pytest.mark.asyncio
    async def test_sequences_stay_gapless_through_eviction(self) -> None:
        """Eviction trims history; it must never renumber what survives."""
        buffer = StreamReplayBuffer(max_frames_per_stream=4)
        for _ in range(20):
            await buffer.record("s", "f")

        sequences = [frame.sequence for frame in await buffer.replay("s")]
        assert sequences == [17, 18, 19, 20]


class TestStreamReplayEnvConfiguration:
    def test_defaults_when_nothing_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "AGENTOMATIC_STREAM_REPLAY",
            "AGENTOMATIC_STREAM_FRAMES",
            "AGENTOMATIC_STREAM_COUNT",
            "AGENTOMATIC_STREAM_MAX_MB",
        ):
            monkeypatch.delenv(var, raising=False)
        assert isinstance(buffer_from_env(), StreamReplayBuffer)

    @pytest.mark.asyncio
    async def test_replay_can_be_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTOMATIC_STREAM_REPLAY", "0")
        buffer = buffer_from_env()
        for _ in range(20):
            await buffer.record("s", "frame")
        # Streaming still works; only the history is gone.
        assert len(await buffer.replay("s")) <= 1

    @pytest.mark.asyncio
    async def test_frame_limit_is_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTOMATIC_STREAM_FRAMES", "3")
        buffer = buffer_from_env()
        for _ in range(10):
            await buffer.record("s", "frame")
        assert len(await buffer.replay("s")) == 3

    @pytest.mark.asyncio
    async def test_a_bad_value_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo in a container's env must not stop the platform booting."""
        monkeypatch.setenv("AGENTOMATIC_STREAM_FRAMES", "lots")
        buffer = buffer_from_env()
        for _ in range(10):
            await buffer.record("s", "frame")
        assert len(await buffer.replay("s")) == 10
