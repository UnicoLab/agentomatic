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
