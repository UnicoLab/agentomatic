# pyright: reportMissingParameterType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for the A2A task stream and the capabilities the agent card claims.

Agentomatic exposed A2A task submission, status and cancel, but nothing an A2A
client could subscribe to: a caller that lost its connection mid-task could
only re-poll for the current state. These cover the resumable A2A stream and
the card that advertises it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.core.manifest import AgentManifest

BASE = "/api/v1"


@dataclass
class _State:
    """Minimal state for the streaming test agent."""

    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


async def _echo_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Return a canned response for the incoming query."""
    return {"response": f"echo: {state.get('current_query', '')}"}


@pytest.fixture
def client() -> Any:
    """A platform with one agent and the task manager enabled."""
    platform = AgentPlatform(
        agents_dir="/tmp/agentomatic_a2a_stream_test",
        title="A2A Stream Test",
        version="0.0.1",
        enable_tasks=True,
    )
    platform.register_agent(
        manifest=AgentManifest(
            name="echo",
            slug="fn-echo",
            description="Echo agent",
            version="1.0.0",
        ),
        node_fn=_echo_fn,
    )
    with TestClient(platform.build()) as test_client:
        yield test_client


def _frames(body: str) -> list[dict[str, Any]]:
    """Parse an SSE body into its JSON data frames."""
    out: list[dict[str, Any]] = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: ") :]))
    return out


def _ids(body: str) -> list[int]:
    """Extract SSE ``id:`` values from a body."""
    return [int(line[len("id: ") :]) for line in body.splitlines() if line.startswith("id: ")]


def _submit(client: Any) -> str:
    """Submit an A2A task and return its id."""
    response = client.post(
        f"{BASE}/echo/a2a/tasks",
        json={"message": {"parts": [{"type": "text", "text": "stream me"}]}},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["task_id"])


class TestA2AStream:
    def test_stream_endpoint_exists(self, client: Any) -> None:
        task_id = _submit(client)
        response = client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_frames_use_a2a_task_states(self, client: Any) -> None:
        """A2A clients expect submitted/working/completed, not internal names."""
        task_id = _submit(client)
        body = client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events").text

        states = {frame.get("status") for frame in _frames(body)}
        assert states, "no frames"
        assert states <= {"submitted", "working", "completed", "failed", "canceled"}

    def test_stream_carries_event_ids(self, client: Any) -> None:
        task_id = _submit(client)
        body = client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events").text
        assert _ids(body), "no id: lines — an A2A client cannot resume"

    def test_since_resumes_without_replaying_seen_events(self, client: Any) -> None:
        task_id = _submit(client)
        client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events").text
        resumed = client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events?since=1").text
        assert all(seq > 1 for seq in _ids(resumed))

    def test_last_event_id_header_matches_since(self, client: Any) -> None:
        task_id = _submit(client)
        by_query = _ids(client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events?since=1").text)
        by_header = _ids(
            client.get(
                f"{BASE}/echo/a2a/tasks/{task_id}/events",
                headers={"Last-Event-ID": "1"},
            ).text
        )
        assert by_header == by_query

    def test_terminal_stream_closes_with_done(self, client: Any) -> None:
        task_id = _submit(client)
        body = client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events").text
        assert body.rstrip().endswith("data: [DONE]")

    def test_terminal_frame_carries_the_result(self, client: Any) -> None:
        task_id = _submit(client)
        body = client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events").text
        terminal = [f for f in _frames(body) if f.get("status") == "completed"]
        assert terminal, "no completed frame"
        assert "result" in terminal[-1]

    def test_unknown_task_404s(self, client: Any) -> None:
        response = client.get(f"{BASE}/echo/a2a/tasks/does-not-exist/events")
        assert response.status_code == 404


class TestAgentCardCapabilities:
    def test_card_advertises_the_stream_endpoint(self, client: Any) -> None:
        card = client.get(f"{BASE}/echo/card").json()
        assert "a2a_events" in card["endpoints"]

    def test_card_reports_resumable_streams(self, client: Any) -> None:
        card = client.get(f"{BASE}/echo/card").json()
        assert card["capabilities"]["resumableStreams"] is True
        assert card["capabilities"]["stateTransitionHistory"] is True
        assert card["capabilities"]["pushNotifications"] is True

    def test_capabilities_are_not_claimed_without_a_task_manager(self) -> None:
        """A card that claims a capability the deployment lacks sends clients
        down a path that then 501s."""
        platform = AgentPlatform(
            agents_dir="/tmp/agentomatic_a2a_stream_test_notasks",
            title="No Tasks",
            version="0.0.1",
            enable_tasks=False,
        )
        platform.register_agent(
            manifest=AgentManifest(name="echo", slug="fn-echo", description="Echo"),
            node_fn=_echo_fn,
        )
        with TestClient(platform.build()) as test_client:
            card = test_client.get(f"{BASE}/echo/card").json()

        assert card["capabilities"]["resumableStreams"] is False
        assert card["capabilities"]["stateTransitionHistory"] is False
        assert card["capabilities"]["pushNotifications"] is False
        assert "a2a_events" not in card["endpoints"]


class TestDiscoveryMatchesTheAgentCard:
    """``/.well-known/agent.json`` is the canonical A2A discovery document.

    It used to emit a reduced card — no capabilities, only ``invoke`` and
    ``chat`` — so a conforming client learned *less* about an agent than one
    that guessed the per-agent URL and could not discover streaming at all.
    """

    def test_discovery_card_matches_the_per_agent_card(self, client: Any) -> None:
        per_agent = client.get(f"{BASE}/echo/card").json()
        discovery = client.get("/.well-known/agent.json").json()["agents"]["echo"]
        assert discovery == per_agent

    def test_discovery_advertises_capabilities(self, client: Any) -> None:
        discovery = client.get("/.well-known/agent.json").json()["agents"]["echo"]
        assert discovery["capabilities"]["resumableStreams"] is True

    def test_discovery_advertises_the_streaming_endpoints(self, client: Any) -> None:
        endpoints = client.get("/.well-known/agent.json").json()["agents"]["echo"]["endpoints"]
        assert "a2a_events" in endpoints
        assert "stream_replay" in endpoints

    def test_discovery_keeps_its_platform_envelope(self, client: Any) -> None:
        """Unifying the card must not change the document around it."""
        body = client.get("/.well-known/agent.json").json()
        assert body["platform"] == "A2A Stream Test"
        assert "version" in body
        assert "echo" in body["agents"]

    def test_discovery_omits_task_endpoints_without_a_task_manager(self) -> None:
        platform = AgentPlatform(
            agents_dir="/tmp/agentomatic_a2a_discovery_notasks",
            title="No Tasks",
            version="0.0.1",
            enable_tasks=False,
        )
        platform.register_agent(
            manifest=AgentManifest(name="echo", slug="fn-echo", description="Echo"),
            node_fn=_echo_fn,
        )
        with TestClient(platform.build()) as test_client:
            discovery = test_client.get("/.well-known/agent.json").json()["agents"]["echo"]

        assert "a2a_events" not in discovery["endpoints"]
        assert discovery["capabilities"]["resumableStreams"] is False


class TestStreamDoesNotHammerTheStore:
    """Rendering an A2A frame needs the task record only for the terminal
    payload. Reading it per event turned a replay into one store round-trip
    per frame — invisible with the in-memory store, N queries with SQL."""

    def test_replay_reads_the_record_at_most_once(self, client: Any, monkeypatch) -> None:
        task_id = _submit(client)
        # Let the task finish so there is a real history to replay.
        client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events")

        from agentomatic.tasks.manager import TaskManager

        original = TaskManager.get
        calls: list[str] = []

        async def counting_get(self, tid: str):
            calls.append(tid)
            return await original(self, tid)

        monkeypatch.setattr(TaskManager, "get", counting_get)
        body = client.get(f"{BASE}/echo/a2a/tasks/{task_id}/events").text

        frames = len(_frames(body))
        assert frames >= 1
        # One lookup to 404-check, one for the snapshot, at most one for the
        # terminal payload — never one per frame.
        assert len(calls) <= 3, f"{len(calls)} store reads for {frames} frames"
