"""End-to-end regression tests for class-agent server paths (P0-1).

Class agents (``BaseGraphAgent``) use a dataclass state and rely on
``input_to_state`` to convert the incoming request dict into that state.
Every server-side path (REST invoke/chat/stream + Studio streaming) must
route through ``input_to_state``/``atransform`` instead of passing a raw dict
into ``graph.ainvoke``/``astream`` — otherwise nodes receive a ``dict`` and
raise ``AttributeError`` (HTTP 500 / Studio ``run_error``).

These tests would have caught the regression where the REST router preferred
``graph_fn().ainvoke(dict)`` for class agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.agents import BaseGraphAgent


@dataclass
class _EchoState:
    """Dataclass state — accessing attributes on a raw dict would raise."""

    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class _EchoClassAgent(BaseGraphAgent[_EchoState]):
    """Minimal dataclass-state class agent used to reproduce P0-1."""

    agent_name = "echo_class"
    agent_description = "Echo class agent"
    agent_framework = "graph_agent"

    def build_graph(self) -> Any:
        """Build a single-node graph that echoes the request."""
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: _EchoState) -> _EchoState:
        """Echo the request text back — reads ``state.request`` (dataclass)."""
        state.output = {
            "response": f"Echo: {state.request}",
            "agent_type": "echo-class",
        }
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> _EchoState:
        """Convert the raw request dict into the dataclass state."""
        return _EchoState(request=input_data.get("current_query", ""))

    def state_to_output(self, state: _EchoState) -> dict[str, Any]:
        """Return the node's output dict."""
        return state.output


@pytest.fixture
def client() -> Any:
    """Build a platform with one class agent + Studio and yield a client."""
    platform = AgentPlatform(
        agents_dir="/tmp/agentomatic_class_agent_test_empty",
        title="Class Agent Test",
        version="0.0.1",
        enable_studio=True,
    )
    reg = _EchoClassAgent().as_registered_agent()
    platform.register_agent(
        manifest=reg.manifest,
        node_fn=reg.node_fn,
        graph_fn=reg.graph_fn,
        class_instance=reg.class_instance,
    )
    app = platform.build()
    with TestClient(app) as test_client:
        yield test_client


def test_rest_invoke_class_agent(client: TestClient) -> None:
    """POST /api/v1/<name>/invoke returns 200 with the converted output."""
    resp = client.post("/api/v1/echo_class/invoke", json={"query": "hello"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["response"] == "Echo: hello"


def test_rest_chat_class_agent(client: TestClient) -> None:
    """POST /api/v1/<name>/chat returns 200 with the converted output."""
    resp = client.post("/api/v1/echo_class/chat", json={"content": "hey"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["response"] == "Echo: hey"


def test_rest_invoke_stream_class_agent(client: TestClient) -> None:
    """POST /api/v1/<name>/invoke/stream streams the output and terminates."""
    resp = client.post("/api/v1/echo_class/invoke/stream", json={"query": "streamed"})
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "Echo: streamed" in body
    assert "[DONE]" in body
    assert "error" not in body.lower()


def test_studio_run_stream_class_agent(client: TestClient) -> None:
    """POST /studio/agents/<name>/runs/stream streams events, not run_error."""
    resp = client.post(
        "/studio/agents/echo_class/runs/stream",
        json={"query": "studio"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "run_error" not in body
    assert "Echo: studio" in body
