"""Tests for the production control plane admin API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.connections.manager import reset_connections
from agentomatic.endpoints import BaseEndpoint


class _PingEndpoint(BaseEndpoint):
    endpoint_name = "ping"

    async def handle(self, request):  # type: ignore[override]
        return {"pong": True}


@pytest.fixture(autouse=True)
def _clean():
    reset_connections()
    yield
    reset_connections()


async def _echo(state):
    return {"response": "ok", "agent_type": "echo"}


@pytest.fixture
def platform(tmp_path):
    p = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        title="Control Test",
        version="9.9.9",
        enable_control_plane=True,
        control_token="secret-token",
    )
    p.register_agent(
        manifest=AgentManifest(name="echo_agent", slug="echo", description="Echo"),
        node_fn=_echo,
    )
    p.register_endpoint(_PingEndpoint())
    return p


@pytest.fixture
def client(platform):
    with TestClient(platform.build()) as c:
        yield c


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_control_info(client):
    resp = client.get("/api/v1/control")
    assert resp.status_code == 200
    data = resp.json()
    assert data["platform"] == "Control Test"
    assert data["version"] == "9.9.9"
    assert data["agent_count"] == 1
    assert data["endpoint_count"] == 1
    assert data["maintenance_mode"] is False


def test_control_list_agents(client):
    resp = client.get("/api/v1/control/agents")
    assert resp.status_code == 200
    agents = resp.json()
    assert len(agents) == 1
    assert agents[0]["name"] == "echo_agent"
    assert agents[0]["enabled"] is True


def test_control_get_agent(client):
    resp = client.get("/api/v1/control/agents/echo_agent")
    assert resp.status_code == 200
    assert resp.json()["name"] == "echo_agent"


def test_control_get_agent_404(client):
    resp = client.get("/api/v1/control/agents/nope")
    assert resp.status_code == 404


def test_control_list_endpoints(client):
    resp = client.get("/api/v1/control/endpoints")
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()]
    assert "ping" in names


def test_control_metrics_summary(client):
    resp = client.get("/api/v1/control/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["agents"] == 1
    assert data["endpoints"] == 1


def test_control_config(client):
    resp = client.get("/api/v1/control/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Control Test"
    assert "features" in data


def test_control_health(client):
    resp = client.get("/api/v1/control/health")
    assert resp.status_code == 200
    assert "agents" in resp.json()


# ---------------------------------------------------------------------------
# Mutating operations + token auth
# ---------------------------------------------------------------------------


def test_maintenance_requires_token(client):
    resp = client.post("/api/v1/control/maintenance", json={"enabled": True})
    assert resp.status_code == 401


def test_maintenance_toggle_with_token(client):
    resp = client.post(
        "/api/v1/control/maintenance",
        json={"enabled": True},
        headers={"X-Control-Token": "secret-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "maintenance"

    # Turn it back off so other requests are not blocked.
    resp = client.post(
        "/api/v1/control/maintenance",
        json={"enabled": False},
        headers={"X-Control-Token": "secret-token"},
    )
    assert resp.json()["state"] == "active"


def test_disable_and_enable_agent(client):
    resp = client.post(
        "/api/v1/control/agents/echo_agent/disable",
        headers={"X-Control-Token": "secret-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "disabled"

    # A disabled agent's routes should be gated (503).
    inv = client.post("/api/v1/echo_agent/invoke", json={"query": "hi"})
    assert inv.status_code == 503

    resp = client.post(
        "/api/v1/control/agents/echo_agent/enable",
        headers={"X-Control-Token": "secret-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "enabled"


def test_disable_agent_wrong_token(client):
    resp = client.post(
        "/api/v1/control/agents/echo_agent/disable",
        headers={"X-Control-Token": "wrong"},
    )
    assert resp.status_code == 401


def test_maintenance_mode_blocks_agent_calls(client):
    # Enable maintenance mode.
    client.post(
        "/api/v1/control/maintenance",
        json={"enabled": True},
        headers={"X-Control-Token": "secret-token"},
    )
    resp = client.post("/api/v1/echo_agent/invoke", json={"query": "hi"})
    assert resp.status_code == 503

    # Control plane itself stays reachable during maintenance.
    assert client.get("/api/v1/control").status_code == 200

    # Restore.
    client.post(
        "/api/v1/control/maintenance",
        json={"enabled": False},
        headers={"X-Control-Token": "secret-token"},
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
