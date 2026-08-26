# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for the production control plane admin API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.connections.manager import PLATFORM_SCOPE, get_connections, reset_connections
from agentomatic.endpoints import BaseEndpoint


class _PingEndpoint(BaseEndpoint):
    endpoint_name = "ping"

    async def handle(self, request):
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
    assert data["control_token_required"] is True
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


def test_control_connection_diagnostics_never_expose_health_check_secrets(client):
    """Control reads must not turn custom health data into a secret leak."""

    class _SensitiveConnection:
        name = "private_service"
        config = None

        async def health_check(self):
            return {
                "connection": self.name,
                "kind": "custom",
                "status": "unhealthy",
                "url": "postgresql://operator:TOPSECRET@db.internal/app",
                "headers": {"Authorization": "Bearer TOPSECRET"},
                "error": "Cannot connect to postgresql://operator:TOPSECRET@db.internal/app",
            }

    get_connections(PLATFORM_SCOPE)._connections[_SensitiveConnection.name] = (
        _SensitiveConnection()
    )

    listed = client.get("/api/v1/control/connections")
    assert listed.status_code == 200
    entry = listed.json()[0]["connections"]["private_service"]
    probe = client.get("/api/v1/control/connections/__platform__/private_service")
    assert probe.status_code == 200
    for payload in (entry, probe.json()):
        assert payload["connection"] == "private_service"
        assert payload["status"] == "unhealthy"
        assert "TOPSECRET" not in str(payload)
        assert "url" not in payload
        assert "headers" not in payload


def test_control_health_never_exposes_agent_or_connection_health_secrets(client, platform):
    """The aggregate readiness endpoint has the same redaction contract."""

    def _broken_graph():
        raise RuntimeError("graph setup failed with token TOPSECRET")

    class _SensitiveConnection:
        name = "private_service"
        config = None

        async def health_check(self):
            return {
                "connection": self.name,
                "status": "unhealthy",
                "url": "postgresql://operator:TOPSECRET@db.internal/app",
                "error": "Cannot connect with TOPSECRET",
            }

    agent = platform._registry.get("echo_agent")
    assert agent is not None
    agent.graph_fn = _broken_graph
    get_connections(PLATFORM_SCOPE)._connections[_SensitiveConnection.name] = (
        _SensitiveConnection()
    )

    agent_detail = client.get("/api/v1/control/agents/echo_agent")
    aggregate = client.get("/api/v1/control/health")
    assert agent_detail.status_code == aggregate.status_code == 200
    for payload in (agent_detail.json(), aggregate.json()):
        assert "TOPSECRET" not in str(payload)

    health = aggregate.json()
    assert health["agents"]["echo_agent"]["graph_ready"] is False
    assert "graph_error" not in health["agents"]["echo_agent"]
    assert "url" not in health["connections"][PLATFORM_SCOPE]["private_service"]


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


def test_disable_agent_blocks_both_name_and_slug_aliases(client):
    """Regression: an agent mounted under both its folder name and its
    manifest slug (name="echo_agent", slug="echo") must be fully drained
    when disabled by either alias — not leave the other alias's routes live.
    """
    resp = client.post(
        "/api/v1/control/agents/echo_agent/disable",
        headers={"X-Control-Token": "secret-token"},
    )
    assert resp.status_code == 200

    # Disabling by folder name must also block the slug-mounted alias.
    by_name = client.post("/api/v1/echo_agent/invoke", json={"query": "hi"})
    by_slug = client.post("/api/v1/echo/invoke", json={"query": "hi"})
    assert by_name.status_code == 503
    assert by_slug.status_code == 503

    # Re-enabling must restore both aliases.
    resp = client.post(
        "/api/v1/control/agents/echo_agent/enable",
        headers={"X-Control-Token": "secret-token"},
    )
    assert resp.status_code == 200
    assert client.post("/api/v1/echo_agent/invoke", json={"query": "hi"}).status_code == 200
    assert client.post("/api/v1/echo/invoke", json={"query": "hi"}).status_code == 200


def test_disable_agent_by_slug_blocks_name_alias_too(client):
    """Same guarantee in the other direction: disabling by slug must also
    block the folder-name-mounted alias.
    """
    resp = client.post(
        "/api/v1/control/agents/echo/disable",
        headers={"X-Control-Token": "secret-token"},
    )
    assert resp.status_code == 200
    assert client.post("/api/v1/echo/invoke", json={"query": "hi"}).status_code == 503
    assert client.post("/api/v1/echo_agent/invoke", json={"query": "hi"}).status_code == 503

    client.post(
        "/api/v1/control/agents/echo/enable",
        headers={"X-Control-Token": "secret-token"},
    )


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
