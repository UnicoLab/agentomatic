"""Tests for the custom endpoints subsystem."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agentomatic import AgentPlatform
from agentomatic.endpoints import (
    AggregationStrategy,
    AuthType,
    BaseEndpoint,
    EndpointRegistry,
    UpstreamAuthConfig,
    UpstreamConfig,
)
from agentomatic.endpoints.auth import UpstreamAuthenticator, resolve_env
from agentomatic.endpoints.client import MultiModelClient
from agentomatic.endpoints.models import (
    EndpointCallRequest,
    EndpointCallResponse,
    UpstreamResult,
)

# ---------------------------------------------------------------------------
# Env interpolation + auth headers
# ---------------------------------------------------------------------------


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("MY_SECRET", "s3cr3t")
    assert resolve_env("${MY_SECRET}") == "s3cr3t"
    assert resolve_env("prefix-${MY_SECRET}-suffix") == "prefix-s3cr3t-suffix"
    assert resolve_env("no-placeholder") == "no-placeholder"
    # Missing vars resolve to empty string.
    assert resolve_env("${DOES_NOT_EXIST}") == ""


async def test_auth_headers_none():
    auth = UpstreamAuthenticator(UpstreamAuthConfig(type=AuthType.NONE))
    assert await auth.headers(client=None) == {}


async def test_auth_headers_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "abc123")
    cfg = UpstreamAuthConfig(
        type=AuthType.API_KEY,
        api_key="${API_KEY}",
        header_name="X-API-Key",
        header_prefix="",
    )
    headers = await UpstreamAuthenticator(cfg).headers(client=None)
    assert headers == {"X-API-Key": "abc123"}


async def test_auth_headers_bearer():
    cfg = UpstreamAuthConfig(type=AuthType.BEARER, api_key="tok")
    headers = await UpstreamAuthenticator(cfg).headers(client=None)
    assert headers == {"Authorization": "Bearer tok"}


async def test_auth_headers_basic():
    cfg = UpstreamAuthConfig(type=AuthType.BASIC, username="user", password="pass")
    headers = await UpstreamAuthenticator(cfg).headers(client=None)
    assert headers["Authorization"].startswith("Basic ")


# ---------------------------------------------------------------------------
# MultiModelClient aggregation
# ---------------------------------------------------------------------------


def _multi(strategy: AggregationStrategy) -> MultiModelClient:
    return MultiModelClient(
        [
            UpstreamConfig(name="a", base_url="https://a.test"),
            UpstreamConfig(name="b", base_url="https://b.test"),
            UpstreamConfig(name="c", base_url="https://c.test"),
        ],
        strategy=strategy,
    )


async def test_fan_out_all(monkeypatch):
    client = _multi(AggregationStrategy.ALL)

    async def fake_request(self, payload=None, **kwargs):
        return UpstreamResult(upstream=self.name, ok=True, data={"v": self.name})

    monkeypatch.setattr("agentomatic.endpoints.client.UpstreamClient.request", fake_request)
    ok, aggregated, results = await client.fan_out({"x": 1})
    assert ok is True
    assert aggregated == {"a": {"v": "a"}, "b": {"v": "b"}, "c": {"v": "c"}}
    assert len(results) == 3


async def test_fan_out_first_success(monkeypatch):
    client = _multi(AggregationStrategy.FIRST_SUCCESS)

    async def fake_request(self, payload=None, **kwargs):
        ok = self.name != "a"  # 'a' fails
        return UpstreamResult(upstream=self.name, ok=ok, data={"v": self.name})

    monkeypatch.setattr("agentomatic.endpoints.client.UpstreamClient.request", fake_request)
    ok, aggregated, _ = await client.fan_out({"x": 1})
    assert ok is True
    assert aggregated in ({"v": "b"}, {"v": "c"})


async def test_fan_out_majority(monkeypatch):
    client = _multi(AggregationStrategy.MAJORITY)

    async def fake_request(self, payload=None, **kwargs):
        # a and b agree; c disagrees
        data = "yes" if self.name in ("a", "b") else "no"
        return UpstreamResult(upstream=self.name, ok=True, data=data)

    monkeypatch.setattr("agentomatic.endpoints.client.UpstreamClient.request", fake_request)
    ok, aggregated, _ = await client.fan_out({"x": 1})
    assert ok is True
    assert aggregated == "yes"


async def test_fan_out_subset(monkeypatch):
    client = _multi(AggregationStrategy.ALL)

    async def fake_request(self, payload=None, **kwargs):
        return UpstreamResult(upstream=self.name, ok=True, data=self.name)

    monkeypatch.setattr("agentomatic.endpoints.client.UpstreamClient.request", fake_request)
    ok, aggregated, results = await client.fan_out({"x": 1}, upstreams=["a"])
    assert len(results) == 1
    assert aggregated == {"a": "a"}


# ---------------------------------------------------------------------------
# BaseEndpoint schema + handle
# ---------------------------------------------------------------------------


class _In(BaseModel):
    text: str


class _Out(BaseModel):
    echoed: str


class EchoEndpoint(BaseEndpoint[_In, _Out]):
    endpoint_name = "echo"
    endpoint_description = "Echoes text back."
    path = "/echo"

    async def handle(self, request):  # type: ignore[override]
        return _Out(echoed=request.text)


def test_endpoint_schema_extraction():
    ep = EchoEndpoint()
    assert ep.get_input_schema() is _In
    assert ep.get_output_schema() is _Out


def test_endpoint_default_schema():
    ep = BaseEndpoint()
    assert ep.get_input_schema() is EndpointCallRequest
    assert ep.get_output_schema() is EndpointCallResponse


async def test_endpoint_handle_no_upstreams():
    ep = BaseEndpoint()
    result = await ep.handle(EndpointCallRequest(payload={"a": 1}))
    assert isinstance(result, EndpointCallResponse)
    assert result.ok is False
    assert result.results == []


def test_endpoint_info():
    info = EchoEndpoint().info()
    assert info["name"] == "echo"
    assert info["path"] == "/echo"
    assert info["methods"] == ["POST"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_programmatic_register():
    reg = EndpointRegistry()
    reg.register(EchoEndpoint())
    assert reg.count == 1
    assert reg.get("echo") is not None
    assert reg.list_names() == ["echo"]


def test_registry_discover_from_file(tmp_path):
    endpoints_dir = tmp_path / "endpoints"
    endpoints_dir.mkdir()
    (endpoints_dir / "__init__.py").write_text("")
    (endpoints_dir / "my_endpoint.py").write_text(
        "from __future__ import annotations\n"
        "from agentomatic.endpoints import BaseEndpoint\n"
        "class MyEndpoint(BaseEndpoint):\n"
        "    endpoint_name = 'discovered'\n"
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        reg = EndpointRegistry()
        reg.discover(endpoints_dir, "endpoints")
        assert "discovered" in reg.list_names()
    finally:
        sys.path.remove(str(tmp_path))


# ---------------------------------------------------------------------------
# Router integration via the platform
# ---------------------------------------------------------------------------


def test_endpoint_mounted_and_callable(tmp_path):
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
    )
    platform.register_endpoint(EchoEndpoint())
    app = platform.build()

    with TestClient(app) as client:
        resp = client.get("/api/v1/endpoints/echo/health")
        assert resp.status_code == 200
        assert resp.json()["endpoint"] == "echo"

        resp = client.get("/api/v1/endpoints/echo/info")
        assert resp.status_code == 200
        assert resp.json()["name"] == "echo"

        resp = client.post("/api/v1/endpoints/echo/echo", json={"text": "hi"})
        assert resp.status_code == 200
        assert resp.json() == {"echoed": "hi"}

        resp = client.get("/api/v1/endpoints")
        assert resp.status_code == 200
        assert any(e["name"] == "echo" for e in resp.json())


def test_cli_endpoint_template_renders():
    import ast

    from agentomatic.cli.templates import TEMPLATES, get_template_files

    assert "endpoint" in TEMPLATES
    files = get_template_files("endpoint", "ensemble")
    assert "endpoint.py" in files
    ast.parse(files["endpoint.py"])
    assert "BaseEndpoint" in files["endpoint.py"]


def test_cli_connection_template_renders():
    import ast

    from agentomatic.cli.templates import TEMPLATES, get_template_files

    assert "connection" in TEMPLATES
    files = get_template_files("connection", "fraud")
    assert "connections.py" in files
    ast.parse(files["connections.py"])
    assert "CONNECTIONS" in files["connections.py"]
    assert "${FRAUD_DB_URL}" in files["connections.py"]


def test_endpoints_appear_in_openapi(tmp_path):
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
    )
    platform.register_endpoint(EchoEndpoint())
    app = platform.build()
    schema = app.openapi()
    assert "/api/v1/endpoints/echo/echo" in schema["paths"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
