"""Tests for the per-agent connections subsystem."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentomatic.connections import (
    ConnectionKind,
    ConnectionPurpose,
    CustomConnection,
    CustomConnectionConfig,
    DatabaseConnectionConfig,
    HttpConnectionConfig,
    VectorConnection,
    VectorConnectionConfig,
    get_connections,
    register_connection_type,
    register_vector_provider,
)
from agentomatic.connections.database import _inject_credentials
from agentomatic.connections.http import HttpConnection
from agentomatic.connections.manager import (
    ConnectionManager,
    all_managers,
    register_connections,
    reset_connections,
)
from agentomatic.endpoints.models import AuthType, UpstreamAuthConfig, UpstreamResult


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_connections()
    yield
    reset_connections()


# ---------------------------------------------------------------------------
# Credential injection
# ---------------------------------------------------------------------------


def test_inject_credentials_noop_without_creds():
    url = "postgresql+asyncpg://host:5432/db"
    assert _inject_credentials(url, "", "") == url


def test_inject_credentials_user_and_password():
    url = "postgresql+asyncpg://host:5432/db"
    result = _inject_credentials(url, "alice", "s p@ss")
    assert "alice" in result
    # Password special chars must be url-encoded.
    assert "s+p%40ss" in result or "s%20p%40ss" in result
    assert "host:5432" in result


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


def test_manager_add_database_and_http():
    mgr = ConnectionManager("agent_x")
    mgr.add(DatabaseConnectionConfig(name="main", url="sqlite+aiosqlite:///:memory:"))
    mgr.add(HttpConnectionConfig(name="api", base_url="https://api.test"))
    assert mgr.count == 2
    assert set(mgr.list_names()) == {"main", "api"}
    assert mgr.database("main").name == "main"
    assert mgr.http("api").name == "api"


def test_manager_wrong_kind_raises():
    mgr = ConnectionManager("agent_x")
    mgr.add(HttpConnectionConfig(name="api", base_url="https://api.test"))
    with pytest.raises(KeyError):
        mgr.database("api")
    with pytest.raises(KeyError):
        mgr.http("missing")


def test_register_and_get_connections():
    register_connections(
        "agent_y",
        [HttpConnectionConfig(name="svc", base_url="https://svc.test")],
    )
    mgr = get_connections("agent_y")
    assert mgr.count == 1
    assert "agent_y" in all_managers()


def test_get_connections_creates_empty_scope():
    mgr = get_connections("brand_new")
    assert mgr.count == 0
    assert mgr.scope == "brand_new"


# ---------------------------------------------------------------------------
# HttpConnection
# ---------------------------------------------------------------------------


def test_http_connection_built_from_config():
    cfg = HttpConnectionConfig(
        name="svc",
        base_url="https://svc.test",
        auth=UpstreamAuthConfig(type=AuthType.BEARER, api_key="tok"),
    )
    conn = HttpConnection(cfg)
    assert conn.name == "svc"


async def test_http_connection_health_check():
    conn = HttpConnection(HttpConnectionConfig(name="svc", base_url="https://svc.test"))
    health = await conn.health_check()
    assert health["connection"] == "svc"
    assert health["kind"] == "http"


async def test_http_connection_request_emits_metric(monkeypatch):
    conn = HttpConnection(HttpConnectionConfig(name="svc", base_url="https://svc.test"))

    async def fake_request(self, payload=None, **kwargs):
        return UpstreamResult(upstream="svc", ok=True, status_code=200, data={"ok": True})

    monkeypatch.setattr("agentomatic.endpoints.client.UpstreamClient.request", fake_request)
    result = await conn.get("/ping")
    assert result.ok is True
    assert result.data == {"ok": True}


async def test_manager_health_check_aggregates():
    mgr = ConnectionManager("agent_z")
    mgr.add(HttpConnectionConfig(name="svc", base_url="https://svc.test"))
    health = await mgr.health_check()
    assert "svc" in health


# ---------------------------------------------------------------------------
# Purpose tagging + lookup
# ---------------------------------------------------------------------------


def test_purpose_defaults():
    db = DatabaseConnectionConfig(name="d", url="sqlite+aiosqlite:///:memory:")
    assert db.purpose is ConnectionPurpose.GENERAL
    vec = VectorConnectionConfig(name="v", provider="qdrant")
    assert vec.purpose is ConnectionPurpose.VECTOR
    assert vec.kind is ConnectionKind.VECTOR


def test_manager_by_purpose_and_for_purpose():
    mgr = ConnectionManager("rag_agent")
    mgr.add(
        DatabaseConnectionConfig(
            name="memory_db",
            url="sqlite+aiosqlite:///:memory:",
            purpose=ConnectionPurpose.MEMORY,
        )
    )
    mgr.add(VectorConnectionConfig(name="kb", provider="qdrant", purpose=ConnectionPurpose.RAG))
    mgr.add(
        HttpConnectionConfig(
            name="docs", base_url="https://docs.test", purpose=ConnectionPurpose.RAG
        )
    )

    assert set(mgr.by_purpose(ConnectionPurpose.RAG)) == {"kb", "docs"}
    assert set(mgr.by_purpose("rag")) == {"kb", "docs"}
    assert [c.name for c in mgr.for_purpose(ConnectionPurpose.MEMORY)] == ["memory_db"]
    assert mgr.first_for_purpose(ConnectionPurpose.MEMORY).name == "memory_db"
    assert mgr.first_for_purpose(ConnectionPurpose.CACHE) is None


# ---------------------------------------------------------------------------
# Vector connections
# ---------------------------------------------------------------------------


def test_manager_vector_accessor():
    mgr = ConnectionManager("agent_v")
    mgr.add(VectorConnectionConfig(name="kb", provider="qdrant", collection="c"))
    conn = mgr.vector("kb")
    assert isinstance(conn, VectorConnection)
    assert conn.provider == "qdrant"
    assert conn.collection == "c"


def test_manager_vector_wrong_kind_raises():
    mgr = ConnectionManager("agent_v")
    mgr.add(HttpConnectionConfig(name="api", base_url="https://api.test"))
    with pytest.raises(KeyError):
        mgr.vector("api")


def test_vector_unknown_provider_raises():
    conn = VectorConnection(VectorConnectionConfig(name="x", provider="does_not_exist"))
    with pytest.raises(ValueError, match="Unknown vector provider"):
        _ = conn.client


def test_register_vector_provider_custom():
    class _FakeClient:
        def __init__(self, cfg):
            self.cfg = cfg

    register_vector_provider("fake_store", lambda cfg: _FakeClient(cfg))
    conn = VectorConnection(VectorConnectionConfig(name="kb", provider="fake_store"))
    assert isinstance(conn.client, _FakeClient)
    assert conn.client.cfg.name == "kb"


async def test_vector_health_check_custom_provider():
    register_vector_provider("fake_store2", lambda cfg: object())
    conn = VectorConnection(VectorConnectionConfig(name="kb", provider="fake_store2"))
    health = await conn.health_check()
    assert health["kind"] == "vector"
    assert health["status"] == "healthy"


# ---------------------------------------------------------------------------
# Extensible connection-type registry
# ---------------------------------------------------------------------------


class _RedisLikeConfig(BaseModel):
    name: str
    url: str
    purpose: ConnectionPurpose = ConnectionPurpose.CACHE


class _RedisLikeConnection:
    def __init__(self, config: _RedisLikeConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    async def initialize(self) -> None:  # pragma: no cover - trivial
        pass

    async def health_check(self) -> dict:
        return {"connection": self.name, "kind": "redis", "status": "configured"}

    async def close(self) -> None:  # pragma: no cover - trivial
        pass


def test_register_connection_type_custom_backend():
    register_connection_type(_RedisLikeConfig, _RedisLikeConnection)
    mgr = ConnectionManager("agent_cache")
    mgr.add(_RedisLikeConfig(name="cache", url="redis://localhost:6379"))
    conn = mgr.get("cache")
    assert isinstance(conn, _RedisLikeConnection)
    assert set(mgr.by_purpose(ConnectionPurpose.CACHE)) == {"cache"}


async def test_custom_backend_health_aggregated():
    register_connection_type(_RedisLikeConfig, _RedisLikeConnection)
    mgr = ConnectionManager("agent_cache2")
    mgr.add(_RedisLikeConfig(name="cache", url="redis://localhost:6379"))
    health = await mgr.health_check()
    assert health["cache"]["kind"] == "redis"


def test_unsupported_config_raises():
    class _Unknown(BaseModel):
        name: str = "x"

    mgr = ConnectionManager("agent_bad")
    with pytest.raises(TypeError, match="Unsupported connection config"):
        mgr.add(_Unknown())


# ---------------------------------------------------------------------------
# Generic factory-based connections (any backend, zero classes)
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    def __init__(self, url, *, token=None):
        self.url = url
        self.token = token
        self.closed = False

    async def ping(self):
        return True

    async def aclose(self):
        self.closed = True


def _make_fake_client(url, *, token=None):
    return _FakeAsyncClient(url, token=token)


async def _make_fake_client_async(url, *, token=None):
    return _FakeAsyncClient(url, token=token)


async def test_custom_connection_callable_factory():
    conn = CustomConnection(
        CustomConnectionConfig(
            name="cache",
            factory=_make_fake_client,
            args=["redis://localhost"],
            kwargs={"token": "secret"},
            purpose=ConnectionPurpose.CACHE,
        )
    )
    await conn.initialize()
    assert isinstance(conn.client, _FakeAsyncClient)
    assert conn.client.url == "redis://localhost"
    assert conn.client.token == "secret"


async def test_custom_connection_async_factory():
    conn = CustomConnection(
        CustomConnectionConfig(name="c", factory=_make_fake_client_async, args=["u"])
    )
    await conn.initialize()
    assert isinstance(conn.client, _FakeAsyncClient)


async def test_custom_connection_env_interpolation(monkeypatch):
    monkeypatch.setenv("MY_REDIS_URL", "redis://prod:6379")
    conn = CustomConnection(
        CustomConnectionConfig(
            name="cache",
            factory=_make_fake_client,
            args=["${MY_REDIS_URL}"],
        )
    )
    await conn.initialize()
    assert conn.client.url == "redis://prod:6379"


async def test_custom_connection_dotted_path_factory():
    # ``builtins.dict`` is a trivial, always-importable factory.
    conn = CustomConnection(
        CustomConnectionConfig(
            name="c",
            factory="builtins:dict",
            kwargs={"host": "localhost"},
        )
    )
    await conn.initialize()
    assert conn.client == {"host": "localhost"}


async def test_custom_connection_health_and_close():
    conn = CustomConnection(
        CustomConnectionConfig(name="c", factory=_make_fake_client, args=["u"])
    )
    health = await conn.health_check()
    assert health["status"] == "healthy"  # auto-detected ping()
    client = conn.client
    await conn.close()
    assert client.closed is True  # auto-detected aclose()


def test_custom_connection_client_before_init_raises():
    conn = CustomConnection(
        CustomConnectionConfig(name="c", factory=_make_fake_client, args=["u"])
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = conn.client


async def test_manager_custom_accessor_and_client_helper():
    mgr = ConnectionManager("agent_any")
    mgr.add(
        CustomConnectionConfig(
            name="cache",
            factory=_make_fake_client,
            args=["redis://x"],
            purpose=ConnectionPurpose.CACHE,
        )
    )
    assert isinstance(mgr.custom("cache"), CustomConnection)
    # Uniform client() helper initialises on demand.
    client = await mgr.client("cache")
    assert isinstance(client, _FakeAsyncClient)
    assert set(mgr.by_purpose(ConnectionPurpose.CACHE)) == {"cache"}


async def test_manager_client_helper_missing_raises():
    mgr = ConnectionManager("agent_any")
    with pytest.raises(KeyError):
        await mgr.client("nope")


def test_import_from_path_both_forms():
    from agentomatic.connections.custom import import_from_path, resolve_env_deep

    assert import_from_path("agentomatic.connections.custom:resolve_env_deep") is resolve_env_deep
    assert import_from_path("agentomatic.connections.custom.resolve_env_deep") is resolve_env_deep


def test_resolve_env_deep(monkeypatch):
    from agentomatic.connections.custom import resolve_env_deep

    monkeypatch.setenv("X", "1")
    monkeypatch.setenv("Y", "2")
    out = resolve_env_deep({"a": "${X}", "b": ["${Y}", 3], "c": 4})
    assert out == {"a": "1", "b": ["2", 3], "c": 4}


# ---------------------------------------------------------------------------
# Memory alignment (DatabaseConnection.create_store)
# ---------------------------------------------------------------------------


async def test_database_connection_create_store_shares_engine():
    pytest.importorskip("sqlalchemy")
    from agentomatic.connections.database import DatabaseConnection

    conn = DatabaseConnection(
        DatabaseConnectionConfig(
            name="memory",
            url="sqlite+aiosqlite:///:memory:",
            purpose=ConnectionPurpose.MEMORY,
        )
    )
    store = await conn.create_store()
    # Store reuses the connection's engine and must not own it.
    assert store._engine is conn.engine
    assert store._owns_engine is False

    thread = await store.create_thread("t1", "u1", "agent")
    assert thread["id"] == "t1"

    # Closing the store leaves the shared engine usable by the connection.
    await store.close()
    health = await conn.health_check()
    assert health["status"] == "healthy"
    await conn.close()


# ---------------------------------------------------------------------------
# ConnectionsMiddleware
# ---------------------------------------------------------------------------


async def test_connections_middleware_sets_request_state():
    from types import SimpleNamespace

    from agentomatic.middleware.connections import ConnectionsMiddleware

    register_connections(
        "weather", [HttpConnectionConfig(name="svc", base_url="https://svc.test")]
    )
    registry = SimpleNamespace(get=lambda name: object() if name == "weather" else None)
    mw = ConnectionsMiddleware(app=object(), registry=registry, api_prefix="/api/v1")

    captured = {}

    async def call_next(request):
        captured["connections"] = request.state.connections
        captured["agent"] = request.state.agent_name
        return "response"

    request = SimpleNamespace(
        url=SimpleNamespace(path="/api/v1/weather/chat"),
        state=SimpleNamespace(),
    )
    result = await mw.dispatch(request, call_next)
    assert result == "response"
    assert captured["agent"] == "weather"
    assert captured["connections"].scope == "weather"
    assert captured["connections"].get("svc") is not None


async def test_connections_middleware_falls_back_to_platform():
    from types import SimpleNamespace

    from agentomatic.connections.manager import PLATFORM_SCOPE
    from agentomatic.middleware.connections import ConnectionsMiddleware

    registry = SimpleNamespace(get=lambda name: None)
    mw = ConnectionsMiddleware(app=object(), registry=registry, api_prefix="/api/v1")

    captured = {}

    async def call_next(request):
        captured["connections"] = request.state.connections
        return "ok"

    request = SimpleNamespace(
        url=SimpleNamespace(path="/health"),
        state=SimpleNamespace(),
    )
    await mw.dispatch(request, call_next)
    assert captured["connections"].scope == PLATFORM_SCOPE


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
