"""End-to-end integration tests for arbitrary custom DB / vector-store clients.

These prove that a fully custom Python client package integrates through the
connection abstraction with correct lifecycle — no network, all in-memory:

* a novel async vector provider + adapter, scoped, initialised, used
  (``upsert``/``query``/``delete``) and closed on shutdown;
* a sync-only SDK wrapped with the ``asyncio.to_thread`` pattern;
* a non-vector specialised backend via ``CustomConnectionConfig(factory=...)``
  retrieved through :func:`get_connections` and closed on shutdown;
* the standalone :func:`initialize_connections` helper; and
* clear errors for unregistered provider names.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentomatic.connections import (
    ConnectionPurpose,
    CustomConnectionConfig,
    VectorConnectionConfig,
    get_connections,
    initialize_connections,
    register_connection_type,
    register_vector_provider,
    register_vector_store_adapter,
)
from agentomatic.connections.manager import reset_connections
from agentomatic.connections.vector import (
    _VECTOR_PROVIDERS,
    _VECTOR_STORE_ADAPTERS,
    VectorConnection,
)


@pytest.fixture(autouse=True)
def _clean_registries():
    """Isolate connection + vector-provider registries between tests."""
    reset_connections()
    providers = dict(_VECTOR_PROVIDERS)
    adapters = dict(_VECTOR_STORE_ADAPTERS)
    yield
    reset_connections()
    _VECTOR_PROVIDERS.clear()
    _VECTOR_PROVIDERS.update(providers)
    _VECTOR_STORE_ADAPTERS.clear()
    _VECTOR_STORE_ADAPTERS.update(adapters)


# ---------------------------------------------------------------------------
# Fake specialised clients (stand-ins for real vendor SDKs)
# ---------------------------------------------------------------------------


class FakeAsyncVectorClient:
    """A fully custom async vector DB client with its own lifecycle."""

    def __init__(self, url: str, api_key: str, **options: Any) -> None:
        self.url = url
        self.api_key = api_key
        self.options = options
        self.store: dict[str, dict[str, Any]] = {}
        self.closed = False

    async def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        for i, text, meta in zip(ids, texts, metadatas, strict=False):
            self.store[i] = {"text": text, "metadata": meta}

    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        hits = [
            {"id": i, "text": doc["text"], "score": 1.0, "metadata": doc["metadata"]}
            for i, doc in self.store.items()
            if query.lower() in doc["text"].lower()
        ]
        return hits[:top_k]

    async def remove(self, ids: list[str]) -> None:
        for i in ids:
            self.store.pop(i, None)

    async def aclose(self) -> None:
        self.closed = True


class FakeAsyncVectorStoreAdapter:
    """VectorStore adapter mapping the agnostic surface to the fake client."""

    def __init__(self, config: VectorConnectionConfig, client: FakeAsyncVectorClient) -> None:
        self.config = config
        self.client = client

    async def upsert(self, texts, metadatas=None, ids=None, *, embeddings=None) -> list[str]:
        ids = list(ids or [str(n) for n in range(len(texts))])
        metadatas = list(metadatas or [{} for _ in texts])
        await self.client.add(ids=ids, texts=list(texts), metadatas=metadatas)
        return ids

    async def query(self, text=None, *, embedding=None, k=5, filter=None) -> list[dict[str, Any]]:
        return await self.client.search(query=text or "", top_k=k)

    async def delete(self, ids) -> None:
        await self.client.remove(list(ids))


class FakeSyncSdk:
    """A sync-only SDK — calling it directly would block the event loop."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.rows: dict[str, str] = {}
        self.closed = False

    def put(self, key: str, value: str) -> None:
        self.rows[key] = value

    def get(self, key: str) -> str | None:
        return self.rows.get(key)

    def close(self) -> None:
        self.closed = True


class FakeSyncStoreAdapter:
    """Adapter that offloads blocking sync SDK calls via ``asyncio.to_thread``."""

    def __init__(self, config: VectorConnectionConfig, client: FakeSyncSdk) -> None:
        self.config = config
        self.client = client

    async def upsert(self, texts, metadatas=None, ids=None, *, embeddings=None) -> list[str]:
        ids = list(ids or [str(n) for n in range(len(texts))])
        for i, text in zip(ids, texts, strict=False):
            await asyncio.to_thread(self.client.put, i, text)
        return ids

    async def query(self, text=None, *, embedding=None, k=5, filter=None) -> list[dict[str, Any]]:
        matches = []
        for key in list(self.client.rows):
            value = await asyncio.to_thread(self.client.get, key)
            if value and (text or "").lower() in value.lower():
                matches.append({"id": key, "text": value, "score": 1.0, "metadata": {}})
        return matches[:k]

    async def delete(self, ids) -> None:
        for i in ids:
            self.client.rows.pop(i, None)


class FakeGraphDbClient:
    """A non-vector specialised backend (e.g. a graph DB) with async close."""

    def __init__(self, uri: str, *, token: str = "") -> None:
        self.uri = uri
        self.token = token
        self.edges: list[tuple[str, str]] = []
        self.closed = False

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.append((src, dst))

    def neighbors(self, node: str) -> list[str]:
        return [dst for src, dst in self.edges if src == node]

    async def disconnect(self) -> None:
        self.closed = True


def build_fake_async_vector(cfg: VectorConnectionConfig) -> FakeAsyncVectorClient:
    """Provider builder returning the custom async client."""
    return FakeAsyncVectorClient(url=cfg.url, api_key=cfg.api_key, **cfg.options)


def build_fake_sync_vector(cfg: VectorConnectionConfig) -> FakeSyncSdk:
    """Provider builder returning the sync-only SDK."""
    return FakeSyncSdk(dsn=cfg.url)


def make_graph_db(uri: str, *, token: str = "") -> FakeGraphDbClient:
    """Factory for the non-vector specialised backend."""
    return FakeGraphDbClient(uri, token=token)


# ---------------------------------------------------------------------------
# Example A — custom ASYNC vector DB, full lifecycle through a scope
# ---------------------------------------------------------------------------


async def test_custom_async_vector_client_end_to_end() -> None:
    register_vector_provider("fake_async_db", build_fake_async_vector)
    register_vector_store_adapter("fake_async_db", FakeAsyncVectorStoreAdapter)

    manager = await initialize_connections(
        "rag_agent",
        [
            VectorConnectionConfig(
                name="kb",
                provider="fake_async_db",
                url="mem://fake",
                api_key="secret",
                purpose=ConnectionPurpose.RAG,
                options={"tenant": "acme"},
            )
        ],
    )

    conn = get_connections("rag_agent").vector("kb")
    assert isinstance(conn, VectorConnection)
    # options forwarded to the custom client
    assert conn.client.options == {"tenant": "acme"}

    store = await conn.as_store()
    assert isinstance(store, FakeAsyncVectorStoreAdapter)

    ids = await store.upsert(
        texts=["Agentomatic wires custom clients", "unrelated doc"],
        metadatas=[{"src": "docs"}, {"src": "misc"}],
        ids=["a", "b"],
    )
    assert ids == ["a", "b"]

    hits = await store.query(text="custom clients", k=5)
    assert [h["id"] for h in hits] == ["a"]
    assert hits[0]["metadata"] == {"src": "docs"}

    await store.delete(["a"])
    assert await store.query(text="custom clients", k=5) == []

    # purpose lookup finds it regardless of kind
    assert "kb" in get_connections("rag_agent").by_purpose(ConnectionPurpose.RAG)

    client = conn.client
    await manager.close()
    assert client.closed is True


# ---------------------------------------------------------------------------
# Example B — sync-only SDK via the asyncio.to_thread pattern
# ---------------------------------------------------------------------------


async def test_sync_only_client_uses_to_thread() -> None:
    register_vector_provider("fake_sync_db", build_fake_sync_vector)
    register_vector_store_adapter("fake_sync_db", FakeSyncStoreAdapter)

    await initialize_connections(
        "sync_agent",
        [VectorConnectionConfig(name="kb", provider="fake_sync_db", url="mem://sync")],
    )

    conn = get_connections("sync_agent").vector("kb")
    store = await conn.as_store()

    await store.upsert(texts=["hello sync world"], ids=["x"])
    hits = await store.query(text="sync", k=3)
    assert hits and hits[0]["id"] == "x"

    client = conn.client
    await get_connections("sync_agent").close()
    # sync client exposes close() (not aclose) — still closed gracefully
    assert client.closed is True


# ---------------------------------------------------------------------------
# Example C — non-vector specialised DB via CustomConnectionConfig(factory=...)
# ---------------------------------------------------------------------------


async def test_custom_factory_non_vector_backend_lifecycle() -> None:
    manager = await initialize_connections(
        "graph_agent",
        [
            CustomConnectionConfig(
                name="graph",
                factory=make_graph_db,
                args=["bolt://mem"],
                kwargs={"token": "${GRAPH_TOKEN}"},
                purpose=ConnectionPurpose.GENERAL,
            )
        ],
    )

    conn = get_connections("graph_agent").custom("graph")
    client = conn.client
    assert isinstance(client, FakeGraphDbClient)

    client.add_edge("a", "b")
    client.add_edge("a", "c")
    assert set(client.neighbors("a")) == {"b", "c"}

    # retrievable via the uniform native-client accessor too
    same = await get_connections("graph_agent").client("graph")
    assert same is client

    await manager.close()
    assert client.closed is True


async def test_custom_factory_env_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_TOKEN", "s3cr3t")
    await initialize_connections(
        "graph_agent",
        [
            CustomConnectionConfig(
                name="graph",
                factory=make_graph_db,
                args=["bolt://mem"],
                kwargs={"token": "${GRAPH_TOKEN}"},
            )
        ],
    )
    client = get_connections("graph_agent").custom("graph").client
    assert client.token == "s3cr3t"


# ---------------------------------------------------------------------------
# register_connection_type — arbitrary connection class
# ---------------------------------------------------------------------------


async def test_register_connection_type_arbitrary_backend() -> None:
    from pydantic import BaseModel

    class TimeseriesConfig(BaseModel):
        name: str
        purpose: ConnectionPurpose = ConnectionPurpose.ANALYTICS
        dsn: str = "mem://ts"

    class TimeseriesConnection:
        def __init__(self, config: TimeseriesConfig) -> None:
            self.config = config
            self._client: Any = None
            self.closed = False

        @property
        def name(self) -> str:
            return self.config.name

        @property
        def client(self) -> Any:
            return self._client

        async def initialize(self) -> None:
            self._client = {"dsn": self.config.dsn, "series": {}}

        async def health_check(self) -> dict[str, Any]:
            return {"connection": self.name, "status": "healthy"}

        async def close(self) -> None:
            self.closed = True

    register_connection_type(TimeseriesConfig, TimeseriesConnection)

    manager = await initialize_connections("ts_agent", [TimeseriesConfig(name="metrics")])
    conn = get_connections("ts_agent").get("metrics")
    assert isinstance(conn, TimeseriesConnection)
    assert conn.client == {"dsn": "mem://ts", "series": {}}

    health = await manager.health_check()
    assert health["metrics"]["status"] == "healthy"

    await manager.close()
    assert conn.closed is True


# ---------------------------------------------------------------------------
# Negative — unregistered provider name raises a clear, actionable error
# ---------------------------------------------------------------------------


async def test_unknown_provider_raises_clear_error() -> None:
    await initialize_connections(
        "bad_agent",
        [VectorConnectionConfig(name="kb", provider="totally_made_up_db")],
    )
    conn = get_connections("bad_agent").vector("kb")
    with pytest.raises(ValueError, match="register_vector_provider"):
        _ = conn.client


def test_vector_close_handles_disconnect_only_client() -> None:
    """A client exposing only ``disconnect`` (no aclose/close) still closes."""

    class OnlyDisconnect:
        def __init__(self) -> None:
            self.closed = False

        def disconnect(self) -> None:
            self.closed = True

    register_vector_provider("disc_db", lambda cfg: OnlyDisconnect())
    conn = VectorConnection(VectorConnectionConfig(name="kb", provider="disc_db"))
    client = conn.client
    asyncio.run(conn.close())
    assert client.closed is True
