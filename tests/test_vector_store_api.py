"""Tests for the provider-agnostic VectorStore API."""

from __future__ import annotations

from typing import Any

import pytest

from agentomatic.connections import (
    VectorConnection,
    VectorConnectionConfig,
    VectorStore,
    register_vector_provider,
    register_vector_store_adapter,
)


class _MockNativeClient:
    """Minimal duck-typed native client that mimics upsert/query/delete."""

    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.query_calls: list[dict[str, Any]] = []

    async def upsert(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        payload = {
            "texts": list(texts),
            "metadatas": list(metadatas) if metadatas else None,
            "ids": list(ids) if ids else None,
            "embeddings": embeddings,
        }
        self.upserts.append(payload)
        return list(ids) if ids else [f"gen-{i}" for i, _ in enumerate(texts)]

    async def query(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.query_calls.append({"text": text, "embedding": embedding, "k": k, "filter": filter})
        return [{"id": "a", "text": "hello", "metadata": {}, "score": 0.9}][:k]

    async def delete(self, ids: list[str]) -> None:
        self.deleted.extend(ids)


@pytest.fixture(autouse=True)
def _mock_provider():
    """Register a mock vector provider that returns the shared client."""
    holder: dict[str, _MockNativeClient] = {}

    def _builder(cfg):  # noqa: ANN001
        client = _MockNativeClient()
        holder["client"] = client
        return client

    register_vector_provider("mock_vs_api", _builder)
    yield holder


def test_vector_store_protocol_runtime_checkable():
    """A duck-typed adapter should pass isinstance(store, VectorStore)."""
    conn = VectorConnection(VectorConnectionConfig(name="k", provider="mock_vs_api"))
    store = conn.store()
    assert isinstance(store, VectorStore)


async def test_generic_adapter_upsert_and_query(_mock_provider):
    """The generic adapter should route to the underlying client."""
    conn = VectorConnection(VectorConnectionConfig(name="k", provider="mock_vs_api"))
    store = await conn.as_store()

    ids = await store.upsert(
        texts=["hello", "world"],
        metadatas=[{"src": "a"}, {"src": "b"}],
        ids=["id1", "id2"],
    )
    assert ids == ["id1", "id2"]

    hits = await store.query(text="hello", k=3)
    assert isinstance(hits, list)
    assert hits[0]["id"] == "a"

    await store.delete(["id1"])
    native = _mock_provider["client"]
    assert native.upserts[0]["texts"] == ["hello", "world"]
    assert native.deleted == ["id1"]


async def test_generic_adapter_raises_for_missing_methods():
    """A native client without any known method should raise NotImplementedError."""

    class _Empty:
        pass

    register_vector_provider("empty_provider", lambda cfg: _Empty())
    conn = VectorConnection(VectorConnectionConfig(name="e", provider="empty_provider"))
    store = await conn.as_store()
    with pytest.raises(NotImplementedError):
        await store.upsert(texts=["x"])
    with pytest.raises(NotImplementedError):
        await store.query(text="x")
    with pytest.raises(NotImplementedError):
        await store.delete(["x"])


async def test_register_vector_store_adapter_custom():
    """A registered adapter should be preferred over the generic one."""

    class _CustomAdapter:
        def __init__(self, cfg, client) -> None:  # noqa: ANN001
            self.cfg = cfg
            self.client = client
            self.upserted = False

        async def upsert(self, texts, metadatas=None, ids=None, *, embeddings=None):  # noqa: ANN001, ANN002
            self.upserted = True
            return list(ids or [])

        async def query(self, text=None, *, embedding=None, k=5, filter=None):  # noqa: ANN001
            return []

        async def delete(self, ids):  # noqa: ANN001
            return None

    register_vector_provider("adapter_provider", lambda cfg: object())
    register_vector_store_adapter("adapter_provider", _CustomAdapter)

    conn = VectorConnection(VectorConnectionConfig(name="c", provider="adapter_provider"))
    store = await conn.as_store()
    assert isinstance(store, _CustomAdapter)
    await store.upsert(texts=["x"], ids=["a"])
    assert store.upserted is True
