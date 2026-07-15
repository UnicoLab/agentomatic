"""Tests for pluggable vector providers (user-owned integrations).

Agentomatic ships ops (``register_vector_provider``, ``VectorStore`` Protocol)
— vendor SDKs (Cosmos, etc.) are registered by the user.
"""

from __future__ import annotations

import importlib

import pytest

from agentomatic.connections import (
    VectorConnection,
    VectorConnectionConfig,
    register_vector_provider,
    registered_vector_providers,
)
from agentomatic.connections.vector import register_vector_store_adapter


def test_built_in_providers_do_not_include_vendor_specific() -> None:
    """Core must stay vendor-agnostic — no first-party Cosmos connectors."""
    providers = registered_vector_providers()
    assert "cosmos_nosql" not in providers
    assert "cosmos_mongo_vcore" not in providers
    for name in ("qdrant", "chroma", "weaviate", "pinecone", "milvus"):
        assert name in providers


def test_user_can_register_custom_provider() -> None:
    """Users own vendor integrations via register_vector_provider."""
    built: list[str] = []

    def build_fake(cfg: VectorConnectionConfig) -> object:
        built.append(cfg.name)

        class _Client:
            def ping(self) -> str:
                return "ok"

        return _Client()

    register_vector_provider("my_cosmos", build_fake)
    try:
        assert "my_cosmos" in registered_vector_providers()
        conn = VectorConnection(
            VectorConnectionConfig(
                name="kb",
                provider="my_cosmos",
                url="https://example.invalid",
                api_key="secret",
                options={"database": "rag"},
            )
        )
        client = conn.client
        assert client.ping() == "ok"
        assert built == ["kb"]
    finally:
        # Leave the registry clean for other tests
        from agentomatic.connections import vector as vmod

        vmod._VECTOR_PROVIDERS.pop("my_cosmos", None)


async def test_user_can_register_vector_store_adapter() -> None:
    """Optional VectorStore adapter is also user-pluggable."""

    class _FakeClient:
        pass

    class _FakeStore:
        def __init__(self, config: VectorConnectionConfig, client: object) -> None:
            self.config = config
            self.client = client
            self.upserted: list[str] = []

        async def upsert(self, texts, metadatas=None, ids=None, embeddings=None):
            self.upserted.extend(texts)
            return list(ids or [])

        async def query(self, *, text=None, embedding=None, k=5, filter=None):
            return [{"id": "1", "text": text or "", "score": 1.0, "metadata": {}}]

        async def delete(self, ids):
            return None

    register_vector_provider("fake_vs", lambda cfg: _FakeClient())
    register_vector_store_adapter("fake_vs", _FakeStore)
    try:
        conn = VectorConnection(
            VectorConnectionConfig(name="kb", provider="fake_vs", url="mem://")
        )
        store = conn.store()
        ids = await store.upsert(texts=["hello"], ids=["1"])
        assert ids == ["1"]
        hits = await store.query(text="hello", k=1)
        assert hits[0]["id"] == "1"
    finally:
        from agentomatic.connections import vector as vmod

        vmod._VECTOR_PROVIDERS.pop("fake_vs", None)
        vmod._VECTOR_STORE_ADAPTERS.pop("fake_vs", None)


def test_unknown_provider_raises_with_register_hint() -> None:
    """Unknown providers must point users at the registration hook."""
    conn = VectorConnection(VectorConnectionConfig(name="kb", provider="not_a_real_backend"))
    with pytest.raises(ValueError, match="register_vector_provider"):
        _ = conn.client


def test_vector_module_reimport_stable() -> None:
    """Re-importing the vector module must keep built-ins registered."""
    import agentomatic.connections.vector as mod

    importlib.reload(mod)
    assert "qdrant" in mod.registered_vector_providers()
    assert "cosmos_nosql" not in mod.registered_vector_providers()
