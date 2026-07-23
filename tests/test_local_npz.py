"""Tests for local_npz vector store and TextEncoder hash fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from agentomatic.connections.encoder import TextEncoder, hash_embed
from agentomatic.connections.local_npz import (
    EmbeddingStore,
    LocalNpzVectorStore,
    register_local_npz_backends,
)
from agentomatic.connections.vector import registered_vector_providers


def test_hash_embed_deterministic() -> None:
    """Hash embeddings are L2-normalised and deterministic."""
    a = hash_embed(["hello world"], dimension=32)
    b = hash_embed(["hello world"], dimension=32)
    assert a.shape == (1, 32)
    assert abs(float((a * a).sum()) - 1.0) < 1e-6
    assert (a == b).all()


def test_text_encoder_hash_fallback() -> None:
    """Dummy provider settings force hash fallback."""

    class _Emb:
        provider = "dummy"
        model = "x"
        dimension = 16
        base_url = ""
        api_key = ""
        enabled = True

    class _Settings:
        embedding = _Emb()

    enc = TextEncoder.from_settings(_Settings())
    vec = enc.encode(["alpha beta"])
    assert vec.shape == (1, 16)


@pytest.mark.asyncio
async def test_local_npz_upsert_query_roundtrip(tmp_path: Path) -> None:
    """Upsert then query returns the nearest stored row."""
    enc = TextEncoder(provider="hash", dimension=32, enabled=False)
    store = LocalNpzVectorStore(tmp_path / "vecs", encoder=enc)
    await store.upsert(
        texts=["cats and dogs", "quantum physics"],
        ids=["a", "b"],
        metadatas=[{"k": 1}, {"k": 2}],
    )
    hits = await store.query(text="cats and dogs", k=1)
    assert hits
    assert hits[0]["id"] == "a"
    assert hits[0]["text"] == "cats and dogs"


@pytest.mark.asyncio
async def test_local_npz_upsert_replaces_same_id(tmp_path: Path) -> None:
    """Upsert with an existing id replaces the row instead of appending."""
    enc = TextEncoder(provider="hash", dimension=16, enabled=False)
    store = LocalNpzVectorStore(tmp_path / "vecs", encoder=enc)
    await store.upsert(texts=["first"], ids=["same"], metadatas=[{"v": 1}])
    await store.upsert(texts=["second"], ids=["same"], metadatas=[{"v": 2}])
    assert EmbeddingStore(tmp_path / "vecs").count() == 1
    hits = await store.query(text="second", k=1)
    assert hits[0]["id"] == "same"
    assert hits[0]["text"] == "second"
    assert hits[0]["metadata"]["v"] == 2


@pytest.mark.asyncio
async def test_local_npz_delete(tmp_path: Path) -> None:
    """Delete removes rows by id."""
    enc = TextEncoder(provider="hash", dimension=16, enabled=False)
    store = LocalNpzVectorStore(tmp_path / "vecs", encoder=enc)
    await store.upsert(texts=["one", "two"], ids=["1", "2"])
    await store.delete(["1"])
    emb_store = EmbeddingStore(tmp_path / "vecs")
    assert emb_store.count() == 1


def test_local_npz_registered() -> None:
    """local_npz is a registered vector provider."""
    register_local_npz_backends()
    providers = registered_vector_providers()
    assert "local_npz" in providers
