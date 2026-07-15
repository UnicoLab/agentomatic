"""Tests for the extended embeddings registry (azure_openai + register hook)."""

from __future__ import annotations

import pytest

from agentomatic.providers.embeddings import (
    get_embeddings,
    register_embedding_provider,
    registered_embedding_providers,
    reset_embeddings,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_embeddings()
    yield
    reset_embeddings()


def test_builtin_providers_registered():
    providers = registered_embedding_providers()
    for name in ("ollama", "openai", "azure_openai", "hash", "dummy"):
        assert name in providers


def test_hash_provider_still_works():
    emb = get_embeddings("hash", dimension=32)
    vec = emb.embed_query("hello world")
    assert len(vec) == 32


def test_register_embedding_provider_custom():
    class _StubEmbedder:
        def __init__(self, dim=8):
            self.dim = dim

        def embed_documents(self, texts):
            return [[0.0] * self.dim for _ in texts]

        def embed_query(self, text):  # noqa: ARG002
            return [1.0] * self.dim

    register_embedding_provider("stub_test", lambda **kw: _StubEmbedder(dim=kw.get("dim", 4)))
    assert "stub_test" in registered_embedding_providers()
    emb = get_embeddings("stub_test", dim=16)
    assert isinstance(emb, _StubEmbedder)
    assert emb.dim == 16


def test_azure_openai_falls_back_to_hash_when_missing(monkeypatch):
    """When azure_openai builder raises, we fall back to a hash embedder."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langchain_openai" or name.startswith("langchain_openai."):
            raise ImportError("langchain_openai is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    emb = get_embeddings("azure_openai", model="text-embedding-3-small", dimension=64)
    vec = emb.embed_query("hello")
    assert len(vec) == 64


def test_unknown_provider_falls_back_to_hash():
    emb = get_embeddings("does_not_exist", dimension=32)
    vec = emb.embed_query("hello")
    assert len(vec) == 32
