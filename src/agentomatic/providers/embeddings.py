"""Embedding provider factory.

Provides a small, cached factory over embedding backends. Unlike a naive
singleton, instances are cached **per (provider, kwargs)** so multiple distinct
embedding models can coexist in one process.

Supported providers:

* ``ollama`` — local Ollama embeddings (requires ``langchain-ollama``)
* ``openai`` — OpenAI embeddings (requires ``langchain-openai``)
* ``hash`` — a deterministic, dependency-free hash embedder (great for tests
  and offline development)
* ``dummy`` — deterministic fake embeddings (requires ``langchain-core``),
  falling back to ``hash`` if unavailable
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from loguru import logger

_embeddings_cache: dict[str, Any] = {}


class HashEmbedder:
    """Deterministic, dependency-free embedder based on hashed token buckets.

    Produces stable, normalised vectors of a fixed dimension. Useful for tests,
    offline development, and small deployments where semantic quality is not
    critical but reproducibility and zero dependencies are.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha1(token.encode("utf-8")).digest()  # noqa: S324
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._vector(text)


def _cache_key(provider: str, kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for a provider + kwargs combination."""
    return provider + ":" + json.dumps(kwargs, sort_keys=True, default=str)


def get_embeddings(provider: str = "hash", **kwargs: Any) -> Any:
    """Get or create a cached embeddings instance for ``provider``.

    Args:
        provider: One of ``ollama``, ``openai``, ``hash``, ``dummy``.
        **kwargs: Provider-specific options (e.g. ``model``, ``base_url``,
            ``dimension``).

    Returns:
        An embeddings object exposing ``embed_documents`` / ``embed_query``.
    """
    key = _cache_key(provider, kwargs)
    cached = _embeddings_cache.get(key)
    if cached is not None:
        return cached

    instance = _build_embeddings(provider, kwargs)
    _embeddings_cache[key] = instance
    return instance


def _build_embeddings(provider: str, kwargs: dict[str, Any]) -> Any:
    """Construct a fresh embeddings instance (uncached)."""
    try:
        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(
                model=kwargs.get("model", "nomic-embed-text"),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
            )
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=kwargs.get("model", "text-embedding-3-small"),
                **{k: v for k, v in kwargs.items() if k != "model"},
            )
        if provider == "hash":
            return HashEmbedder(dimension=kwargs.get("dimension", 256))
        # "dummy" / default
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=kwargs.get("dimension", 768))
    except Exception as exc:  # noqa: BLE001 - fall back to the dependency-free embedder
        logger.warning(f"Failed to build '{provider}' embeddings: {exc}. Using hash embedder.")
        return HashEmbedder(dimension=kwargs.get("dimension", 256))


def reset_embeddings() -> None:
    """Clear the embeddings cache (primarily for tests)."""
    _embeddings_cache.clear()
