"""Embedding provider factory.

Provides a small, cached factory over embedding backends. Unlike a naive
singleton, instances are cached **per (provider, kwargs)** so multiple distinct
embedding models can coexist in one process.

Built-in providers:

* ``ollama`` — local Ollama embeddings (requires ``langchain-ollama``)
* ``openai`` — OpenAI embeddings (requires ``langchain-openai``)
* ``azure_openai`` — Azure OpenAI embeddings (requires ``langchain-openai``)
* ``hash`` — a deterministic, dependency-free hash embedder (great for tests
  and offline development)
* ``dummy`` — deterministic fake embeddings (requires ``langchain-core``),
  falling back to ``hash`` if unavailable

Register additional providers with :func:`register_embedding_provider`::

    from agentomatic.providers.embeddings import register_embedding_provider

    def build_cohere(**kwargs):
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(**kwargs)

    register_embedding_provider("cohere", build_cohere)
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

_embeddings_cache: dict[str, Any] = {}

#: Provider builders keyed by lowercase provider name.
_EMBEDDING_PROVIDERS: dict[str, Any] = {}


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


def register_embedding_provider(
    name: str,
    builder: Callable[..., Any],
) -> None:
    """Register (or override) an embedding provider builder.

    Args:
        name: Provider identifier (case-insensitive) used with
            :func:`get_embeddings`.
        builder: Callable ``(**kwargs) -> embedder`` that returns an object
            exposing ``embed_documents`` and ``embed_query``.  May raise
            :class:`ImportError` with an install hint when the backing
            library is missing.
    """
    _EMBEDDING_PROVIDERS[name.lower()] = builder


def registered_embedding_providers() -> list[str]:
    """Return the names of all registered embedding providers."""
    return sorted(_EMBEDDING_PROVIDERS)


def _cache_key(provider: str, kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for a provider + kwargs combination."""
    return provider + ":" + json.dumps(kwargs, sort_keys=True, default=str)


def get_embeddings(provider: str = "hash", **kwargs: Any) -> Any:
    """Get or create a cached embeddings instance for ``provider``.

    Args:
        provider: One of ``ollama``, ``openai``, ``azure_openai``, ``hash``,
            ``dummy`` — or any name registered with
            :func:`register_embedding_provider`.
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
    provider_lc = provider.lower()
    builder = _EMBEDDING_PROVIDERS.get(provider_lc)
    if builder is not None:
        try:
            return builder(**kwargs)
        except Exception as exc:  # noqa: BLE001 - fall back to the hash embedder
            logger.warning(f"Failed to build '{provider}' embeddings: {exc}. Using hash embedder.")
            return HashEmbedder(dimension=kwargs.get("dimension", 256))

    logger.warning(
        f"Unknown embedding provider '{provider}'. Registered: "
        f"{registered_embedding_providers()}. Falling back to hash embedder."
    )
    return HashEmbedder(dimension=kwargs.get("dimension", 256))


# ---------------------------------------------------------------------------
# Built-in provider builders (lazy, optional dependencies)
# ---------------------------------------------------------------------------


def _build_ollama_embeddings(**kwargs: Any) -> Any:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=kwargs.get("model", "nomic-embed-text"),
        base_url=kwargs.get("base_url", "http://localhost:11434"),
    )


def _build_openai_embeddings(**kwargs: Any) -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=kwargs.get("model", "text-embedding-3-small"),
        **{k: v for k, v in kwargs.items() if k != "model"},
    )


def _build_azure_openai_embeddings(**kwargs: Any) -> Any:
    """Build Azure OpenAI embeddings via ``langchain_openai``.

    Accepts the following keys (with common aliases mapped to the
    canonical LangChain names):

    * ``model`` / ``azure_deployment`` / ``deployment_name`` — the Azure
      deployment name.
    * ``api_base`` / ``base_url`` / ``azure_endpoint`` — the Azure endpoint.
    * ``api_key``, ``api_version`` — Azure credentials.
    """
    try:
        from langchain_openai import AzureOpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "langchain-openai is required for the 'azure_openai' embeddings provider. "
            "Install with: pip install 'agentomatic[azure]'"
        ) from exc

    deployment = (
        kwargs.get("azure_deployment")
        or kwargs.get("deployment_name")
        or kwargs.get("model")
        or "text-embedding-3-small"
    )
    endpoint = (
        kwargs.get("azure_endpoint") or kwargs.get("api_base") or kwargs.get("base_url") or ""
    )
    api_version = kwargs.get("api_version", "2024-02-15-preview")
    extras = {
        k: v
        for k, v in kwargs.items()
        if k
        not in {
            "model",
            "azure_deployment",
            "deployment_name",
            "azure_endpoint",
            "api_base",
            "base_url",
            "api_version",
        }
    }
    return AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_version=api_version,
        **extras,
    )


def _build_hash_embeddings(**kwargs: Any) -> Any:
    return HashEmbedder(dimension=kwargs.get("dimension", 256))


def _build_dummy_embeddings(**kwargs: Any) -> Any:
    try:
        from langchain_core.embeddings import DeterministicFakeEmbedding
    except ImportError:  # pragma: no cover - optional dep
        return HashEmbedder(dimension=kwargs.get("dimension", 256))
    return DeterministicFakeEmbedding(size=kwargs.get("dimension", 768))


register_embedding_provider("ollama", _build_ollama_embeddings)
register_embedding_provider("openai", _build_openai_embeddings)
register_embedding_provider("azure_openai", _build_azure_openai_embeddings)
register_embedding_provider("hash", _build_hash_embeddings)
register_embedding_provider("dummy", _build_dummy_embeddings)


def reset_embeddings() -> None:
    """Clear the embeddings cache (primarily for tests)."""
    _embeddings_cache.clear()
