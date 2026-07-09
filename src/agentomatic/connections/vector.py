"""Vector-store connections for RAG and semantic / vector search.

A :class:`VectorConnection` wraps a provider client (Qdrant, Chroma,
Weaviate, Pinecone, Milvus, …) behind one small, lazily-initialised object
so an agent can obtain an authenticated vector client with a single call::

    from agentomatic.connections import get_connections

    store = get_connections("rag_agent").vector("kb")
    client = store.client  # native provider client, ready to query

Providers are pluggable.  The built-ins are registered on import; register
your own (or override a built-in) with :func:`register_vector_provider`::

    from agentomatic.connections.vector import register_vector_provider

    def build_my_store(cfg):
        return MyVectorClient(cfg.url, api_key=cfg.api_key, **cfg.options)

    register_vector_provider("my_store", build_my_store)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from loguru import logger

from agentomatic.connections.models import VectorConnectionConfig
from agentomatic.endpoints.auth import resolve_env

if TYPE_CHECKING:
    from collections.abc import Callable

#: Provider builders keyed by lowercase provider name.
_VECTOR_PROVIDERS: dict[str, Any] = {}


def register_vector_provider(name: str, builder: Callable[[VectorConnectionConfig], Any]) -> None:
    """Register (or override) a vector-store provider builder.

    Args:
        name: Provider identifier used in ``VectorConnectionConfig.provider``.
        builder: Callable that takes the resolved config and returns a native
            client instance.  It may raise :class:`ImportError` with an
            install hint when the backing library is missing.
    """
    _VECTOR_PROVIDERS[name.lower()] = builder


def registered_vector_providers() -> list[str]:
    """Return the names of all registered vector providers."""
    return sorted(_VECTOR_PROVIDERS)


class VectorConnection:
    """A lazily-initialised vector-store client bound to one provider."""

    def __init__(self, config: VectorConnectionConfig) -> None:
        self.config = config
        self._client: Any = None

    @property
    def name(self) -> str:
        """Return the connection's logical name."""
        return self.config.name

    @property
    def provider(self) -> str:
        """Return the configured provider name."""
        return self.config.provider

    @property
    def collection(self) -> str:
        """Return the configured default collection / index name."""
        return self.config.collection

    async def initialize(self) -> None:
        """Build the underlying client (idempotent)."""
        if self._client is None:
            self._client = self._build_client()

    @property
    def client(self) -> Any:
        """Return the native provider client, building it on first access."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        builder = _VECTOR_PROVIDERS.get(self.config.provider.lower())
        if builder is None:
            raise ValueError(
                f"Unknown vector provider '{self.config.provider}' for connection "
                f"'{self.name}'. Registered providers: {registered_vector_providers()}. "
                "Register a custom one with `register_vector_provider(name, builder)`."
            )
        client = builder(self.config)
        logger.info(f"🧭 Vector connection '{self.name}' ready (provider={self.config.provider})")
        return client

    async def health_check(self) -> dict[str, Any]:
        """Report provider-level health (best effort, no network guarantee)."""
        base = {
            "connection": self.name,
            "kind": "vector",
            "provider": self.config.provider,
            "collection": self.config.collection,
        }
        try:
            self.initialize_sync()
        except Exception as exc:  # noqa: BLE001
            return {**base, "status": "unhealthy", "error": str(exc)}
        return {**base, "status": "healthy" if self._client is not None else "configured"}

    def initialize_sync(self) -> None:
        """Synchronously ensure the client is built (used by health checks)."""
        if self._client is None:
            self._client = self._build_client()

    async def close(self) -> None:
        """Close the underlying client if it exposes a close method."""
        client = self._client
        if client is None:
            return
        for attr in ("aclose", "close"):
            fn = getattr(client, attr, None)
            if fn is None:
                continue
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Vector connection '{self.name}' close error: {exc}")
            break
        self._client = None


# ---------------------------------------------------------------------------
# Built-in provider builders (lazy, optional dependencies)
# ---------------------------------------------------------------------------


def _host_port(url: str, default_port: int) -> tuple[str, int]:
    """Split a ``host:port`` / URL into ``(host, port)``."""
    parts = urlsplit(url if "//" in url else f"//{url}")
    return parts.hostname or "localhost", parts.port or default_port


def _build_qdrant(cfg: VectorConnectionConfig) -> Any:
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "qdrant-client is required for the 'qdrant' provider. "
            "Install with: pip install qdrant-client"
        ) from exc
    return AsyncQdrantClient(
        url=resolve_env(cfg.url) or None,
        api_key=resolve_env(cfg.api_key) or None,
        **cfg.options,
    )


def _build_chroma(cfg: VectorConnectionConfig) -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "chromadb is required for the 'chroma' provider. Install with: pip install chromadb"
        ) from exc
    url = resolve_env(cfg.url)
    if url:
        host, port = _host_port(url, 8000)
        return chromadb.HttpClient(host=host, port=port, **cfg.options)
    return chromadb.Client(**cfg.options)


def _build_weaviate(cfg: VectorConnectionConfig) -> Any:
    try:
        import weaviate
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "weaviate-client is required for the 'weaviate' provider. "
            "Install with: pip install weaviate-client"
        ) from exc
    return weaviate.connect_to_custom(  # type: ignore[attr-defined]
        http_host=resolve_env(cfg.url) or "localhost",
        **cfg.options,
    )


def _build_pinecone(cfg: VectorConnectionConfig) -> Any:
    try:
        from pinecone import Pinecone
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "pinecone is required for the 'pinecone' provider. Install with: pip install pinecone"
        ) from exc
    return Pinecone(api_key=resolve_env(cfg.api_key), **cfg.options)


def _build_milvus(cfg: VectorConnectionConfig) -> Any:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "pymilvus is required for the 'milvus' provider. Install with: pip install pymilvus"
        ) from exc
    return MilvusClient(
        uri=resolve_env(cfg.url),
        token=resolve_env(cfg.api_key) or None,
        **cfg.options,
    )


register_vector_provider("qdrant", _build_qdrant)
register_vector_provider("chroma", _build_chroma)
register_vector_provider("weaviate", _build_weaviate)
register_vector_provider("pinecone", _build_pinecone)
register_vector_provider("milvus", _build_milvus)
