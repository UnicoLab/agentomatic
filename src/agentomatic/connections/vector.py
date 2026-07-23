"""Vector-store connections for RAG and semantic / vector search.

A :class:`VectorConnection` wraps a provider client (Qdrant, Chroma,
Weaviate, Pinecone, Milvus, or any custom backend…) behind one small,
lazily-initialised object so an agent can obtain an authenticated vector
client with a single call::

    from agentomatic.connections import get_connections

    store = get_connections("rag_agent").vector("kb")
    client = store.client  # native provider client, ready to query

For a provider-agnostic surface — ``upsert`` / ``query`` / ``delete`` —
use :meth:`VectorConnection.store` (or :meth:`VectorConnection.as_store`)
to obtain a :class:`VectorStore`::

    store = get_connections("rag_agent").vector("kb").store()
    await store.upsert(texts=[...], metadatas=[...], ids=[...])
    hits = await store.query(text="what is agentomatic?", k=3)

Providers are pluggable.  The built-ins are registered on import; register
your own (or override a built-in) with :func:`register_vector_provider`::

    from agentomatic.connections.vector import register_vector_provider

    def build_my_store(cfg):
        return MyVectorClient(cfg.url, api_key=cfg.api_key, **cfg.options)

    register_vector_provider("my_store", build_my_store)

To plug in a proprietary / vendor-specific backend (e.g. Azure Cosmos DB),
implement a thin builder and register it — Agentomatic never ships
vendor-specific connectors in-core::

    def build_cosmos(cfg):
        from azure.cosmos import CosmosClient
        return CosmosClient(url=cfg.url, credential=cfg.api_key)

    register_vector_provider("cosmos", build_cosmos)
    # Optional: also register_vector_store_adapter("cosmos", MyCosmosStore)
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from loguru import logger

from agentomatic.connections.models import VectorConnectionConfig
from agentomatic.endpoints.auth import resolve_env

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Provider builders keyed by lowercase provider name.
_VECTOR_PROVIDERS: dict[str, Any] = {}

#: Optional provider-specific :class:`VectorStore` adapter factories.
#: Registered separately so custom clients can plug in an adapter that
#: implements the ``upsert``/``query``/``delete`` surface.
_VECTOR_STORE_ADAPTERS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Provider-agnostic VectorStore API
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Provider-agnostic vector store surface.

    Every backend adapter implements the same three async operations so
    application code can swap providers without changes:

    * :meth:`upsert` writes documents (with metadata) at the given IDs.
    * :meth:`query` retrieves the top-``k`` documents by semantic similarity.
    * :meth:`delete` removes documents by ID.

    Adapters may accept a raw ``text`` query, a pre-computed ``embedding``,
    or both.  Concrete adapters typically wrap the underlying native
    client (see :func:`register_vector_store_adapter`).
    """

    async def upsert(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Insert or update documents.

        Args:
            texts: Raw document texts.
            metadatas: Optional per-document metadata dicts (same length as
                ``texts``).
            ids: Optional stable IDs (auto-generated when omitted).
            embeddings: Optional pre-computed embeddings; if omitted, the
                adapter is expected to embed ``texts`` itself.

        Returns:
            The list of IDs that were written.
        """
        ...

    async def query(
        self,
        text: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-``k`` most similar documents.

        Args:
            text: Query text (adapter must embed if ``embedding`` is None).
            embedding: Pre-computed query embedding.
            k: Number of results to return.
            filter: Optional provider-specific metadata filter.

        Returns:
            Ordered list of hit dicts with at least ``id``, ``score``,
            ``text`` and ``metadata`` keys.
        """
        ...

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete documents by ID."""
        ...


def register_vector_provider(name: str, builder: Callable[[VectorConnectionConfig], Any]) -> None:
    """Register (or override) a vector-store provider builder.

    Args:
        name: Provider identifier used in ``VectorConnectionConfig.provider``.
        builder: Callable that takes the resolved config and returns a native
            client instance.  It may raise :class:`ImportError` with an
            install hint when the backing library is missing.
    """
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("vector provider name must be a non-empty string")
    if builder is None:
        raise ValueError(f"vector provider builder for '{key}' must not be None")
    _VECTOR_PROVIDERS[key] = builder


def register_vector_store_adapter(
    name: str,
    adapter: Callable[[VectorConnectionConfig, Any], VectorStore],
) -> None:
    """Register a :class:`VectorStore` adapter factory for a provider.

    Args:
        name: Provider identifier (case-insensitive).
        adapter: Callable ``(config, native_client) -> VectorStore`` that
            wraps the provider's native client into the provider-agnostic
            :class:`VectorStore` surface.
    """
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("vector store adapter name must be a non-empty string")
    if adapter is None:
        raise ValueError(f"vector store adapter for '{key}' must not be None")
    _VECTOR_STORE_ADAPTERS[key] = adapter


def registered_vector_providers() -> list[str]:
    """Return the names of all registered vector providers."""
    return sorted(_VECTOR_PROVIDERS)


class VectorConnection:
    """A lazily-initialised vector-store client bound to one provider."""

    def __init__(self, config: VectorConnectionConfig) -> None:
        self.config = config
        self._client: Any = None
        self._store: VectorStore | None = None

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
        provider_key = str(self.config.provider or "").strip().lower()
        builder = _VECTOR_PROVIDERS.get(provider_key)
        if builder is None:
            # Dual-import guard: custom providers may have registered on a
            # sibling module copy when PYTHONPATH + site-packages both ship
            # agentomatic. Merge any sibling registries before failing.
            import sys

            for mod in list(sys.modules.values()):
                if getattr(mod, "__name__", "").endswith("agentomatic.connections.vector"):
                    sibling = getattr(mod, "_VECTOR_PROVIDERS", None)
                    if isinstance(sibling, dict) and provider_key in sibling:
                        builder = sibling.get(provider_key)
                        if builder is not None:
                            _VECTOR_PROVIDERS[provider_key] = builder
                            break
        if builder is None:
            raise ValueError(
                f"Unknown vector provider '{self.config.provider}' for connection "
                f"'{self.name}'. Registered providers: {registered_vector_providers()}. "
                "Register a custom one with `register_vector_provider(name, builder)`."
            )
        client = builder(self.config)
        logger.info(f"🧭 Vector connection '{self.name}' ready (provider={self.config.provider})")
        return client

    def store(self) -> VectorStore:
        """Return a provider-agnostic :class:`VectorStore` adapter.

        Adapters are cached per connection.  If the provider registered a
        custom adapter via :func:`register_vector_store_adapter`, it is
        used; otherwise a :class:`_GenericVectorStoreAdapter` wraps the
        native client with best-effort duck-typed dispatch.
        """
        if self._store is not None:
            return self._store
        if self._client is None:
            self._client = self._build_client()
        adapter = _VECTOR_STORE_ADAPTERS.get(str(self.config.provider or "").strip().lower())
        if adapter is not None:
            self._store = adapter(self.config, self._client)
        else:
            self._store = _GenericVectorStoreAdapter(self.config, self._client)
        return self._store

    async def as_store(self) -> VectorStore:
        """Async variant of :meth:`store` that also awaits initialisation."""
        await self.initialize()
        return self.store()

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
        """Close the underlying client if it exposes a shutdown method.

        Any client is accepted: the first of ``aclose`` / ``close`` /
        ``disconnect`` that exists is called (awaited when it returns an
        awaitable).  Clients exposing none of these are dropped silently.
        """
        client = self._client
        if client is None:
            return
        for attr in ("aclose", "close", "disconnect"):
            fn = getattr(client, attr, None)
            if not callable(fn):
                continue
            try:
                result = fn()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Vector connection '{self.name}' close error: {exc}")
            break
        self._client = None
        self._store = None


# ---------------------------------------------------------------------------
# Generic duck-typed VectorStore adapter
# ---------------------------------------------------------------------------


class _GenericVectorStoreAdapter:
    """Best-effort :class:`VectorStore` wrapper over an arbitrary client.

    The adapter attempts to dispatch to well-known methods on the native
    client (``upsert`` / ``add`` / ``query`` / ``search`` / ``delete``).
    When the underlying client does not expose any of them, calls raise
    :class:`NotImplementedError` — register a provider-specific adapter
    via :func:`register_vector_store_adapter` for full support.
    """

    def __init__(self, config: VectorConnectionConfig, client: Any) -> None:
        self._config = config
        self._client = client

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _pick(self, *candidates: str) -> Any:
        for name in candidates:
            fn = getattr(self._client, name, None)
            if callable(fn):
                return fn
        return None

    async def upsert(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Route to the client's native upsert/add method when available."""
        fn = self._pick("upsert", "add", "add_texts", "add_documents")
        if fn is None:
            raise NotImplementedError(
                f"Provider '{self._config.provider}' has no generic upsert; register "
                "an adapter with `register_vector_store_adapter`."
            )
        kwargs: dict[str, Any] = {"texts": list(texts)}
        if metadatas is not None:
            kwargs["metadatas"] = list(metadatas)
        if ids is not None:
            kwargs["ids"] = list(ids)
        if embeddings is not None:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        try:
            result = await self._maybe_await(fn(**kwargs))
        except TypeError:
            result = await self._maybe_await(fn(list(texts)))
        if isinstance(result, list):
            return [str(x) for x in result]
        return list(ids) if ids is not None else []

    async def query(
        self,
        text: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Route to the client's native query/search method when available."""
        fn = self._pick("query", "search", "similarity_search")
        if fn is None:
            raise NotImplementedError(
                f"Provider '{self._config.provider}' has no generic query; register "
                "an adapter with `register_vector_store_adapter`."
            )
        kwargs: dict[str, Any] = {"k": k}
        if text is not None:
            kwargs["text"] = text
        if embedding is not None:
            kwargs["embedding"] = list(embedding)
        if filter is not None:
            kwargs["filter"] = filter
        try:
            result = await self._maybe_await(fn(**kwargs))
        except TypeError:
            result = await self._maybe_await(fn(text or embedding, k))
        if isinstance(result, list):
            return [x if isinstance(x, dict) else {"item": x} for x in result]
        return []

    async def delete(self, ids: Sequence[str]) -> None:
        """Route to the client's native delete method when available."""
        fn = self._pick("delete", "delete_many", "remove")
        if fn is None:
            raise NotImplementedError(
                f"Provider '{self._config.provider}' has no generic delete; register "
                "an adapter with `register_vector_store_adapter`."
            )
        try:
            await self._maybe_await(fn(ids=list(ids)))
        except TypeError:
            await self._maybe_await(fn(list(ids)))


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

# Zero-infra local .npz backend (soft-deps on numpy).
try:
    from agentomatic.connections.local_npz import register_local_npz_backends

    register_local_npz_backends()
except Exception:  # noqa: BLE001 - optional; ImportError if numpy missing later at use
    pass
