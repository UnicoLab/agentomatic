"""Azure Cosmos DB for MongoDB vCore — thin Agentomatic adapter.

Uses the Microsoft-maintained LangChain integration
(``langchain-azure-ai`` → :class:`AzureCosmosDBMongoVCoreVectorSearch`) for
index creation and ``cosmosSearch`` queries. We only add:

* Agentomatic ``VectorStore`` surface (``upsert`` / ``query`` / ``delete``)
* Stable string ids (case_id) for historical-case upserts
* Cosine fallback against plain Mongo (compose profile ``cosmos``) when
  ``cosmosSearch`` is unavailable

Never construct clients in ``BaseGraphAgent.__init__`` — register via
:func:`register_vector_provider` / :func:`register_vector_store_adapter`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote_plus

import numpy as np

from ai_core.embeddings import TextEncoder

_DEFAULT_VECTOR_FIELD = "vectorContent"
_DEFAULT_TEXT_FIELD = "textContent"
_DEFAULT_INDEX_NAME = "vectorSearchIndex"


def _env(name: str, default: str = "") -> str:
    """Return a stripped env value or *default*."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env flag."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_local_connection_string() -> str:
    """Build a local Mongo connection string from host/user/password env."""
    host = _env("AI_COSMOS_HOST", "localhost")
    port = _env("AI_COSMOS_PORT", "27017")
    user = _env("AI_COSMOS_USER", "scooper")
    password = _env("AI_COSMOS_PASSWORD", "scooper")
    auth_source = _env("AI_COSMOS_AUTH_SOURCE", "admin")
    if user:
        return (
            f"mongodb://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/?authSource={quote_plus(auth_source)}"
        )
    return f"mongodb://{host}:{port}/"


class TextEncoderEmbeddings:
    """LangChain ``Embeddings`` shim over :class:`TextEncoder`."""

    def __init__(self, encoder: TextEncoder | None = None) -> None:
        self._encoder = encoder or TextEncoder.from_settings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        matrix = self._encoder.encode(list(texts))
        return [list(map(float, row)) for row in matrix]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return list(map(float, self._encoder.encode_one(text)))


def cosmos_settings_from_env(config: Any | None = None) -> dict[str, Any]:
    """Resolve Cosmos Mongo connection knobs from config + environment."""
    options: dict[str, Any] = dict(getattr(config, "options", None) or {})
    connection_string = (
        options.get("connection_string")
        or getattr(config, "url", None)
        or _env("AI_COSMOS_CONNECTION_STRING")
        or _default_local_connection_string()
    )
    database = options.get("database") or _env("AI_COSMOS_DATABASE", "scooper")
    collection = (
        getattr(config, "collection", None)
        or options.get("collection")
        or options.get("container")
        or _env("AI_COSMOS_COLLECTION")
        or _env("AI_COSMOS_CONTAINER", "cases")
    )
    dimension = (
        getattr(config, "dimension", None)
        or options.get("dimension")
        or int(_env("AI_EMBED_DIMENSION", "768"))
    )
    distance = getattr(config, "distance", None) or options.get("distance") or "cosine"
    vector_field = (
        options.get("vector_field")
        or options.get("vector_path")
        or _env("AI_COSMOS_VECTOR_FIELD", _DEFAULT_VECTOR_FIELD)
    ).lstrip("/")
    text_field = options.get("text_field") or _env("AI_COSMOS_TEXT_FIELD", _DEFAULT_TEXT_FIELD)
    index_name = options.get("index_name") or _env("AI_COSMOS_INDEX_NAME", _DEFAULT_INDEX_NAME)
    index_kind = options.get("index_kind") or _env("AI_COSMOS_INDEX_KIND", "vector-ivf")
    num_lists = int(options.get("num_lists") or _env("AI_COSMOS_NUM_LISTS", "1"))
    use_cosmos_search = options.get(
        "use_cosmos_search",
        _env_bool("AI_COSMOS_USE_COSMOS_SEARCH", True),
    )
    return {
        "connection_string": str(connection_string),
        "database": str(database),
        "collection": str(collection),
        "namespace": f"{database}.{collection}",
        "dimension": int(dimension),
        "distance": str(distance).lower(),
        "vector_field": str(vector_field),
        "text_field": str(text_field),
        "index_name": str(index_name),
        "index_kind": str(index_kind),
        "num_lists": num_lists,
        "use_cosmos_search": bool(use_cosmos_search),
    }


def build_cosmos_client(config: Any) -> Any:
    """Build the official LangChain Cosmos Mongo vCore vector store.

    Args:
        config: :class:`VectorConnectionConfig` from Agentomatic.

    Returns:
        ``AzureCosmosDBMongoVCoreVectorSearch`` instance (native client).

    Raises:
        RuntimeError: When dependencies or connection string are missing.
    """
    try:
        from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
            AzureCosmosDBMongoVCoreVectorSearch,
            CosmosDBSimilarityType,
            CosmosDBVectorSearchType,
        )
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Azure Cosmos Mongo vector provider requires 'langchain-azure-ai' "
            "and 'pymongo'. Install with: "
            "uv add langchain-azure-ai pymongo"
        ) from exc

    settings = cosmos_settings_from_env(config)
    if not settings["connection_string"]:
        raise RuntimeError(
            "AI_COSMOS_CONNECTION_STRING (or VectorConnectionConfig.url) is "
            "required for provider=azure_cosmos (MongoDB vCore)."
        )

    embeddings = TextEncoderEmbeddings(TextEncoder.from_settings())
    store = AzureCosmosDBMongoVCoreVectorSearch.from_connection_string(
        settings["connection_string"],
        settings["namespace"],
        embeddings,
        application_name="scooper-ai-platform",
        index_name=settings["index_name"],
        text_key=settings["text_field"],
        embedding_key=settings["vector_field"],
    )
    # Touch server early.
    store._collection.database.client.admin.command("ping")

    if settings["use_cosmos_search"]:
        try:
            if not store.index_exists():
                kind_map = {
                    "vector-ivf": CosmosDBVectorSearchType.VECTOR_IVF,
                    "vector-hnsw": CosmosDBVectorSearchType.VECTOR_HNSW,
                    "vector-diskann": CosmosDBVectorSearchType.VECTOR_DISKANN,
                }
                kind = kind_map.get(settings["index_kind"], CosmosDBVectorSearchType.VECTOR_IVF)
                distance_key = settings["distance"].replace(" ", "").replace("_", "")
                similarity = {
                    "cosine": CosmosDBSimilarityType.COS,
                    "cos": CosmosDBSimilarityType.COS,
                    "euclidean": CosmosDBSimilarityType.L2,
                    "l2": CosmosDBSimilarityType.L2,
                    "dot": CosmosDBSimilarityType.IP,
                    "dotproduct": CosmosDBSimilarityType.IP,
                    "ip": CosmosDBSimilarityType.IP,
                }.get(distance_key, CosmosDBSimilarityType.COS)
                # Pass the enum — create_index compares with ``==`` on the enum.
                store.create_index(
                    num_lists=settings["num_lists"],
                    dimensions=settings["dimension"],
                    similarity=similarity,
                    kind=kind,
                )
        except Exception:  # noqa: BLE001 - plain local Mongo has no cosmosSearch
            settings["use_cosmos_search"] = False

    store._scooper_cosmos = settings  # type: ignore[attr-defined]
    return store


class AzureCosmosVectorStore:
    """Agentomatic ``VectorStore`` wrapper over the LangChain Cosmos Mongo store."""

    def __init__(
        self,
        config: Any,
        client: Any,
        *,
        encoder: TextEncoder | None = None,
    ) -> None:
        """Wrap an ``AzureCosmosDBMongoVCoreVectorSearch`` client.

        Args:
            config: Vector connection config.
            client: LangChain store from :func:`build_cosmos_client`.
            encoder: Optional text encoder (used for precomputed / fallback).
        """
        self._config = config
        self._lc = client
        self._encoder = encoder or TextEncoder.from_settings()
        self._settings = getattr(client, "_scooper_cosmos", None) or cosmos_settings_from_env(
            config
        )
        self._collection = client._collection
        self._text_field = getattr(client, "_text_key", self._settings["text_field"])
        self._vector_field = getattr(client, "_embedding_key", self._settings["vector_field"])

    def _embed(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]] | None,
    ) -> list[list[float]]:
        """Return embedding vectors for *texts* (or use precomputed)."""
        if embeddings is not None:
            return [list(map(float, row)) for row in embeddings]
        return TextEncoderEmbeddings(self._encoder).embed_documents(list(texts))

    def _upsert_sync(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> list[str]:
        """Upsert with stable string ids (LangChain add_texts uses ObjectIds)."""
        written: list[str] = []
        for text, meta, doc_id, vector in zip(texts, metadatas, ids, vectors, strict=True):
            body: dict[str, Any] = {
                "_id": str(doc_id),
                self._text_field: text,
                self._vector_field: list(map(float, vector)),
                "metadata": dict(meta),
            }
            for key in (
                "case_id",
                "project_name",
                "actual_effort_days",
                "total_actual_days",
                "total_modules",
                "domain",
            ):
                if key in meta and key not in body:
                    body[key] = meta[key]
            self._collection.replace_one({"_id": str(doc_id)}, body, upsert=True)
            written.append(str(doc_id))
        return written

    def _query_langchain(
        self,
        text: str | None,
        embedding: Sequence[float] | None,
        k: int,
        filter: dict[str, Any] | None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        """Query via official LangChain ``cosmosSearch`` path."""
        if embedding is not None:
            docs = self._lc._similarity_search_with_score(
                embeddings=list(map(float, embedding)),
                k=int(k),
                pre_filter=filter,
            )
        else:
            docs = self._lc.similarity_search_with_score(
                query=text or "",
                k=int(k),
                pre_filter=filter,
            )
        results: list[dict[str, Any]] = []
        for doc, score in docs:
            meta = dict(doc.metadata or {})
            results.append(
                {
                    "id": str(meta.get("_id") or meta.get("case_id") or ""),
                    "score": float(score),
                    "text": doc.page_content or "",
                    "metadata": meta,
                }
            )
        return results

    def _query_cosine_fallback(
        self,
        vector: Sequence[float],
        k: int,
        filter: dict[str, Any] | None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        """Local Mongo stand-in: brute-force cosine over stored embeddings."""
        from ai_core.similarity import brute_force_cosine_search

        query: dict[str, Any] = dict(filter or {})
        docs = list(self._collection.find(query))
        if not docs:
            return []
        matrix = []
        keep: list[dict[str, Any]] = []
        for doc in docs:
            emb = doc.get(self._vector_field)
            if not emb:
                continue
            matrix.append(list(map(float, emb)))
            keep.append(doc)
        if not matrix:
            return []
        matches = brute_force_cosine_search(
            np.asarray(vector, dtype=float),
            np.asarray(matrix, dtype=float),
            top_k=int(k),
        )
        results: list[dict[str, Any]] = []
        for idx, score in matches:
            doc = keep[idx]
            meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            results.append(
                {
                    "id": str(doc.get("_id", "")),
                    "score": float(score),
                    "text": doc.get(self._text_field) or "",
                    "metadata": {**meta, "_id": doc.get("_id")},
                }
            )
        return results

    def _query_sync(
        self,
        text: str | None,
        embedding: Sequence[float] | None,
        k: int,
        filter: dict[str, Any] | None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        """Blocking vector query (LangChain cosmosSearch or cosine fallback)."""
        if self._settings.get("use_cosmos_search", True):
            try:
                return self._query_langchain(text, embedding, k, filter)
            except Exception:  # noqa: BLE001 - degrade to local cosine
                self._settings["use_cosmos_search"] = False
        vector = (
            list(map(float, embedding))
            if embedding is not None
            else TextEncoderEmbeddings(self._encoder).embed_query(text or "")
        )
        return self._query_cosine_fallback(vector, k, filter)

    def _delete_sync(self, ids: Sequence[str]) -> None:
        """Delete by string ``_id`` (case ids), not ObjectId."""
        if not ids:
            return
        self._collection.delete_many({"_id": {"$in": [str(i) for i in ids]}})

    async def upsert(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Insert or update documents with embeddings."""
        texts_list = list(texts)
        metadatas_list = list(metadatas or [{} for _ in texts_list])
        ids_list = list(ids or [f"row-{i}" for i in range(len(texts_list))])
        if len(metadatas_list) != len(texts_list) or len(ids_list) != len(texts_list):
            raise ValueError("texts, metadatas, and ids must be the same length")
        vectors = self._embed(texts_list, embeddings)
        return await asyncio.to_thread(
            self._upsert_sync, texts_list, metadatas_list, ids_list, vectors
        )

    async def query(
        self,
        text: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        """Return the top-*k* nearest documents."""
        return await asyncio.to_thread(self._query_sync, text, embedding, k, filter)

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete documents by id."""
        await asyncio.to_thread(self._delete_sync, list(ids))

    def count_sync(self) -> int:
        """Return document count."""
        return int(self._collection.count_documents({}))
