"""On-disk vector store (.npz) and Agentomatic provider registration.

Provides:
- :class:`EmbeddingStore` — compact on-disk store used by artifact bundles.
- :class:`LocalNpzVectorStore` — Agentomatic ``VectorStore`` over .npz.
- :func:`register_vector_backends` — registers ``local_npz`` and
  ``azure_cosmos`` (Cosmos Mongo vCore via ``langchain-azure-ai``; see
  :mod:`ai_core.cosmos_vector`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ai_core.embeddings import TextEncoder

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ai_core.settings import Settings


class EmbeddingStore:
    """A directory-backed embedding store: ``embeddings.npz`` + metadata jsonl."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._emb_path = self.directory / "embeddings.npz"
        self._meta_path = self.directory / "metadata.jsonl"

    def exists(self) -> bool:
        """Return whether both the embeddings and metadata files exist."""
        return self._emb_path.exists() and self._meta_path.exists()

    def count(self) -> int:
        """Return the number of stored rows (0 when the store is absent)."""
        if not self.exists():
            return 0
        return len(self.load()[1])

    def save(self, embeddings: np.ndarray, metadatas: Sequence[dict[str, Any]]) -> None:
        """Persist an embedding matrix and aligned metadata rows."""
        if len(embeddings) != len(metadatas):
            raise ValueError("embeddings and metadatas must be the same length")
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._emb_path, embeddings=np.asarray(embeddings, dtype=float))
        with self._meta_path.open("w", encoding="utf-8") as fh:
            for meta in metadatas:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def load(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Load the embedding matrix and metadata rows."""
        if not self.exists():
            return np.zeros((0, 0), dtype=float), []
        with np.load(self._emb_path) as data:
            embeddings = data["embeddings"]
        metadatas: list[dict[str, Any]] = []
        with self._meta_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    metadatas.append(json.loads(line))
        return embeddings, metadatas


class LocalNpzVectorStore:
    """Adapter implementing Agentomatic's ``VectorStore`` protocol over .npz."""

    def __init__(self, directory: str | Path, encoder: TextEncoder | None = None) -> None:
        self.store = EmbeddingStore(directory)
        self.encoder = encoder or TextEncoder.from_settings()

    async def upsert(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Embed and append *texts* to the store, returning their ids."""
        metadatas = list(metadatas or [{} for _ in texts])
        ids = list(ids or [f"row-{i}" for i in range(len(texts))])
        vectors = (
            np.asarray(embeddings, dtype=float)
            if embeddings is not None
            else self.encoder.encode(list(texts))
        )
        existing_emb, existing_meta = self.store.load()
        merged_meta = existing_meta + [
            {**m, "_id": i, "_text": t} for m, i, t in zip(metadatas, ids, texts, strict=False)
        ]
        merged_emb = np.vstack([existing_emb, vectors]) if existing_emb.size else vectors
        self.store.save(merged_emb, merged_meta)
        return list(ids)

    async def query(
        self,
        text: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002 - protocol name
    ) -> list[dict[str, Any]]:
        """Return the top-*k* nearest rows by cosine similarity."""
        from ai_core.similarity import brute_force_cosine_search

        emb, metas = self.store.load()
        if emb.size == 0:
            return []
        query_vec = (
            np.asarray(embedding, dtype=float)
            if embedding is not None
            else self.encoder.encode_one(text or "")
        )
        results = brute_force_cosine_search(query_vec, emb, top_k=k)
        hits: list[dict[str, Any]] = []
        for i, score in results:
            meta = metas[i]
            hits.append(
                {
                    "id": str(meta.get("_id") or meta.get("case_id") or i),
                    "score": score,
                    "text": meta.get("_text") or "",
                    "metadata": meta,
                }
            )
        return hits

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete rows by id (rewrites the store)."""
        emb, metas = self.store.load()
        keep = [i for i, m in enumerate(metas) if m.get("_id") not in set(ids)]
        if len(keep) == len(metas):
            return
        self.store.save(emb[keep] if emb.size else emb, [metas[i] for i in keep])


def _local_npz_provider(config: Any) -> LocalNpzVectorStore:
    """Provider builder for the local .npz vector store."""
    directory = getattr(config, "url", None) or ".local/artifacts/current/embeddings"
    return LocalNpzVectorStore(directory)


def _azure_cosmos_provider(config: Any) -> Any:
    """Provider builder: return a CosmosClient (database + container ensured)."""
    from ai_core.cosmos_vector import build_cosmos_client

    return build_cosmos_client(config)


def _azure_cosmos_adapter(config: Any, client: Any) -> Any:
    """Adapter factory: wrap CosmosClient as :class:`AzureCosmosVectorStore`."""
    from ai_core.cosmos_vector import AzureCosmosVectorStore

    return AzureCosmosVectorStore(config, client)


def register_vector_backends(settings: Settings | None = None) -> None:
    """Register the local and Cosmos vector providers with Agentomatic.

    Safe to call at import time in ``main.py``; unknown-provider errors point at
    ``register_vector_provider`` per framework convention.
    """
    del settings  # reserved for future settings-driven registration
    try:
        from agentomatic.connections import (
            register_vector_provider,
            register_vector_store_adapter,
        )
    except Exception:  # noqa: BLE001 - framework optional at unit-test time
        return
    register_vector_provider("local_npz", _local_npz_provider)
    register_vector_store_adapter("local_npz", lambda cfg, client: client)
    # Common typo / legacy alias seen in env stacks (tensorflow vs npz).
    register_vector_provider("local_tensorflow", _local_npz_provider)
    register_vector_store_adapter("local_tensorflow", lambda cfg, client: client)
    register_vector_provider("azure_cosmos", _azure_cosmos_provider)
    register_vector_store_adapter("azure_cosmos", _azure_cosmos_adapter)
