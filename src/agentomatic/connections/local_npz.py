"""On-disk ``.npz`` vector store and ``local_npz`` provider registration.

Provides a zero-infra local RAG backend suitable for demos, tests, and
artifact bundles. Requires ``numpy`` (``pip install numpy`` or
``agentomatic[vector]``).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentomatic.connections.encoder import TextEncoder, brute_force_cosine_search


def _require_numpy() -> Any:
    """Import numpy or raise a clear install hint."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "numpy is required for local_npz. "
            "Install with: pip install numpy  (or agentomatic[vector])"
        ) from exc
    return np


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

    def save(self, embeddings: Any, metadatas: Sequence[dict[str, Any]]) -> None:
        """Persist an embedding matrix and aligned metadata rows atomically."""
        np = _require_numpy()
        if len(embeddings) != len(metadatas):
            raise ValueError("embeddings and metadatas must be the same length")
        self.directory.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(embeddings, dtype=float)

        # Suffix must end with ``.npz`` so numpy does not append another one.
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, suffix=".tmp.npz")
        os.close(fd)
        tmp_npz_path = Path(tmp_name)
        try:
            np.savez_compressed(tmp_npz_path, embeddings=matrix)
            os.replace(tmp_npz_path, self._emb_path)
        except Exception:
            tmp_npz_path.unlink(missing_ok=True)
            raise

        fd, tmp_meta = tempfile.mkstemp(dir=self.directory, suffix=".jsonl.tmp")
        tmp_meta_path = Path(tmp_meta)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for meta in metadatas:
                    fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_meta_path, self._meta_path)
        except Exception:
            tmp_meta_path.unlink(missing_ok=True)
            raise
        else:
            tmp_meta_path.unlink(missing_ok=True)

    def load(self) -> tuple[Any, list[dict[str, Any]]]:
        """Load the embedding matrix and metadata rows."""
        np = _require_numpy()
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
        """Embed and insert-or-update *texts* by id, returning their ids."""
        np = _require_numpy()
        metadatas = list(metadatas or [{} for _ in texts])
        if len(metadatas) != len(texts):
            raise ValueError("metadatas must be the same length as texts")
        ids = list(ids) if ids is not None else [f"row-{uuid.uuid4().hex[:12]}" for _ in texts]
        if len(ids) != len(texts):
            raise ValueError("ids must be the same length as texts")
        vectors = (
            np.asarray(embeddings, dtype=float)
            if embeddings is not None
            else self.encoder.encode(list(texts))
        )
        if len(vectors) != len(texts):
            raise ValueError("embeddings must be the same length as texts")

        existing_emb, existing_meta = self.store.load()
        id_to_index = {
            str(m.get("_id")): i for i, m in enumerate(existing_meta) if m.get("_id") is not None
        }

        # Start from a mutable copy of the existing store.
        if existing_emb.size:
            emb_list: list[Any] = [existing_emb[i] for i in range(len(existing_meta))]
        else:
            emb_list = []
        meta_list: list[dict[str, Any]] = list(existing_meta)

        for text, meta, doc_id, vector in zip(texts, metadatas, ids, vectors, strict=True):
            row = {**meta, "_id": doc_id, "_text": text}
            if doc_id in id_to_index:
                idx = id_to_index[doc_id]
                emb_list[idx] = vector
                meta_list[idx] = row
            else:
                id_to_index[doc_id] = len(meta_list)
                emb_list.append(vector)
                meta_list.append(row)

        merged_emb = np.vstack(emb_list) if emb_list else np.zeros((0, 0), dtype=float)
        self.store.save(merged_emb, meta_list)
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
        del filter  # optional metadata filter not yet implemented for .npz
        np = _require_numpy()
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
                    "id": str(meta.get("_id") or i),
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
    directory = getattr(config, "url", None) or ".local/artifacts/current"
    # Allow options.directory override
    options = getattr(config, "options", None) or {}
    if isinstance(options, dict) and options.get("directory"):
        directory = options["directory"]
    return LocalNpzVectorStore(directory)


def register_local_npz_backends() -> None:
    """Register ``local_npz`` (and ``local_tensorflow`` alias) with Agentomatic.

    Safe to call multiple times; called automatically when
    :mod:`agentomatic.connections.vector` is imported.
    """
    from agentomatic.connections.vector import (
        register_vector_provider,
        register_vector_store_adapter,
    )

    register_vector_provider("local_npz", _local_npz_provider)
    register_vector_store_adapter("local_npz", lambda cfg, client: client)
    # Common typo / legacy alias (tensorflow vs npz).
    register_vector_provider("local_tensorflow", _local_npz_provider)
    register_vector_store_adapter("local_tensorflow", lambda cfg, client: client)
