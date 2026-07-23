"""Text embedding via OpenAI-compatible APIs, with a hash fallback.

Uses platform embedding settings when the provider is a real backend (never
``dummy`` / ``fake`` / ``hash`` / ``test``). Falls back to a deterministic
hash embedder so similarity search and tests remain functional offline.

Requires ``numpy`` (``pip install numpy`` or ``agentomatic[vector]``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

_FAKE_PROVIDERS = frozenset({"dummy", "fake", "hash", "test", ""})


def _require_numpy() -> Any:
    """Import numpy or raise a clear install hint."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "numpy is required for TextEncoder / local_npz. "
            "Install with: pip install numpy  (or agentomatic[vector])"
        ) from exc
    return np


def _l2_normalize(matrix: Any) -> Any:
    """L2-normalise rows so cosine similarity reduces to a dot product."""
    np = _require_numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def hash_embed(texts: Sequence[str], dimension: int = 768) -> Any:
    """Deterministic bag-of-hashed-tokens embedding (offline fallback)."""
    np = _require_numpy()
    out = np.zeros((len(texts), dimension), dtype=float)
    for i, text in enumerate(texts):
        for token in (text or "").lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            out[i, idx] += sign
    return _l2_normalize(out)


def brute_force_cosine_search(
    query: Any,
    corpus: Any,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return the top-*k* ``(index, score)`` matches by cosine similarity.

    Both *query* and *corpus* are assumed L2-normalised.
    """
    np = _require_numpy()
    if corpus is None or len(corpus) == 0:
        return []
    scores = corpus @ query
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    ranked = sorted(top_idx, key=lambda i: float(scores[i]), reverse=True)
    return [(int(i), float(scores[i])) for i in ranked]


class TextEncoder:
    """Encode text into L2-normalised embedding vectors."""

    def __init__(
        self,
        *,
        base_url: str = "",
        model: str = "nomic-embed-text",
        api_key: str = "",
        dimension: int = 768,
        provider: str = "hash",
        enabled: bool = True,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.dimension = dimension
        self.provider = provider
        self.enabled = enabled
        self._client: Any = None
        self._client_failed = False

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> TextEncoder:
        """Build an encoder from platform embedding settings.

        When the configured provider is a fake/test provider, the encoder
        uses the hash fallback immediately.
        """
        if settings is None:
            from agentomatic.config.settings import get_settings

            settings = get_settings()
        emb = getattr(settings, "embedding", settings)
        provider = str(getattr(emb, "provider", "hash") or "hash").strip().lower()
        model = str(getattr(emb, "model", None) or "nomic-embed-text")
        dimension = int(getattr(emb, "dimension", None) or 768)
        base_url = str(getattr(emb, "base_url", None) or "")
        api_key = str(getattr(emb, "api_key", None) or "")
        enabled = bool(getattr(emb, "enabled", True))
        if provider in _FAKE_PROVIDERS:
            provider = "hash"
            enabled = False
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            dimension=dimension,
            provider=provider,
            enabled=enabled,
        )

    def _get_client(self) -> Any:
        """Lazily construct an embeddings client via Agentomatic providers."""
        if not self.enabled or self.provider in _FAKE_PROVIDERS or self.provider == "hash":
            self._client_failed = True
            return None
        if self._client is not None or self._client_failed:
            return self._client
        try:
            from agentomatic.providers.embeddings import get_embeddings

            kwargs: dict[str, Any] = {"model": self.model}
            if self.provider in {"openai", "azure_openai", "azure"}:
                kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url
            if self.provider == "hash":
                kwargs["dimension"] = self.dimension
            self._client = get_embeddings(self.provider, **kwargs)
        except Exception:  # noqa: BLE001 - any import/config failure -> fallback
            self._client_failed = True
            self._client = None
        return self._client

    def encode(self, texts: Sequence[str]) -> Any:
        """Encode *texts*, falling back to hashing on any client failure."""
        np = _require_numpy()
        if not texts:
            return np.zeros((0, self.dimension), dtype=float)
        client = self._get_client()
        if client is not None:
            try:
                vectors = client.embed_documents(list(texts))
                matrix = np.array(vectors, dtype=float)
                if matrix.ndim == 2 and matrix.shape[0] == len(texts):
                    return _l2_normalize(matrix)
            except Exception:  # noqa: BLE001 - network/model failure -> fallback
                self._client_failed = True
        return hash_embed(texts, self.dimension)

    def encode_one(self, text: str) -> Any:
        """Encode a single string into a 1-D vector."""
        return self.encode([text])[0]
