"""Text embedding via OpenAI-compatible APIs, with a hash fallback.

Prefers domain ``AI_EMBED_*`` settings (MLX / SecureGPT). Optionally overlays
Agentomatic stack embedding config when it is a real provider (never ``dummy``).
Falls back to a deterministic hash embedder so similarity search and tests
remain functional offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ai_core.settings import Settings, get_settings

_FAKE_PROVIDERS = frozenset({"dummy", "fake", "hash", "test"})


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine similarity reduces to a dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def hash_embed(texts: Sequence[str], dimension: int = 768) -> np.ndarray:
    """Deterministic bag-of-hashed-tokens embedding (offline fallback)."""
    out = np.zeros((len(texts), dimension), dtype=float)
    for i, text in enumerate(texts):
        for token in (text or "").lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            out[i, idx] += sign
    return _l2_normalize(out)


class TextEncoder:
    """Encode text into L2-normalised embedding vectors."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        dimension: int,
        provider: str = "openai",
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
    def from_settings(cls, settings: Settings | None = None) -> TextEncoder:
        """Build an encoder from domain settings, overlaying a real stack config."""
        cfg = settings or get_settings()
        provider = "openai"
        model = cfg.embed_model
        dimension = cfg.embed_dimension
        base_url = cfg.embed_base_url
        api_key = cfg.embed_api_key
        try:
            from agentomatic.config.settings import get_settings as am_settings

            emb = am_settings().embedding
            stack_provider = str(getattr(emb, "provider", "") or "").strip().lower()
            if stack_provider and stack_provider not in _FAKE_PROVIDERS:
                provider = stack_provider
                model = getattr(emb, "model", None) or model
                dimension = int(getattr(emb, "dimension", None) or dimension)
                stack_base = getattr(emb, "base_url", None)
                stack_key = getattr(emb, "api_key", None)
                if stack_base:
                    base_url = str(stack_base)
                if stack_key:
                    api_key = str(stack_key)
        except Exception:  # noqa: BLE001 - stack optional in unit tests
            pass
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            dimension=dimension,
            provider=provider,
            enabled=cfg.embed_enabled,
        )

    def _get_client(self) -> Any:
        """Lazily construct an embeddings client via Agentomatic providers."""
        if not self.enabled:
            self._client_failed = True
            return None
        if self._client is not None or self._client_failed:
            return self._client
        try:
            from agentomatic.providers.embeddings import get_embeddings

            kwargs: dict[str, Any] = {"model": self.model}
            if self.provider in {"openai", "azure_openai"}:
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

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode *texts*, falling back to hashing on any client failure."""
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

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string into a 1-D vector."""
        return self.encode([text])[0]
