"""Embedding provider factory."""
from __future__ import annotations

from typing import Any

from loguru import logger

_embeddings_instance: Any = None


def get_embeddings(provider: str = "dummy", **kwargs: Any) -> Any:
    """Get or create a singleton embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    try:
        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            _embeddings_instance = OllamaEmbeddings(
                model=kwargs.get("model", "nomic-embed-text"),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
            )
        else:
            from langchain_core.embeddings import DeterministicFakeEmbedding
            _embeddings_instance = DeterministicFakeEmbedding(
                size=kwargs.get("dimension", 768),
            )
    except Exception as exc:
        logger.warning(f"Failed to build embeddings: {exc}. Using dummy.")
        from langchain_core.embeddings import DeterministicFakeEmbedding
        _embeddings_instance = DeterministicFakeEmbedding(size=768)

    return _embeddings_instance


def reset_embeddings() -> None:
    """Reset embeddings singleton."""
    global _embeddings_instance
    _embeddings_instance = None
