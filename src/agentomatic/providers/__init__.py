"""LLM and embedding provider abstractions."""

from __future__ import annotations

from .embeddings import HashEmbedder, get_embeddings, reset_embeddings
from .llm import (
    get_failover_count,
    get_llm,
    get_llm_for_agent,
    get_named_llm,
    get_structured_llm,
    invoke_with_retry,
    record_failover,
    reset_llm,
    set_llm,
)

__all__ = [
    "HashEmbedder",
    "get_embeddings",
    "get_failover_count",
    "get_llm",
    "get_llm_for_agent",
    "get_named_llm",
    "get_structured_llm",
    "invoke_with_retry",
    "record_failover",
    "reset_embeddings",
    "reset_llm",
    "set_llm",
]
