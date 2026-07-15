"""LLM and embedding provider abstractions."""

from __future__ import annotations

from .embeddings import (
    HashEmbedder,
    get_embeddings,
    register_embedding_provider,
    registered_embedding_providers,
    reset_embeddings,
)
from .llm import (
    apply_stack_defaults,
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
    "apply_stack_defaults",
    "get_embeddings",
    "get_failover_count",
    "get_llm",
    "get_llm_for_agent",
    "get_named_llm",
    "get_structured_llm",
    "invoke_with_retry",
    "record_failover",
    "register_embedding_provider",
    "registered_embedding_providers",
    "reset_embeddings",
    "reset_llm",
    "set_llm",
]
