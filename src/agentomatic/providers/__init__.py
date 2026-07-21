"""LLM and embedding provider abstractions."""

from __future__ import annotations

from .embeddings import (
    HashEmbedder,
    get_embeddings,
    register_embedding_provider,
    registered_embedding_providers,
    reset_embeddings,
)
from .fallback import (
    DEFAULT_FALLBACK_ON,
    EmptyLLMResponseError,
    FallbackLLM,
    is_empty_llm_response,
    model_label,
    normalize_fallback_on,
    should_fallback,
)
from .llm import (
    StructuredOutputFallbackWrapper,
    apply_stack_defaults,
    astream_with_thinking,
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
from .message_utils import (
    SplitMessage,
    attach_thinking_metadata,
    llm_result_metadata,
    message_text,
    message_thinking,
    split_llm_message,
    split_thinking_text,
    strip_thinking_for_json,
)

__all__ = [
    "DEFAULT_FALLBACK_ON",
    "EmptyLLMResponseError",
    "FallbackLLM",
    "HashEmbedder",
    "SplitMessage",
    "StructuredOutputFallbackWrapper",
    "apply_stack_defaults",
    "astream_with_thinking",
    "attach_thinking_metadata",
    "get_embeddings",
    "get_failover_count",
    "get_llm",
    "get_llm_for_agent",
    "get_named_llm",
    "get_structured_llm",
    "invoke_with_retry",
    "is_empty_llm_response",
    "llm_result_metadata",
    "message_text",
    "message_thinking",
    "model_label",
    "normalize_fallback_on",
    "record_failover",
    "register_embedding_provider",
    "registered_embedding_providers",
    "reset_embeddings",
    "reset_llm",
    "set_llm",
    "should_fallback",
    "split_llm_message",
    "split_thinking_text",
    "strip_thinking_for_json",
]
