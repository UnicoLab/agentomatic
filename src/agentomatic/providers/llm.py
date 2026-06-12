"""Multi-provider LLM factory."""

from __future__ import annotations

from typing import Any

from loguru import logger

_llm_instance: Any = None


def get_llm(provider: str = "ollama", **kwargs: Any) -> Any:
    """Get or create a singleton LLM instance.

    Supports: ollama, azure, openai, vertex, dummy.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    try:
        _llm_instance = _build_llm(provider, **kwargs)
    except Exception as exc:
        logger.warning(f"Failed to build {provider} LLM: {exc}. Falling back to dummy.")
        _llm_instance = _build_dummy_llm()

    return _llm_instance


def _build_llm(provider: str, **kwargs: Any) -> Any:
    """Build an LLM instance for the given provider."""
    provider = provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=kwargs.get("model", "mistral:7b"),
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            temperature=kwargs.get("temperature", 0.1),
        )

    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            api_key=kwargs.get("api_key", ""),
            azure_endpoint=kwargs.get("api_base", ""),
            api_version=kwargs.get("api_version", "2024-02-15-preview"),
            azure_deployment=kwargs.get("deployment_name", ""),
            temperature=kwargs.get("temperature", 0.1),
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "gpt-4"),
            temperature=kwargs.get("temperature", 0.1),
        )

    elif provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model_name=kwargs.get("model", "gemini-2.0-flash"),
            project=kwargs.get("project", ""),
            location=kwargs.get("location", "us-central1"),
            temperature=kwargs.get("temperature", 0.1),
        )

    elif provider == "dummy":
        return _build_dummy_llm()

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _build_dummy_llm() -> Any:
    """Build a dummy LLM for testing."""
    from langchain_core.language_models import FakeListChatModel

    return FakeListChatModel(
        responses=["This is a dummy response from agentomatic."],
    )


def reset_llm() -> None:
    """Reset the LLM singleton."""
    global _llm_instance
    _llm_instance = None


async def invoke_with_retry(
    llm: Any,
    messages: list,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Any:
    """Invoke LLM with retry logic."""
    import asyncio

    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.warning(f"LLM attempt {attempt + 1} failed: {exc}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

    raise last_exc
