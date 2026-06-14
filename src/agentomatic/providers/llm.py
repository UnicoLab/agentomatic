"""Multi-provider LLM factory."""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

_llm_instance: Any = None
_failover_count: int = 0
_llm_lock = threading.Lock()


def get_failover_count() -> int:
    """Return the number of LLM failover events recorded."""
    return _failover_count


def record_failover(primary_provider: str, fallback_provider: str, error: str) -> None:
    """Record a failover event for telemetry."""
    global _failover_count
    _failover_count += 1
    logger.warning(
        f"🔄 LLM failover #{_failover_count}: {primary_provider} -> {fallback_provider} | Error: {error}"
    )


def get_llm(provider: str = "ollama", fallbacks: list[str] | None = None, **kwargs: Any) -> Any:
    """Get or create a singleton LLM instance.

    Supports: ollama, azure, openai, vertex, dummy.
    """
    global _llm_instance
    if _llm_instance is not None:
        logger.debug("Returning cached LLM instance (call reset_llm() to reconfigure)")
        return _llm_instance

    with _llm_lock:
        # Double-checked locking
        if _llm_instance is not None:
            return _llm_instance

        try:
            primary = _build_llm(provider, **kwargs)
            if fallbacks:
                fallback_models = []
                for fb in fallbacks:
                    try:
                        fallback_models.append(_build_llm(fb, **kwargs))
                    except Exception as exc:
                        logger.warning(f"Failed to build fallback LLM {fb}: {exc}")
                if fallback_models:
                    primary = primary.with_fallbacks(
                        fallback_models,
                        exceptions_to_handle=(Exception,),
                    )
                    logger.info(f"🛡️ LLM failover chain configured: {provider} -> {fallbacks}")
            _llm_instance = primary
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
    global _llm_instance, _failover_count
    with _llm_lock:
        _llm_instance = None
        _failover_count = 0


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

    raise last_exc  # type: ignore[misc]


class StructuredOutputFallbackWrapper:
    """Fallback wrapper for models that don't support .with_structured_output natively (e.g. dummy/fake models in tests)."""

    def __init__(self, llm: Any, response_model: type) -> None:
        self.llm = llm
        self.response_model = response_model

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        res = self.llm.invoke(*args, **kwargs)
        return self._parse_to_model(res)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        res = await self.llm.ainvoke(*args, **kwargs)
        return self._parse_to_model(res)

    def _parse_to_model(self, res: Any) -> Any:
        content = res.content if hasattr(res, "content") else str(res)
        import json

        from pydantic import BaseModel
        from pydantic_core import PydanticUndefined

        if issubclass(self.response_model, BaseModel):
            try:
                data = json.loads(content)
                return self.response_model.model_validate(data)
            except Exception:
                fields = {}
                for name, field in self.response_model.model_fields.items():
                    if field.default is not PydanticUndefined:
                        fields[name] = field.default
                    elif field.default_factory is not None:
                        fields[name] = field.default_factory()  # type: ignore[call-arg]
                    else:
                        t = field.annotation
                        if t is str:
                            fields[name] = "dummy_str"
                        elif t is int:
                            fields[name] = 0
                        elif t is float:
                            fields[name] = 0.0
                        elif t is bool:
                            fields[name] = False
                        elif t is list or getattr(t, "__origin__", None) is list:
                            fields[name] = []
                        elif t is dict or getattr(t, "__origin__", None) is dict:
                            fields[name] = {}
                        else:
                            fields[name] = None
                return self.response_model(**fields)
        return content


def get_structured_llm(
    response_model: type,
    provider: str = "ollama",
    **kwargs: Any,
) -> Any:
    """Get an LLM instance bound to return a structured output matching response_model."""
    llm = _build_llm(provider, **kwargs)
    if hasattr(llm, "with_structured_output"):
        try:
            return llm.with_structured_output(response_model)
        except NotImplementedError:
            return StructuredOutputFallbackWrapper(llm, response_model)
    return StructuredOutputFallbackWrapper(llm, response_model)
