"""Multi-provider LLM factory."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.stacks.manager import StackManager

_llm_instance: Any = None
_named_instances: dict[str, Any] = {}
_failover_count: int = 0
_llm_lock = threading.Lock()


def get_failover_count() -> int:
    """Return the number of LLM failover events recorded."""
    with _llm_lock:
        return _failover_count


def record_failover(
    primary_provider: str,
    fallback_provider: str,
    error: str,
) -> None:
    """Record a failover event for telemetry."""
    global _failover_count
    with _llm_lock:
        _failover_count += 1
        count = _failover_count
    logger.warning(
        f"🔄 LLM failover #{count}: {primary_provider} -> {fallback_provider} | Error: {error}"
    )


def get_llm(
    provider: str = "ollama",
    fallbacks: list[str] | None = None,
    *,
    instance: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Get or create a singleton LLM instance.

    Supports: ollama, azure, openai, vertex, dummy.

    If *instance* is provided, it is stored as the global singleton
    directly — bypassing the factory.  This lets users inject a custom
    LLM (e.g. a LangChain model, a callable, or any object that
    implements ``ainvoke`` / ``invoke``).

    Example::

        from langchain_openai import ChatOpenAI
        from agentomatic.providers import get_llm

        # Use a custom pre-built model
        get_llm(instance=ChatOpenAI(model="gpt-4o"))

    Args:
        provider: Provider identifier when building from factory.
        fallbacks: Optional fallback provider identifiers.
        instance: Pre-built LLM instance to use directly.
        **kwargs: Provider-specific parameters for the factory.

    Returns:
        An LLM instance.
    """
    global _llm_instance

    # Inject a pre-built instance directly
    if instance is not None:
        with _llm_lock:
            _llm_instance = instance
            logger.info(f"Custom LLM instance set: {type(instance).__name__}")
            return _llm_instance

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


def set_llm(instance: Any) -> None:
    """Set a custom LLM instance as the global singleton.

    This is the recommended way to inject a custom model that
    doesn't fit the built-in provider system.  Any object that
    implements ``ainvoke(messages)`` or ``invoke(messages)`` will
    work, as will plain callables.

    Example::

        from agentomatic.providers import set_llm

        # Async callable
        async def my_llm(messages):
            return await my_api.chat(messages)

        set_llm(my_llm)

        # Or a LangChain model
        from langchain_openai import ChatOpenAI
        set_llm(ChatOpenAI(model="gpt-4o"))

    Args:
        instance: Pre-built LLM object to use as the global singleton.
    """
    global _llm_instance
    with _llm_lock:
        _llm_instance = instance
        logger.info(f"Global LLM set to: {type(instance).__name__}")


def _build_llm(provider: str, **kwargs: Any) -> Any:
    """Build an LLM instance for the given provider.

    Recognised providers:

    * ``ollama`` — local Ollama server (``base_url`` respected).
    * ``openai`` — OpenAI Chat models; ``base_url`` (also accepted as
      ``api_base``) forwards to ``langchain_openai.ChatOpenAI``.
    * ``openai_compatible`` — any OpenAI-API-compatible endpoint (Groq,
      Together, LM Studio, vLLM, LiteLLM, …) — same as ``openai`` but the
      ``base_url`` is mandatory.
    * ``azure`` — Azure OpenAI; accepts ``api_base`` / ``base_url`` as the
      endpoint, plus ``api_version`` and ``deployment_name``.
    * ``vertex`` — Google Vertex AI.
    * ``dummy`` — deterministic fake for tests.
    """
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

        endpoint = (
            kwargs.get("azure_endpoint") or kwargs.get("api_base") or kwargs.get("base_url") or ""
        )
        deployment = (
            kwargs.get("azure_deployment")
            or kwargs.get("deployment_name")
            or kwargs.get("model")
            or ""
        )
        return AzureChatOpenAI(
            api_key=kwargs.get("api_key", ""),
            azure_endpoint=endpoint,
            api_version=kwargs.get("api_version", "2024-02-15-preview"),
            azure_deployment=deployment,
            temperature=kwargs.get("temperature", 0.1),
        )

    elif provider in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI

        base_url = kwargs.get("base_url") or kwargs.get("api_base") or None
        if provider == "openai_compatible" and not base_url:
            raise ValueError(
                "openai_compatible LLMs require a base_url (or api_base). "
                "Pass e.g. base_url='https://api.groq.com/openai/v1'."
            )
        ctor_kwargs: dict[str, Any] = {
            "api_key": kwargs.get("api_key", ""),
            "model": kwargs.get("model", "gpt-4"),
            "temperature": kwargs.get("temperature", 0.1),
        }
        if base_url:
            ctor_kwargs["base_url"] = base_url
        return ChatOpenAI(**ctor_kwargs)

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
    """Reset the LLM singleton and all named instances."""
    global _llm_instance, _failover_count, _named_instances
    with _llm_lock:
        _llm_instance = None
        _named_instances.clear()
        _failover_count = 0


def get_named_llm(
    name: str,
    provider: str = "ollama",
    *,
    instance: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Get or create a named LLM instance.

    Unlike :func:`get_llm`, this function maintains a registry of named
    instances so multiple LLM configurations can coexist (e.g. ``"default"``,
    ``"fast"``, ``"judge"``).

    If *instance* is provided, it is stored directly under *name*
    without going through the factory.

    Args:
        name: Unique name for this LLM instance.
        provider: Provider identifier (``"ollama"``, ``"openai"``, etc.).
        instance: Pre-built LLM instance to store directly.
        **kwargs: Provider-specific parameters.

    Returns:
        An LLM instance.
    """
    if instance is not None:
        with _llm_lock:
            _named_instances[name] = instance
            logger.debug(f"Custom LLM instance registered as '{name}': {type(instance).__name__}")
            return instance

    if name in _named_instances:
        return _named_instances[name]

    with _llm_lock:
        if name in _named_instances:
            return _named_instances[name]
        try:
            built = _build_llm(provider, **kwargs)
            _named_instances[name] = built
            logger.debug(f"Created named LLM instance '{name}' ({provider})")
        except Exception as exc:
            logger.warning(f"Failed to build LLM '{name}' ({provider}): {exc}. Using dummy.")
            built = _build_dummy_llm()
            _named_instances[name] = built
        return built


def apply_stack_defaults(stack_manager: StackManager | None) -> Any:
    """Apply the active stack's default LLM profile to ``get_llm()``.

    Reads the ``"default"`` profile from the active stack and reuses
    :func:`get_named_llm` to build (and cache) it under the same name.
    The resulting instance is then promoted to the global singleton via
    :func:`set_llm` so :func:`get_llm` becomes stack-aware — every
    caller that has not explicitly injected an LLM now gets one built
    from the current stack.

    Args:
        stack_manager: The platform's active :class:`StackManager` (or
            ``None`` when no stack was loaded — the function then no-ops).

    Returns:
        The built LLM instance, or ``None`` when nothing was applied.
    """
    if stack_manager is None:
        return None
    try:
        entry = stack_manager.get_llm_config("default")
    except (ValueError, KeyError):
        logger.debug("No 'default' LLM profile in active stack; skipping apply_stack_defaults()")
        return None

    kwargs: dict[str, Any] = {
        "provider": entry.provider,
        "model": entry.model,
        "temperature": entry.temperature,
    }
    if entry.api_key:
        kwargs["api_key"] = entry.api_key
    if entry.base_url:
        kwargs["base_url"] = entry.base_url
    kwargs.update(entry.extra)

    try:
        instance = get_named_llm(name="default", **kwargs)
        set_llm(instance)
        logger.info(
            f"🧠 Global LLM initialised from active stack "
            f"(provider={entry.provider}, model={entry.model})"
        )
        return instance
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to apply stack default LLM: {exc}")
        return None


def get_llm_for_agent(
    agent_name: str,
    role: str = "default",
    stack_manager: StackManager | None = None,
) -> Any:
    """Get an LLM instance for a specific agent and role.

    Resolution order:
      1. Agent's ``llm_config`` maps *role* → stack profile name.
      2. Stack's ``llm[profile_name]`` provides provider / model / credentials.
      3. Falls back to the stack's ``"default"`` profile.
      4. Falls back to the global :func:`get_llm` singleton.

    Args:
        agent_name: Name of the requesting agent.
        role: Logical role (``"default"``, ``"fast"``, ``"judge"``, …).
        stack_manager: Optional :class:`StackManager` for stack resolution.

    Returns:
        An LLM instance.
    """
    if stack_manager is None:
        return get_llm()

    try:
        entry = stack_manager.get_llm_config(role)
    except (ValueError, KeyError):
        try:
            entry = stack_manager.get_llm_config("default")
        except (ValueError, KeyError):
            logger.debug(
                f"No stack LLM config for agent={agent_name} role={role}, "
                "falling back to global LLM"
            )
            return get_llm()

    instance_name = f"{agent_name}:{role}"
    return get_named_llm(
        name=instance_name,
        provider=entry.provider,
        model=entry.model,
        temperature=entry.temperature,
        api_key=entry.api_key,
        base_url=entry.base_url,
        **entry.extra,
    )


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
            except Exception as exc:
                logger.warning(
                    f"Structured output parse failed for "
                    f"{self.response_model.__name__}: {exc}. "
                    f"Falling back to dummy values. "
                    f"Raw content (first 200 chars): "
                    f"{content[:200]!r}"
                )
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
    *,
    instance: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Get an LLM instance bound to return structured output.

    If *instance* is provided, it is used directly instead of
    building from the factory.  This lets users inject a custom
    LLM (LangChain model, callable, etc.) and still get
    structured output parsing.

    Args:
        response_model: Pydantic model class for output parsing.
        provider: Provider identifier for the factory.
        instance: Pre-built LLM instance to use directly.
        **kwargs: Provider-specific parameters for the factory.

    Returns:
        An LLM instance bound to the response model.
    """
    llm = instance if instance is not None else _build_llm(provider, **kwargs)
    if hasattr(llm, "with_structured_output"):
        try:
            return llm.with_structured_output(response_model)
        except NotImplementedError:
            return StructuredOutputFallbackWrapper(llm, response_model)
    return StructuredOutputFallbackWrapper(llm, response_model)
