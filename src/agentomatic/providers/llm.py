"""Multi-provider LLM factory."""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.stacks.manager import StackManager

#: Where a local oMLX / llama.cpp / vLLM style server listens by default.
#: Kept in sync with ``agentomatic.optimize.llm_caller`` so the serving path
#: and the optimization path agree on one endpoint.
DEFAULT_OMLX_BASE_URL = "http://127.0.0.1:8000/v1"

_llm_instance: Any = None
_named_instances: dict[str, Any] = {}
_failover_count: int = 0
_llm_lock = threading.Lock()

#: Custom provider builders keyed by lowercase provider name. Populated via
#: :func:`register_llm_provider` so hosts (or plugins) can add hubs the core
#: factory doesn't know about (e.g. a token-gated internal gateway) without
#: forking ``_build_llm``. Checked *before* the built-in providers, so a
#: registered name may also override a built-in one.
_LLM_PROVIDERS: dict[str, Callable[..., Any]] = {}
_llm_provider_lock = threading.RLock()


def _llm_provider_key(name: str) -> str:
    """Normalize and validate a custom provider name."""
    if not isinstance(name, str):
        raise TypeError("LLM provider name must be a string")
    key = name.strip().lower()
    if not key:
        raise ValueError("LLM provider name must be a non-empty string")
    return key


def register_llm_provider(
    name: str,
    builder: Callable[..., Any],
    *,
    overwrite: bool = True,
) -> None:
    """Register (or override) a custom LLM provider builder.

    Mirrors :func:`agentomatic.providers.embeddings.register_embedding_provider`.
    Once registered, stack YAML / ``get_llm(provider=name, ...)`` calls build
    the model through *builder* instead of the built-in provider chain.

    Args:
        name: Provider identifier (case-insensitive), e.g. ``"securegpt"``.
        builder: Callable ``(**kwargs) -> llm`` returning any object exposing
            ``invoke`` / ``ainvoke`` (a LangChain chat model or compatible
            wrapper). Receives all resolved stack/factory kwargs (``model``,
            ``api_key``, ``base_url``, ``temperature``, ``extra`` merged in,
            etc.) — the builder decides which ones it needs.
        overwrite: Replace an existing custom builder under the same normalized
            name. Set to ``False`` to detect accidental duplicate registration.

    Example::

        from agentomatic.providers import register_llm_provider

        def build_securegpt(**kwargs):
            from myapp.securegpt import SecureGPTChat
            return SecureGPTChat(**kwargs)

        register_llm_provider("securegpt", build_securegpt)
        # stacks/*.yaml: llm.default.provider: securegpt
    """
    key = _llm_provider_key(name)
    if not callable(builder):
        raise TypeError(f"LLM provider builder for '{key}' must be callable")
    with _llm_provider_lock:
        if not overwrite and key in _LLM_PROVIDERS:
            raise ValueError(f"LLM provider '{key}' is already registered")
        _LLM_PROVIDERS[key] = builder


def unregister_llm_provider(name: str) -> bool:
    """Remove a custom provider builder.

    Existing cached LLM instances are unaffected; call :func:`reset_llm` when
    replacing a live provider configuration.

    Args:
        name: Provider identifier (case-insensitive).

    Returns:
        ``True`` when a builder was removed, otherwise ``False``.
    """
    key = _llm_provider_key(name)
    with _llm_provider_lock:
        return _LLM_PROVIDERS.pop(key, None) is not None


def registered_llm_providers() -> list[str]:
    """Return the names of all registered custom LLM providers."""
    with _llm_provider_lock:
        return sorted(_LLM_PROVIDERS)


_NO_CUSTOM_TEMPERATURE_MODEL_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:gpt-?5|o[134](?![0-9]))",
    re.IGNORECASE,
)


def supports_custom_temperature(model: str | None) -> bool:
    """Return ``False`` for GPT-5 and o1/o3/o4 models that reject custom temperature."""
    name = (model or "").strip()
    if not name:
        return True
    return _NO_CUSTOM_TEMPERATURE_MODEL_RE.search(name) is None


def _strip_disallowed_temperature(mapping: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip ``temperature`` from top-level and nested extra dicts."""
    cleaned: dict[str, Any] = {}
    for key, val in mapping.items():
        if key == "temperature":
            continue
        if key in ("extra", "extra_body", "model_kwargs") and isinstance(val, dict):
            nested = _strip_disallowed_temperature(val)
            if nested:
                cleaned[key] = nested
        else:
            cleaned[key] = val
    return cleaned


_MAX_ERROR_MESSAGE_LEN = 500
_MAX_FAILED_JSON_PROBES = 64
_MAX_RECURSION_JSON_PROBES = 2
_JSON_DECODER = json.JSONDecoder()

_URL_USERINFO_RE = re.compile(r"(https?://)[^:@/\s]+:[^@/\s]+@", re.IGNORECASE)
_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)\?[^\s#]*", re.IGNORECASE)
_BEARER_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{6,}")
_COOKIE_HEADER_RE = re.compile(
    r"(?i)\b((?:set-)?cookie)(?:"
    r"['\"]\s*[:=]\s*(['\"])(?:\\[\s\S]|(?!\2)[^\\])*(?:\2|$)"
    r"|\s*=\s*(['\"])(?:\\[\s\S]|(?!\3)[^\\])*\3(?![;=])"
    r"|\s*[:=]\s*[^\r\n]+"
    r")"
)
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|ocp-apim-subscription-key|x-goog-api-key)"
    r"(?:['\"])?\s*[:=]\s*(?:"
    r"(['\"])(?:\\[\s\S]|(?!\2)[^\\])*(?:\2|$)"
    r"|(?:(?:Bearer|Basic|Token|ApiKey)\s+)?[^\s,;]+"
    r")"
)
_SECRET_KV_RE = re.compile(
    r"(?i)(?:"
    r"\b("
    r"[a-z0-9_-]*(?:api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|id[_-]?token"
    r"|password|passwd|secret|private[_-]?key|credential|system[_-]?prompt|user[_-]?prompt|raw[_-]?prompt"
    r"|raw[_-]?response|raw[_-]?request)"
    r"|token|session|sid|jwt|sig|prompt"
    r")(?:['\"])?\s*[:=]"
    r"|(?:['\"](messages|document|content)['\"]\s*[:=]|\b(messages|document|content)\s*=)"
    r")\s*(?:"
    r"(['\"])(?:\\[\s\S]|(?!\4)[^\\])*(?:\4|$)"
    r"|\[[^\[\]]*(?:\]|$)"
    r"|[^\s,;'\"}\]]+"
    r")"
)
_SENSITIVE_KV_PREFIX_RE = re.compile(
    r"(?i)(?:"
    r"\b("
    r"[a-z0-9_-]*(?:api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|id[_-]?token"
    r"|password|passwd|secret|private[_-]?key|credential|system[_-]?prompt|user[_-]?prompt|raw[_-]?prompt"
    r"|raw[_-]?response|raw[_-]?request)"
    r"|token|session|sid|jwt|sig|prompt"
    r")(?:['\"])?\s*[:=]"
    r"|(?:['\"](messages|document|content)['\"]\s*[:=]|\b(messages|document|content)\s*=)"
    r")\s*(?=[{\[])"
)
_CODE_KV_RE = re.compile(r"(?i)\b(code)\s*[:=]\s*(['\"]?)[^\s,;'\"]+\2")
_SK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_GEMINI_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")

_SAFE_TOKEN_META_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "prompt_tokens",
        "completion_tokens",
        "tokens",
        "token_count",
        "max_tokens",
        "max_completion_tokens",
    }
)
_JSON_SENSITIVE_KEYS = frozenset(
    {
        "messages",
        "raw_request",
        "raw_response",
        "document",
        "content",
        "prompt",
        "system_prompt",
        "user_prompt",
        "raw_prompt",
        "authorization",
        "cookie",
        "set-cookie",
        "jwt",
        "session",
        "sid",
        "sig",
        "token",
        "secret",
        "password",
        "passwd",
        "api_key",
        "apikey",
        "credential",
        "private_key",
        "file_content",
        "user_message",
        "assistant_message",
        "system_message",
    }
)
_JSON_SENSITIVE_SUBSTRINGS = (
    "secret",
    "token",
    "api_key",
    "apikey",
    "password",
    "passwd",
    "private_key",
    "credential",
    "prompt",
    "raw_response",
    "raw_request",
    "bearer",
    "cookie",
    "file_content",
)


def _is_sensitive_json_key(key: str, value: Any = None) -> bool:
    lower = key.strip().lower()
    if not lower:
        return True
    if lower in _SAFE_TOKEN_META_KEYS:
        return isinstance(value, str)
    if lower in _JSON_SENSITIVE_KEYS:
        return True
    if any(sub in lower for sub in _JSON_SENSITIVE_SUBSTRINGS):
        return True
    if "token" in lower and not lower.endswith("_tokens"):
        return True
    return False


def _redact_json_value(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, (dict, list)):
        return value, False
    changed = False
    root_out: Any = {} if isinstance(value, dict) else []
    stack: list[tuple[Any, Any]] = [(value, root_out)]
    while stack:
        src, dst = stack.pop()
        if isinstance(src, dict):
            for k, v in src.items():
                key_str = str(k)
                if _is_sensitive_json_key(key_str, v):
                    dst[key_str] = "[REDACTED]"
                    changed = True
                elif isinstance(v, dict):
                    child_dict: dict[str, Any] = {}
                    dst[key_str] = child_dict
                    stack.append((v, child_dict))
                elif isinstance(v, list):
                    child_list: list[Any] = []
                    dst[key_str] = child_list
                    stack.append((v, child_list))
                else:
                    dst[key_str] = v
        elif isinstance(src, list):
            for item in src:
                if isinstance(item, dict):
                    child_dict = {}
                    dst.append(child_dict)
                    stack.append((item, child_dict))
                elif isinstance(item, list):
                    child_list = []
                    dst.append(child_list)
                    stack.append((item, child_list))
                else:
                    dst.append(item)
    return (root_out, True) if changed else (value, False)


def _redact_json_fragments(text: str) -> str:
    if "{" not in text and "[" not in text:
        return text
    out: list[str] = []
    pos = 0
    length = len(text)
    failed_probes = 0
    recursion_probes = 0
    while pos < length:
        next_obj = text.find("{", pos)
        next_arr = text.find("[", pos)
        candidates = [idx for idx in (next_obj, next_arr) if idx != -1]
        if not candidates:
            out.append(text[pos:])
            break
        idx = min(candidates)
        if text.startswith("[REDACTED]", idx):
            out.append(text[pos : idx + 10])
            pos = idx + 10
            continue
        if (
            failed_probes >= _MAX_FAILED_JSON_PROBES
            or recursion_probes >= _MAX_RECURSION_JSON_PROBES
        ):
            out.append(text[pos:])
            break
        try:
            parsed, end = _JSON_DECODER.raw_decode(text, idx)
            if isinstance(parsed, (dict, list)):
                sanitized, changed = _redact_json_value(parsed)
                if changed:
                    dumped = json.dumps(sanitized, ensure_ascii=False)
                    out.append(text[pos:idx])
                    out.append(dumped)
                    pos = end
                    continue
        except RecursionError:
            recursion_probes += 1
            failed_probes += 1
            skip_end = idx + 1
            ch = text[idx]
            while skip_end < length and text[skip_end] == ch:
                skip_end += 1
            out.append(text[pos:skip_end])
            pos = skip_end
            continue
        except json.JSONDecodeError:
            failed_probes += 1
            skip_end = idx + 1
            ch = text[idx]
            while skip_end < length and text[skip_end] == ch:
                skip_end += 1
            out.append(text[pos:skip_end])
            pos = skip_end
            continue
        out.append(text[pos:end])
        pos = end
    return "".join(out)


def _scan_balanced_container_end(text: str, start: int) -> int:
    """Return end index (exclusive) of a possibly nested/truncated ``[...]`` or ``{...}``."""
    length = len(text)
    stack: list[str] = []
    i = start
    in_quote: str | None = None
    escaped = False
    while i < length:
        ch = text[i]
        if in_quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_quote:
                in_quote = None
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "[":
                stack.append("]")
            elif ch == "{":
                stack.append("}")
            elif ch in ("]", "}"):
                if stack and ch == stack[-1]:
                    stack.pop()
                    if not stack:
                        return i + 1
                elif ch in stack:
                    while stack and stack[-1] != ch:
                        stack.pop()
                    if stack:
                        stack.pop()
                        if not stack:
                            return i + 1
                else:
                    return i
        i += 1
    return length


def _redact_malformed_sensitive_containers(text: str) -> str:
    """Redact nested/truncated ``[...]`` or ``{...}`` values of sensitive keys in O(N)."""
    if "{" not in text and "[" not in text:
        return text
    out: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        match = _SENSITIVE_KV_PREFIX_RE.search(text, pos)
        if match is None:
            out.append(text[pos:])
            break
        key = match.group(1) or match.group(2) or match.group(3) or "secret"
        open_idx = match.end()
        out.append(text[pos : match.start()])
        out.append(f"{key}=[REDACTED]")
        if text.startswith("[REDACTED]", open_idx):
            pos = open_idx + 10
        else:
            pos = _scan_balanced_container_end(text, open_idx)
    return "".join(out)


def _replace_secret_kv(match: re.Match[str]) -> str:
    key = match.group(1) or match.group(2) or match.group(3) or "secret"
    return f"{key}=[REDACTED]"


def _sanitize_error_for_log(raw: Any, *, max_len: int = _MAX_ERROR_MESSAGE_LEN) -> str:
    """Redact credentials, tokens, URL query/userinfo, and prompts for log lines."""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = _redact_json_fragments(text)
    text = _redact_malformed_sensitive_containers(text)
    text = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", text)
    text = _URL_QUERY_RE.sub(r"\1", text)
    text = _BEARER_RE.sub(r"\1 [REDACTED]", text)
    text = _COOKIE_HEADER_RE.sub(r"\1=[REDACTED]", text)
    text = _AUTH_HEADER_RE.sub(r"\1=[REDACTED]", text)
    text = _SECRET_KV_RE.sub(_replace_secret_kv, text)
    text = _CODE_KV_RE.sub(r"\1=[REDACTED]", text)
    text = _SK_KEY_RE.sub("[REDACTED]", text)
    text = _GEMINI_KEY_RE.sub("[REDACTED]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len]
    return text


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
    safe_error = _sanitize_error_for_log(error)
    logger.warning(
        f"🔄 LLM failover #{count}: {primary_provider} -> {fallback_provider} | Error: {safe_error}"
    )


def get_llm(
    provider: str = "ollama",
    fallbacks: list[Any] | None = None,
    *,
    fallback_on: list[str] | None = None,
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

        # Ordered failover: primary openai → ollama → inline model dict
        get_llm(
            provider="openai",
            model="gpt-4o",
            fallbacks=["ollama", {"provider": "openai", "model": "gpt-4o-mini"}],
            fallback_on=["timeout", "connection", "rate_limit", "empty_response"],
        )

    Args:
        provider: Provider identifier when building from factory.
        fallbacks: Optional ordered fallbacks — provider slugs, ``provider/model``
            strings, or dict / stack-entry specs with their own provider/model.
        fallback_on: Conditions that advance the chain (``timeout``,
            ``connection``, ``rate_limit``, ``empty_response``, ``any_error``).
            ``None`` uses library defaults.
        instance: Pre-built LLM instance to use directly.
        **kwargs: Provider-specific parameters for the factory.

    Returns:
        An LLM instance (possibly wrapped in :class:`FallbackLLM`).
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
            _llm_instance = _wrap_with_fallbacks(
                primary,
                provider=provider,
                primary_kwargs=kwargs,
                fallbacks=fallbacks,
                fallback_on=fallback_on,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to build {provider} LLM: {_sanitize_error_for_log(exc)}. "
                "Falling back to dummy."
            )
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


#: The pip extra that supplies each provider's client library.
_PROVIDER_EXTRAS: dict[str, str] = {
    "ollama": "ollama",
    "openai": "openai",
    "openai_compatible": "openai",
    "azure": "azure",
    "vertex": "vertex",
}


class LLMDriverMissingError(RuntimeError):
    """A configured LLM provider's client library is not installed.

    Distinct from a backend that is merely unreachable. An unreachable
    backend may recover; a missing driver never will, and until this was
    raised the platform quietly substituted a dummy model — so a deployment
    that had configured a real provider booted healthy and answered every
    request with fabricated text.
    """


def _build_llm(provider: str, **kwargs: Any) -> Any:
    """Build an LLM instance for the given provider.

    Recognised providers:

    * ``ollama`` — local Ollama server (``base_url`` respected).
    * ``openai`` — OpenAI Chat models; ``base_url`` (also accepted as
      ``api_base``) forwards to ``langchain_openai.ChatOpenAI``.
    * ``openai_compatible`` — any OpenAI-API-compatible endpoint (Groq,
      Together, LM Studio, vLLM, LiteLLM, …) — same as ``openai`` but the
      ``base_url`` is mandatory.
    * ``omlx`` — a local OpenAI-compatible small-model server (oMLX, and by
      extension llama.cpp / vLLM / LM Studio). Same wire protocol as
      ``openai``; the ``base_url`` and ``api_key`` default to
      ``OMLX_BASE_URL`` / ``OMLX_API_KEY`` so a scaffolded local stack runs
      with no credentials at all.
    * ``azure`` — Azure OpenAI; accepts ``api_base`` / ``base_url`` as the
      endpoint, plus ``api_version`` and ``deployment_name``.
    * ``vertex`` — Google Vertex AI.
    * ``dummy`` — deterministic fake for tests.

    Any name registered via :func:`register_llm_provider` is tried first,
    so hosts can add or override providers without touching this function.
    """
    provider = _llm_provider_key(provider)

    with _llm_provider_lock:
        custom_builder = _LLM_PROVIDERS.get(provider)
    if custom_builder is not None:
        return custom_builder(**kwargs)

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
        model_name = str(kwargs.get("model") or deployment or "")
        allow_temp = supports_custom_temperature(model_name)
        azure_kwargs: dict[str, Any] = {
            "api_key": kwargs.get("api_key", ""),
            "azure_endpoint": endpoint,
            "api_version": kwargs.get("api_version", "2024-02-15-preview"),
            "azure_deployment": deployment,
        }
        if kwargs.get("model"):
            azure_kwargs["model"] = kwargs["model"]
        if allow_temp:
            azure_kwargs["temperature"] = kwargs.get("temperature", 0.1)
        else:
            azure_kwargs["temperature"] = None
        if kwargs.get("max_tokens") is not None:
            azure_kwargs["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("max_completion_tokens") is not None:
            azure_kwargs["max_completion_tokens"] = kwargs["max_completion_tokens"]
        azure_fields = getattr(AzureChatOpenAI, "model_fields", None)
        if "stream_usage" in kwargs and kwargs["stream_usage"] is not None:
            azure_kwargs["stream_usage"] = kwargs["stream_usage"]
        elif isinstance(azure_fields, dict) and "stream_usage" in azure_fields:
            azure_kwargs["stream_usage"] = True
        openai_bits = _openai_compat_kwargs(kwargs)
        if not allow_temp:
            openai_bits = _strip_disallowed_temperature(openai_bits)
        azure_kwargs.update(openai_bits)
        return AzureChatOpenAI(**azure_kwargs)

    elif provider in ("openai", "openai_compatible", "omlx"):
        from langchain_openai import ChatOpenAI

        base_url = kwargs.get("base_url") or kwargs.get("api_base") or None
        if provider == "omlx":
            # A local server needs neither a URL nor a key in the stack file:
            # fall back to the same env vars the optimizer's LLMCaller reads,
            # then to oMLX's own default port.
            base_url = base_url or os.getenv("OMLX_BASE_URL") or DEFAULT_OMLX_BASE_URL
        if provider == "openai_compatible" and not base_url:
            raise ValueError(
                "openai_compatible LLMs require a base_url (or api_base). "
                "Pass e.g. base_url='https://api.groq.com/openai/v1'."
            )
        api_key = kwargs.get("api_key", "")
        if provider == "omlx" and not api_key:
            # ChatOpenAI refuses to build without *some* key; local servers
            # ignore its value.
            api_key = os.getenv("OMLX_API_KEY") or os.getenv("OPENAI_API_KEY") or "omlx"
        model_name = str(kwargs.get("model", "gpt-4"))
        allow_temp = supports_custom_temperature(model_name)
        ctor_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": model_name,
            "temperature": kwargs.get("temperature", 0.1) if allow_temp else None,
        }
        if base_url:
            ctor_kwargs["base_url"] = base_url
        if kwargs.get("max_tokens") is not None:
            ctor_kwargs["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("max_completion_tokens") is not None:
            ctor_kwargs["max_completion_tokens"] = kwargs["max_completion_tokens"]
        chat_fields = getattr(ChatOpenAI, "model_fields", None)
        if "stream_usage" in kwargs and kwargs["stream_usage"] is not None:
            ctor_kwargs["stream_usage"] = kwargs["stream_usage"]
        elif isinstance(chat_fields, dict) and "stream_usage" in chat_fields:
            ctor_kwargs["stream_usage"] = True
        for key in ("max_retries", "default_headers", "default_query"):
            if kwargs.get(key) is not None:
                ctor_kwargs[key] = kwargs[key]
        # Prefer an explicit httpx.Timeout so connect/read/write all abort.
        # A bare float is accepted by ChatOpenAI but some OpenAI-compatible
        # servers (slow token streams) can keep the socket alive past intent.
        if kwargs.get("timeout") is not None:
            import httpx

            raw_timeout = kwargs["timeout"]
            ctor_kwargs["timeout"] = (
                raw_timeout
                if isinstance(raw_timeout, httpx.Timeout)
                else httpx.Timeout(float(raw_timeout))
            )

        # Stack ``extra:`` + flat kwargs → model_kwargs / extra_body (oMLX, vLLM, …)
        openai_bits = _openai_compat_kwargs(kwargs)
        if not allow_temp:
            openai_bits = _strip_disallowed_temperature(openai_bits)
        ctor_kwargs.update(openai_bits)
        return ChatOpenAI(**ctor_kwargs)

    elif provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI  # pyright: ignore[reportDeprecated]

        model_name = kwargs.get("model", "gemini-2.0-flash")
        vertex_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "project": kwargs.get("project", ""),
            "location": kwargs.get("location", "us-central1"),
        }
        if supports_custom_temperature(str(model_name)):
            vertex_kwargs["temperature"] = kwargs.get("temperature", 0.1)
        return ChatVertexAI(**vertex_kwargs)  # pyright: ignore[reportDeprecated]

    elif provider == "dummy":
        return _build_dummy_llm()

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Built-in: ollama, azure, openai, "
            f"openai_compatible, omlx, vertex, dummy. "
            f"Registered: {registered_llm_providers()}. "
            "Use register_llm_provider(name, builder) to add a custom one."
        )


def _openai_compat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map stack ``extra`` / vendor knobs onto ChatOpenAI constructor args.

    Supports modern OpenAI-compatible servers (oMLX, vLLM, Groq, …):

    * ``response_format`` — JSON mode when the server supports it
    * ``enable_thinking`` / ``chat_template_kwargs`` — Qwen-style thinking
    * ``extra_body`` / ``model_kwargs`` — passthrough for other vendor fields
    * ``default_headers`` — custom HTTP headers
    * ``tools`` / ``tool_choice`` — function calling when bound later via
      ``.bind_tools``; stored in ``model_kwargs`` when provided at build time
    """
    import inspect

    from langchain_openai import ChatOpenAI

    extra = dict(kwargs.get("extra") or {})
    # Flat aliases often set directly on the stack profile
    for alias in (
        "response_format",
        "enable_thinking",
        "chat_template_kwargs",
        "extra_body",
        "model_kwargs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    ):
        if alias in kwargs and alias not in extra:
            extra[alias] = kwargs[alias]

    model_kwargs = dict(extra.pop("model_kwargs", None) or {})
    extra_body = dict(extra.pop("extra_body", None) or {})

    response_format = extra.pop("response_format", None)
    if response_format is not None:
        model_kwargs["response_format"] = response_format

    enable_thinking = extra.pop("enable_thinking", None)
    chat_template_kwargs = extra.pop("chat_template_kwargs", None)
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = None
    if chat_template_kwargs is None and enable_thinking is not None:
        # oMLX / Qwen chat templates read thinking via chat_template_kwargs;
        # also keep the flat key for servers that accept it at the root.
        chat_template_kwargs = {"enable_thinking": bool(enable_thinking)}
    elif (
        isinstance(chat_template_kwargs, dict)
        and enable_thinking is not None
        and "enable_thinking" not in chat_template_kwargs
    ):
        chat_template_kwargs = {
            **chat_template_kwargs,
            "enable_thinking": bool(enable_thinking),
        }
    if enable_thinking is not None:
        extra_body["enable_thinking"] = bool(enable_thinking)
    if chat_template_kwargs is not None:
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    for tool_key in ("tools", "tool_choice", "parallel_tool_calls"):
        if tool_key in extra:
            model_kwargs[tool_key] = extra.pop(tool_key)

    # Remaining unknown keys → extra_body (vendor extensions)
    reserved = {
        "provider",
        "model",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "stream_usage",
        "api_key",
        "base_url",
        "api_base",
        "api_version",
        "azure_endpoint",
        "azure_deployment",
        "deployment_name",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "extra",
        "fallbacks",
        "fallback_on",
        "instance",
        "name",
    }
    for key, value in list(extra.items()):
        if key in reserved or value is None:
            continue
        if key == "default_headers":
            continue
        extra_body.setdefault(key, value)

    out: dict[str, Any] = {}
    if "default_headers" in kwargs or "default_headers" in extra:
        out["default_headers"] = kwargs.get("default_headers") or extra.get("default_headers")

    # ChatOpenAI is a Pydantic model: ``__init__`` signature is often just
    # ``*args, **kwargs``, so inspect alone misses first-class fields like
    # ``extra_body``. Prefer model_fields / annotations, then signature.
    supports_extra_body = False
    model_fields = getattr(ChatOpenAI, "model_fields", None)
    if isinstance(model_fields, dict) and "extra_body" in model_fields:
        supports_extra_body = True
    else:
        try:
            supports_extra_body = "extra_body" in inspect.signature(ChatOpenAI.__init__).parameters
        except (TypeError, ValueError):
            supports_extra_body = False

    if extra_body:
        if supports_extra_body:
            out["extra_body"] = extra_body
        else:
            model_kwargs = {
                **model_kwargs,
                "extra_body": {**model_kwargs.get("extra_body", {}), **extra_body},
            }
    if model_kwargs:
        out["model_kwargs"] = model_kwargs
    return {k: v for k, v in out.items() if v is not None}


def _build_dummy_llm() -> Any:
    """Build a dummy LLM for testing."""
    from langchain_core.language_models import FakeListChatModel

    return FakeListChatModel(
        responses=["This is a dummy response from agentomatic."],
    )


def _normalize_fallback_specs(
    fallbacks: list[Any],
    primary_kwargs: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Normalize fallback specs into ``(label, build_kwargs)`` pairs.

    Accepted forms per item:

    * ``"ollama"`` — provider slug; inherits primary kwargs
    * ``"openai/gpt-4o-mini"`` — provider/model shorthand
    * ``{"provider": "...", "model": "...", ...}`` — full inline spec
    * objects with ``provider`` / ``model`` attributes (stack entries)

    Args:
        fallbacks: Raw fallback list from callers / stack YAML.
        primary_kwargs: Kwargs used for the primary model (inherited defaults).

    Returns:
        Ordered ``(label, kwargs)`` pairs for :func:`_build_llm`.
    """
    from agentomatic.providers.fallback import model_label

    base = {
        k: v
        for k, v in primary_kwargs.items()
        if k not in {"fallbacks", "fallback_on", "instance", "name"}
    }
    specs: list[tuple[str, dict[str, Any]]] = []
    for item in fallbacks:
        if item is None:
            continue
        if isinstance(item, str):
            token = item.strip()
            if not token:
                continue
            if "/" in token:
                provider, model = token.split("/", 1)
                kwargs = {**base, "provider": provider, "model": model}
            else:
                kwargs = {**base, "provider": token}
            specs.append((model_label(kwargs["provider"], kwargs.get("model")), kwargs))
            continue

        if hasattr(item, "model_dump") and hasattr(item, "provider"):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = dict(item)
        else:
            logger.warning("Ignoring unsupported fallback spec: {!r}", item)
            continue

        provider = str(data.pop("provider", base.get("provider", "ollama")))
        model = data.pop("model", base.get("model"))
        data.pop("fallbacks", None)
        data.pop("fallback_on", None)
        kwargs = {**base, **data, "provider": provider}
        if model is not None:
            kwargs["model"] = model
        # Prefer explicit empty api_key/base_url from fallback only when set
        for key in ("api_key", "base_url"):
            if key in data and data[key] in ("", None) and base.get(key):
                kwargs[key] = base[key]
        specs.append((model_label(provider, kwargs.get("model")), kwargs))
    return specs


def _wrap_with_fallbacks(
    primary: Any,
    *,
    provider: str,
    primary_kwargs: dict[str, Any],
    fallbacks: list[Any] | None,
    fallback_on: list[str] | None,
) -> Any:
    """Attach an ordered :class:`FallbackLLM` chain when fallbacks are configured.

    Args:
        primary: Already-built primary model.
        provider: Primary provider id (for labels).
        primary_kwargs: Primary build kwargs.
        fallbacks: Optional fallback specs.
        fallback_on: Optional trigger list.

    Returns:
        *primary* unchanged, or a :class:`FallbackLLM` wrapper.
    """
    if not fallbacks:
        return primary

    from agentomatic.providers.fallback import FallbackLLM, model_label

    def _endpoint_key(prov: str, kwargs: dict[str, Any]) -> tuple[str, str, str]:
        base = str(kwargs.get("base_url") or kwargs.get("api_base") or "").rstrip("/")
        return (str(prov or ""), str(kwargs.get("model") or ""), base)

    primary_key = _endpoint_key(provider, primary_kwargs)
    seen: set[tuple[str, str, str]] = {primary_key}

    fallback_models: list[Any] = []
    labels = [model_label(provider, primary_kwargs.get("model"))]
    for label, fb_kwargs in _normalize_fallback_specs(fallbacks, primary_kwargs):
        fb_key = _endpoint_key(str(fb_kwargs.get("provider") or ""), fb_kwargs)
        if fb_key in seen:
            logger.info(
                "Skipping duplicate LLM fallback {} (same provider/model/base_url "
                "as an earlier chain step — avoids waiting a full timeout twice)",
                label,
            )
            continue
        seen.add(fb_key)
        try:
            build_kwargs = {k: v for k, v in fb_kwargs.items() if k != "provider"}
            fallback_models.append(_build_llm(fb_kwargs["provider"], **build_kwargs))
            labels.append(label)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to build fallback LLM {}: {}",
                label,
                _sanitize_error_for_log(exc),
            )

    if not fallback_models:
        return primary

    logger.info(
        "LLM failover chain configured: {} -> {}",
        labels[0],
        labels[1:],
    )
    return FallbackLLM(
        primary,
        fallback_models,
        labels=labels,
        fallback_on=fallback_on,
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
    fallbacks: list[Any] | None = None,
    fallback_on: list[str] | None = None,
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
        fallbacks: Optional ordered fallback specs (same forms as :func:`get_llm`).
        fallback_on: Optional fallback trigger conditions.
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
            built = _wrap_with_fallbacks(
                built,
                provider=provider,
                primary_kwargs=kwargs,
                fallbacks=fallbacks,
                fallback_on=fallback_on,
            )
            _named_instances[name] = built
            logger.debug(f"Created named LLM instance '{name}' ({provider})")
        except ImportError as exc:
            extra = _PROVIDER_EXTRAS.get(provider.lower())
            install = (
                f"pip install 'agentomatic[{extra}]'"
                if extra
                else "install the provider's client library"
            )
            raise LLMDriverMissingError(
                f"LLM '{name}' is configured for provider '{provider}', but its "
                f"client library is not installed ({exc}). Fix the image rather "
                f"than the request: {install}. Refusing to substitute a dummy "
                "model — that would answer every call with fabricated text while "
                "the platform reported healthy."
            ) from exc
        except Exception as exc:
            # The backend may simply be down; that can recover, so keep the
            # dummy so local development still runs. It is logged loudly
            # because every answer from here on is fake.
            logger.warning(
                f"Failed to build LLM '{name}' ({provider}): {_sanitize_error_for_log(exc)}. "
                "Using a DUMMY model — responses are fabricated."
            )
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
        "max_tokens": entry.max_tokens,
    }
    if entry.api_key:
        kwargs["api_key"] = entry.api_key
    if entry.base_url:
        kwargs["base_url"] = entry.base_url
    kwargs.update(entry.extra)

    fallbacks: list[Any] | None = None
    fallback_on: list[str] | None = list(entry.fallback_on) if entry.fallback_on else None
    if entry.fallbacks:
        try:
            fallbacks = stack_manager.resolve_fallbacks(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to resolve stack LLM fallbacks: {}",
                _sanitize_error_for_log(exc),
            )
            fallbacks = [
                fb.model_dump() if hasattr(fb, "model_dump") else fb
                for fb in cast(list[Any], entry.fallbacks)
            ]

    try:
        instance = get_named_llm(
            name="default",
            fallbacks=fallbacks,
            fallback_on=fallback_on,
            **kwargs,
        )
        set_llm(instance)
        logger.info(
            f"🧠 Global LLM initialised from active stack "
            f"(provider={entry.provider}, model={entry.model}"
            f"{f', fallbacks={len(fallbacks)}' if fallbacks else ''})"
        )
        return instance
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to apply stack default LLM: {_sanitize_error_for_log(exc)}")
        return None


def get_llm_for_agent(
    agent_name: str,
    role: str = "default",
    stack_manager: StackManager | None = None,
) -> Any:
    """Get an LLM instance for a specific agent and role.

    Resolution order:
      1. Agent's ``llm_config`` / ``AgentLLMConfig.roles`` maps *role* → stack
         profile name (discovered under ``agents/<name>/llm.py`` when possible).
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

    profile = _resolve_agent_llm_profile(agent_name, role)
    try:
        entry = stack_manager.get_llm_config(profile)
    except (ValueError, KeyError):
        try:
            entry = stack_manager.get_llm_config("default")
        except (ValueError, KeyError):
            logger.debug(
                f"No stack LLM config for agent={agent_name} role={role} "
                f"profile={profile}, falling back to global LLM"
            )
            return get_llm()

    instance_name = f"{agent_name}:{role}"
    fallbacks: list[Any] | None = None
    fallback_on: list[str] | None = list(entry.fallback_on) if entry.fallback_on else None
    if entry.fallbacks:
        try:
            fallbacks = stack_manager.resolve_fallbacks(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to resolve fallbacks for agent={} role={}: {}",
                agent_name,
                role,
                _sanitize_error_for_log(exc),
            )
            fallbacks = [
                fb.model_dump() if hasattr(fb, "model_dump") else fb
                for fb in cast(list[Any], entry.fallbacks)
            ]

    return get_named_llm(
        name=instance_name,
        provider=entry.provider,
        model=entry.model,
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
        api_key=entry.api_key,
        base_url=entry.base_url,
        fallbacks=fallbacks,
        fallback_on=fallback_on,
        **entry.extra,
    )


def _resolve_agent_llm_profile(agent_name: str, role: str) -> str:
    """Map an agent logical *role* to a stack LLM profile name.

    Looks for ``agents/<agent_name>/llm.py`` exposing ``llm_config.roles`` or
    an ``AgentLLMConfig`` instance. Falls back to *role* itself when no
    mapping is found (so ``role="structured"`` still works as a direct
    profile name).
    """
    import importlib
    import sys
    from pathlib import Path

    candidates = [
        f"agents.{agent_name}.llm",
        f"{agent_name}.llm",
    ]
    # Also try loading from a local agents/ folder on sys.path / cwd.
    agents_dir = Path.cwd() / "agents" / agent_name / "llm.py"
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001
            continue
        cfg = getattr(mod, "llm_config", None) or getattr(mod, "AgentLLMConfig", None)
        if cfg is None:
            continue
        roles = getattr(cfg, "roles", None)
        if callable(cfg) and roles is None:
            try:
                cfg = cfg()
                roles = getattr(cfg, "roles", None)
            except Exception:  # noqa: BLE001
                roles = None
        if isinstance(roles, dict) and role in roles:
            return str(roles[role])
        if isinstance(roles, dict) and "default" in roles and role == "default":
            return str(roles["default"])

    if agents_dir.is_file() and str(Path.cwd()) not in sys.path:
        # Best-effort: path exists but module not importable yet.
        logger.debug(
            "Agent llm.py present at {} but not importable as a module",
            agents_dir,
        )
    return role


async def invoke_with_retry(
    llm: Any,
    messages: list,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    *,
    strip_thinking: bool = True,
    keep_thinking_metadata: bool = True,
) -> Any:
    """Invoke LLM with retry logic.

    Args:
        llm: Chat model / runnable with ``ainvoke``.
        messages: Chat messages.
        max_retries: Retries after the first failure.
        retry_delay: Base delay (exponential backoff).
        strip_thinking: When ``True`` (default), separate thinking/reasoning
            from the final answer so ``.content`` is safe for JSON / UX.
        keep_thinking_metadata: Store thinking on ``additional_kwargs`` when
            stripping (debug / Studio). Ignored when ``strip_thinking`` is
            ``False``.

    Returns:
        The model result (optionally normalized).
    """
    import asyncio
    import time

    from agentomatic.observability.metrics import (
        extract_token_usage,
        record_llm_call,
    )
    from agentomatic.providers.message_utils import (
        attach_thinking_metadata,
        message_thinking,
    )

    last_exc = None
    retries_used = 0
    t0 = time.perf_counter()
    outcome = "success"

    for attempt in range(max_retries + 1):
        try:
            result = await llm.ainvoke(messages)
            if not strip_thinking:
                normalized = result
            else:
                normalized = attach_thinking_metadata(result, strip_content=True)
                if not keep_thinking_metadata:
                    additional = dict(getattr(normalized, "additional_kwargs", None) or {})
                    additional.pop("thinking", None)
                    additional.pop("reasoning_content", None)
                    if hasattr(normalized, "model_copy"):
                        try:
                            normalized = normalized.model_copy(
                                update={"additional_kwargs": additional}
                            )
                        except Exception:  # noqa: BLE001
                            from types import SimpleNamespace

                            normalized = SimpleNamespace(
                                content=getattr(normalized, "content", ""),
                                additional_kwargs=additional,
                                response_metadata=getattr(normalized, "response_metadata", {}),
                                usage_metadata=(
                                    getattr(normalized, "usage_metadata", None)
                                    or getattr(result, "usage_metadata", None)
                                ),
                                raw=result,
                            )
                    else:
                        from types import SimpleNamespace

                        normalized = SimpleNamespace(
                            content=getattr(normalized, "content", ""),
                            additional_kwargs=additional,
                            response_metadata=getattr(normalized, "response_metadata", {}),
                            usage_metadata=(
                                getattr(normalized, "usage_metadata", None)
                                or getattr(result, "usage_metadata", None)
                            ),
                            raw=result,
                        )

            duration = time.perf_counter() - t0
            prompt_t, completion_t, _total, thinking_t = extract_token_usage(normalized)
            try:
                from agentomatic.logs.runtime import record_llm_usage

                resp_meta = getattr(normalized, "response_metadata", None) or {}
                resolved_model = (
                    (resp_meta.get("model_name") or resp_meta.get("model"))
                    if isinstance(resp_meta, dict)
                    else None
                ) or (
                    getattr(llm, "model_name", None)
                    or getattr(llm, "model", None)
                    or getattr(llm, "deployment_name", None)
                )
                record_llm_usage(
                    model=str(resolved_model) if resolved_model else None,
                    input_tokens=prompt_t,
                    output_tokens=completion_t,
                    total_tokens=_total or (prompt_t + completion_t),
                )
            except Exception:  # noqa: BLE001
                pass
            thinking_text = ""
            try:
                thinking_text = message_thinking(normalized) or ""
            except Exception:  # noqa: BLE001
                thinking_text = ""
            # Heuristic thinking duration when provider does not split phases.
            thinking_s = 0.0
            if thinking_text and duration > 0:
                ratio = min(0.85, len(thinking_text) / max(1, len(thinking_text) + 200))
                thinking_s = duration * ratio
            record_llm_call(
                llm=llm,
                outcome=outcome,
                stream=False,
                duration_s=duration,
                ttft_s=duration if not thinking_text else max(0.01, duration - thinking_s),
                thinking_s=thinking_s if thinking_s > 0 else None,
                prompt_tokens=prompt_t,
                completion_tokens=completion_t,
                thinking_tokens=thinking_t or (len(thinking_text.split()) if thinking_text else 0),
                retries=retries_used,
                retry_reason="invoke",
            )
            return normalized
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                retries_used += 1
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"LLM attempt {attempt + 1} failed: {_sanitize_error_for_log(exc)}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

    duration = time.perf_counter() - t0
    err_name = type(last_exc).__name__ if last_exc else "error"
    if "timeout" in err_name.lower() or "timeout" in str(last_exc).lower():
        outcome = "timeout"
    elif "http" in err_name.lower() or "status" in str(last_exc).lower():
        outcome = "http_error"
    else:
        outcome = "error"
    record_llm_call(
        llm=llm,
        outcome=outcome,
        stream=False,
        duration_s=duration,
        retries=retries_used,
        retry_reason="invoke",
    )
    raise last_exc  # type: ignore[misc]


async def astream_with_thinking(
    llm: Any,
    messages: list,
    *,
    emit_thinking: bool = False,
) -> Any:
    """Async-stream an LLM, separating thinking tokens from answer tokens.

    Yields dicts::

        {"type": "thinking" | "answer" | "done", "text": str, "thinking": str}

    When the server streams reasoning in separate fields / block types those
    chunks are classified as ``thinking``. Tagged ``<think>`` spans that arrive
    inline are buffered until the closing tag when possible.
    """
    import time

    from agentomatic.observability.metrics import extract_token_usage, record_llm_call
    from agentomatic.providers.message_utils import (
        THINK_TAG_RE,
        message_text,
        message_thinking,
        split_thinking_text,
    )

    t0 = time.perf_counter()
    ttft_s: float | None = None
    thinking_started: float | None = None
    thinking_ended: float | None = None
    first_answer_at: float | None = None

    def _mark_first() -> None:
        nonlocal ttft_s
        if ttft_s is None:
            ttft_s = time.perf_counter() - t0

    if not hasattr(llm, "astream"):
        result = await invoke_with_retry(llm, messages, max_retries=0)
        thinking = message_thinking(result)
        answer = message_text(result)
        if emit_thinking and thinking:
            yield {"type": "thinking", "text": thinking, "thinking": thinking}
        if answer:
            yield {"type": "answer", "text": answer, "thinking": thinking}
        yield {"type": "done", "text": answer, "thinking": thinking}
        return

    answer_buf: list[str] = []
    thinking_buf: list[str] = []
    inline_buf = ""
    in_think_tag = False
    latest_usage_chunk: Any = None
    latest_model_name: str | None = None

    async for chunk in llm.astream(messages):
        if getattr(chunk, "usage_metadata", None) is not None:
            latest_usage_chunk = chunk
        chunk_meta = getattr(chunk, "response_metadata", None)
        if isinstance(chunk_meta, dict):
            if chunk_meta.get("token_usage") or chunk_meta.get("usage"):
                latest_usage_chunk = chunk
            chunk_model = chunk_meta.get("model_name") or chunk_meta.get("model")
            if chunk_model:
                latest_model_name = str(chunk_model)

        # Structured reasoning on the chunk
        for attr in ("reasoning_content", "reasoning", "thinking"):
            piece = getattr(chunk, attr, None)
            if piece:
                text = str(piece)
                thinking_buf.append(text)
                _mark_first()
                if thinking_started is None:
                    thinking_started = time.perf_counter()
                thinking_ended = time.perf_counter()
                if emit_thinking:
                    yield {"type": "thinking", "text": text, "thinking": "".join(thinking_buf)}

        additional = getattr(chunk, "additional_kwargs", None) or {}
        if isinstance(additional, dict):
            for key in ("reasoning_content", "reasoning", "thinking"):
                if additional.get(key):
                    text = str(additional[key])
                    thinking_buf.append(text)
                    _mark_first()
                    if thinking_started is None:
                        thinking_started = time.perf_counter()
                    thinking_ended = time.perf_counter()
                    if emit_thinking:
                        yield {
                            "type": "thinking",
                            "text": text,
                            "thinking": "".join(thinking_buf),
                        }

        content = getattr(chunk, "content", None)
        if content is None:
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and str(block.get("type", "")).lower() in {
                    "thinking",
                    "reasoning",
                    "reason",
                }:
                    text = str(block.get("thinking") or block.get("text") or "")
                    if text:
                        thinking_buf.append(text)
                        _mark_first()
                        if thinking_started is None:
                            thinking_started = time.perf_counter()
                        thinking_ended = time.perf_counter()
                        if emit_thinking:
                            yield {
                                "type": "thinking",
                                "text": text,
                                "thinking": "".join(thinking_buf),
                            }
                elif isinstance(block, dict) and block.get("text"):
                    content = block["text"]
                    break
                else:
                    content = str(block)
                    break
            else:
                continue

        text = str(content)
        inline_buf += text

        # Flush complete <think>...</think> regions from the inline buffer.
        while True:
            match = THINK_TAG_RE.search(inline_buf)
            if not match:
                break
            thinking_buf.append(match.group(1).strip())
            _mark_first()
            if thinking_started is None:
                thinking_started = time.perf_counter()
            thinking_ended = time.perf_counter()
            if emit_thinking and match.group(1).strip():
                yield {
                    "type": "thinking",
                    "text": match.group(1).strip(),
                    "thinking": "\n\n".join(thinking_buf),
                }
            inline_buf = inline_buf[: match.start()] + inline_buf[match.end() :]

        open_idx = inline_buf.lower().find("<think")
        if open_idx != -1 and "</think>" not in inline_buf.lower():
            # Hold partial think tag; emit any leading answer text.
            leading = inline_buf[:open_idx]
            inline_buf = inline_buf[open_idx:]
            in_think_tag = True
            if leading:
                answer_buf.append(leading)
                _mark_first()
                if first_answer_at is None:
                    first_answer_at = time.perf_counter()
                yield {
                    "type": "answer",
                    "text": leading,
                    "thinking": "\n\n".join(thinking_buf),
                }
            continue

        if in_think_tag:
            continue

        if inline_buf:
            answer_buf.append(inline_buf)
            _mark_first()
            if first_answer_at is None:
                first_answer_at = time.perf_counter()
            yield {
                "type": "answer",
                "text": inline_buf,
                "thinking": "\n\n".join(thinking_buf),
            }
            inline_buf = ""

    # Trailing buffer (unclosed think → treat as thinking; else answer)
    if inline_buf:
        split = split_thinking_text(inline_buf)
        if split.thinking:
            thinking_buf.append(split.thinking)
            if emit_thinking:
                yield {
                    "type": "thinking",
                    "text": split.thinking,
                    "thinking": "\n\n".join(thinking_buf),
                }
        if split.answer:
            answer_buf.append(split.answer)
            yield {
                "type": "answer",
                "text": split.answer,
                "thinking": "\n\n".join(thinking_buf),
            }

    duration = time.perf_counter() - t0
    thinking_s = None
    if thinking_started is not None and thinking_ended is not None:
        thinking_s = max(0.0, thinking_ended - thinking_started)
    generation_s = None
    if first_answer_at is not None:
        generation_s = max(0.0, time.perf_counter() - first_answer_at)

    prompt_t = completion_t = total_t = thinking_t = 0
    if latest_usage_chunk is not None:
        prompt_t, completion_t, total_t, thinking_t = extract_token_usage(latest_usage_chunk)
    heuristic_completion = len("".join(answer_buf).split())
    resolved_completion = (
        completion_t if (completion_t > 0 or prompt_t > 0) else heuristic_completion
    )
    resolved_thinking = thinking_t or (
        len("\n\n".join(thinking_buf).split()) if thinking_buf else 0
    )
    resolved_model = (
        latest_model_name
        or getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or getattr(llm, "deployment_name", None)
    )
    try:
        from agentomatic.logs.runtime import record_llm_usage

        record_llm_usage(
            model=str(resolved_model) if resolved_model else None,
            input_tokens=prompt_t,
            output_tokens=completion_t,
            total_tokens=total_t or (prompt_t + completion_t),
        )
    except Exception:  # noqa: BLE001
        pass

    record_llm_call(
        llm=llm,
        outcome="success",
        stream=True,
        duration_s=duration,
        ttft_s=ttft_s,
        thinking_s=thinking_s,
        generation_s=generation_s,
        prompt_tokens=prompt_t,
        completion_tokens=resolved_completion,
        thinking_tokens=resolved_thinking,
    )

    yield {
        "type": "done",
        "text": "".join(answer_buf),
        "thinking": "\n\n".join(thinking_buf),
    }


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
        from agentomatic.providers.message_utils import message_text, strip_thinking_for_json

        content = message_text(res)
        content = strip_thinking_for_json(content)
        import json

        from pydantic import BaseModel
        from pydantic_core import PydanticUndefined

        if issubclass(self.response_model, BaseModel):
            try:
                data = json.loads(content)
                return self.response_model.model_validate(data)
            except Exception:
                # Try first balanced object after thinking strip
                try:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end > start:
                        data = json.loads(content[start : end + 1])
                        return self.response_model.model_validate(data)
                except Exception as exc:
                    logger.warning(
                        f"Structured output parse failed for "
                        f"{self.response_model.__name__}: {_sanitize_error_for_log(exc)}. "
                        f"Falling back to dummy values. "
                        f"Raw content (first 200 chars): "
                        f"{_sanitize_error_for_log(content[:200])!r}"
                    )
                    try:
                        from agentomatic.observability.metrics import record_structure_error

                        record_structure_error(llm=self.llm, agent="structured_output")
                    except Exception:  # noqa: BLE001
                        pass
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
    structured output parsing. Thinking preambles are stripped in the
    fallback parser path.

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
