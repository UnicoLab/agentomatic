"""Unified LLM calling abstraction for the ``optimize`` module.

This module centralises all LLM interactions used during prompt
optimisation — strategy rewriting, metric evaluation, dataset
synthesis, etc.  Instead of each module importing *httpx* and
calling Ollama directly, every call goes through
:class:`LLMCaller` which handles:

* **Provider routing** — ``ollama/``, ``openai/``, ``omlx/``, ``gemini/``,
  ``litellm/`` prefixes (``omlx/`` targets a local OpenAI-compatible server;
  ``gemini/`` uses the Google Generative Language API + ``GEMINI_API_KEY``).
* **Callable dispatch** — custom async/sync callables and LangChain
  models are transparently supported via :data:`LLMSpec`.
* **Graceful degradation** — failures are logged and an empty string
  (or empty dict for JSON calls) is returned so the caller never crashes.
* **JSON extraction** — :meth:`LLMCaller.call_with_json` strips
  markdown fences and retries automatically on parse errors.

Example
-------
>>> text = await LLMCaller.call("ollama/mistral:7b", "Say hello")
>>> data = await LLMCaller.call_with_json("openai/gpt-4o-mini", "Return {\"ok\": true}")
>>>
>>> # Custom callable also works:
>>> async def my_llm(prompt, *, system_prompt=None): return "Hello!"
>>> text = await LLMCaller.call(my_llm, "Say hello")
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec

# =====================================================================
# Constants
# =====================================================================

_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"
_DEFAULT_OMLX_BASE_URL = "http://127.0.0.1:8000/v1"

_SUPPORTED_PROVIDERS = ("ollama", "openai", "litellm", "omlx", "gemini")
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


# =====================================================================
# Helpers
# =====================================================================


def parse_model_spec(model: str) -> tuple[str, str]:
    """Split a *provider/model_name* string into ``(provider, model_name)``.

    If *model* has no ``/`` prefix matching a known provider the default
    provider ``"ollama"`` is assumed for backward compatibility.

    Examples
    --------
    >>> parse_model_spec("ollama/mistral:7b")
    ('ollama', 'mistral:7b')
    >>> parse_model_spec("openai/gpt-4o-mini")
    ('openai', 'gpt-4o-mini')
    >>> parse_model_spec("mistral:7b")
    ('ollama', 'mistral:7b')
    >>> parse_model_spec("litellm/anthropic/claude-3-haiku")
    ('litellm', 'anthropic/claude-3-haiku')
    >>> parse_model_spec("gemini/gemini-3.1-flash-lite")
    ('gemini', 'gemini-3.1-flash-lite')
    """
    for provider in _SUPPORTED_PROVIDERS:
        prefix = f"{provider}/"
        if model.startswith(prefix):
            return provider, model[len(prefix) :]
    return "ollama", model


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object extraction from free-form LLM text.

    Delegates to :func:`agentomatic.providers.jsonutil.extract_json_object`
    (fence strip, thinking strip, trailing-comma repair). Raises
    :exc:`ValueError` when nothing parses — matching the previous contract.
    """
    from agentomatic.providers.jsonutil import extract_json_object

    result = extract_json_object(text)
    if result is None:
        raise ValueError("No JSON object delimiters found in response")
    return result


# =====================================================================
# LLMCaller
# =====================================================================


class LLMCaller:
    """Unified, provider-agnostic LLM caller for the optimise pipeline.

    All methods are **static / async** so that no instance state is
    required — just call ``await LLMCaller.call(model, prompt)``.

    Accepts both **string model specs** (e.g. ``"ollama/mistral:7b"``) and
    **custom callables** (async functions, LangChain models, etc.) via the
    :data:`~agentomatic.optimize.llm_types.LLMSpec` union type.

    Class-level defaults for OpenAI-compatible servers can be set via
    :meth:`configure`, which is useful when running against a local
    OpenAI-compatible endpoint (omlx, Ollama, vLLM, LM Studio) rather
    than ``api.openai.com``.
    """

    # Class-level defaults — applied to every ``openai/`` call when set.
    _default_base_url: str | None = None
    _default_api_key: str | None = None

    @classmethod
    def configure(
        cls,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Set class-level defaults for OpenAI-compatible API calls.

        These defaults are used by every subsequent ``LLMCaller.call()``
        with an ``openai/`` model spec unless the call explicitly passes
        its own ``base_url`` / ``api_key``.

        Args:
            base_url: Base URL of the OpenAI-compatible server, e.g.
                ``"http://127.0.0.1:8000/v1"``.
            api_key: API key (may be arbitrary for local servers).
        """
        cls._default_base_url = base_url
        cls._default_api_key = api_key

    # -----------------------------------------------------------------
    # Core call
    # -----------------------------------------------------------------

    @staticmethod
    async def call(
        model: LLMSpec,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
        timeout: float = 120.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> str:
        """Send a single prompt to *model* and return the generated text.

        Parameters
        ----------
        model:
            Model specification — a string like ``"ollama/mistral:7b"``
            or a callable / LangChain model matching :data:`LLMSpec`.
        prompt:
            User prompt text.
        system_prompt:
            Optional system-level instruction.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum tokens to generate.
        json_mode:
            When ``True`` the provider is asked to return valid JSON
            (supported by OpenAI and Ollama ``format: json``).
        timeout:
            HTTP / client timeout in seconds.
        base_url:
            Override the OpenAI-compatible server URL (``openai/``
            provider only).  Falls back to :attr:`LLMCaller._default_base_url`
            then to the ``OPENAI_BASE_URL`` env var.
        api_key:
            Override the API key (``openai/`` provider only).  Falls back
            to :attr:`LLMCaller._default_api_key` then to the
            ``OPENAI_API_KEY`` env var.

        Returns
        -------
        str
            Generated text, or ``""`` on failure.
        """
        # ── Non-string: delegate to unified callable dispatcher ──
        if not isinstance(model, str):
            from agentomatic.optimize.llm_types import call_llm

            return await call_llm(
                model,
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                timeout=timeout,
            )

        # ── String: route to provider backend ────────────────────
        provider, model_name = parse_model_spec(model)
        # Resolve effective base_url / api_key for OpenAI calls.
        # Track whether the base was set explicitly (call arg or configure)
        # so cloud openai/* models are not hijacked by a local OPENAI_BASE_URL.
        explicit_base = base_url is not None or LLMCaller._default_base_url is not None
        eff_base_url = base_url or LLMCaller._default_base_url
        eff_api_key = api_key or LLMCaller._default_api_key
        try:
            if provider == "ollama":
                return await _call_ollama(
                    model_name,
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                )
            if provider == "openai":
                return await _call_openai(
                    model_name,
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                    base_url=eff_base_url,
                    api_key=eff_api_key,
                    explicit_base=explicit_base,
                )
            if provider == "omlx":
                return await _call_openai(
                    model_name,
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                    base_url=os.getenv("OMLX_BASE_URL", _DEFAULT_OMLX_BASE_URL),
                    api_key=os.getenv("OMLX_API_KEY") or os.getenv("OPENAI_API_KEY") or "omlx",
                    disable_thinking=True,
                    explicit_base=True,
                )
            if provider == "gemini":
                return await _call_gemini(
                    model_name,
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                    api_key=eff_api_key,
                )
            if provider == "litellm":
                return await _call_litellm(
                    model_name,
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            logger.warning(f"Unknown provider '{provider}', falling back to Ollama")
            return await _call_ollama(
                model_name,
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM call failed ({provider}/{model_name}): {exc}")
            return ""

    # -----------------------------------------------------------------
    # JSON convenience
    # -----------------------------------------------------------------

    @staticmethod
    async def call_with_json(
        model: LLMSpec,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_retries: int = 2,
        timeout: float = 120.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Call the LLM expecting a JSON object response.

        The *prompt* is augmented with an instruction to reply in JSON.
        If the first attempt fails to parse, the call is retried up to
        *max_retries* times.  Returns an empty ``{}`` on total failure.

        Accepts both string model specs and callables via :data:`LLMSpec`.

        Parameters
        ----------
        model:
            Model specification — string or callable (see :meth:`call`).
        prompt:
            User prompt — a JSON-return instruction is appended automatically.
        system_prompt:
            Optional system instruction.
        temperature:
            Sampling temperature (default lower for structured output).
        max_retries:
            How many extra attempts after the first parse failure.
        timeout:
            HTTP / client timeout in seconds.
        base_url:
            Override the OpenAI-compatible server URL (``openai/`` only).
        api_key:
            Override the API key (``openai/`` only).

        Returns
        -------
        dict[str, Any]
            Parsed JSON object, or ``{}`` on failure.
        """
        # For non-string models, delegate to call_llm_json
        if not isinstance(model, str):
            from agentomatic.optimize.llm_types import call_llm_json

            return await call_llm_json(
                model,
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
            )

        json_instruction = (
            "\n\nIMPORTANT: Reply with ONLY a valid JSON object. "
            "Do not include any other text, explanation, or "
            "markdown formatting."
        )
        augmented_prompt = prompt + json_instruction

        last_error: str = ""
        for attempt in range(1 + max_retries):
            raw = await LLMCaller.call(
                model,
                augmented_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=True,
                timeout=timeout,
                base_url=base_url,
                api_key=api_key,
            )
            if not raw:
                last_error = "empty response from LLM"
                logger.debug(f"JSON call attempt {attempt + 1}: empty response")
                continue
            try:
                return _extract_json_object(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                logger.debug(
                    f"JSON parse attempt {attempt + 1} failed: {exc} — "
                    f"raw (first 200 chars): {raw[:200]!r}"
                )

        logger.warning(f"call_with_json failed after {1 + max_retries} attempts: {last_error}")
        return {}


# =====================================================================
# Provider back-ends (private)
# =====================================================================


async def _call_ollama(
    model_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
    timeout: float = 120.0,
) -> str:
    """Call the local Ollama HTTP API."""
    import httpx

    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt
    if json_mode:
        payload["format"] = "json"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(_OLLAMA_GENERATE_ENDPOINT, json=payload)
        resp.raise_for_status()
        return str(resp.json().get("response", "")).strip()


def _normalize_llm_text(text: str) -> str:
    """Strip thinking / reasoning preambles before returning optimize text."""
    from agentomatic.providers.message_utils import strip_thinking_for_json

    return strip_thinking_for_json(text or "").strip()


def _is_openai_compatible_local(base_url: str | None) -> bool:
    """Return True when *base_url* looks like a local OpenAI-compatible server."""
    if not base_url:
        return False
    lowered = base_url.lower()
    if "api.openai.com" in lowered:
        return False
    return any(
        token in lowered
        for token in ("127.0.0.1", "localhost", "0.0.0.0", ":8000", ":11434", ":1234")
    )


def _looks_like_openai_cloud_model(model_name: str) -> bool:
    """Return True for first-party OpenAI chat / reasoning model ids."""
    name = (model_name or "").strip().lower()
    if not name:
        return False
    return bool(
        re.match(
            r"^(gpt-|o[1-9]|chatgpt-|text-embedding-|whisper-|tts-|omni-)",
            name,
        )
    )


def _openai_uses_max_completion_tokens(model_name: str) -> bool:
    """Newer OpenAI models reject ``max_tokens`` in favour of max_completion_tokens."""
    name = (model_name or "").strip().lower()
    return name.startswith(("o1", "o3", "o4", "gpt-5"))


def _openai_supports_temperature(model_name: str) -> bool:
    """Reasoning models often only allow the default temperature."""
    name = (model_name or "").strip().lower()
    return not name.startswith(("o1", "o3", "o4"))


def _resolve_openai_endpoint(
    model_name: str,
    *,
    base_url: str | None,
    api_key: str | None,
    explicit_base: bool,
) -> tuple[str | None, str | None]:
    """Resolve OpenAI base URL + API key with safe cloud defaults.

    Priority for base URL:
    1. Explicit *base_url* from the call / ``LLMCaller.configure``
    2. ``OPENAI_BASE_URL`` — **except** when it points at a local server and
       *model_name* looks like a first-party OpenAI cloud model.  In that
       case we ignore the local env hijack so ``openai/gpt-4o-mini`` still
       hits ``api.openai.com`` while local work uses ``omlx/`` or an
       explicit ``llm_base_url``.

    Priority for API key:
    1. Explicit *api_key*
    2. ``OPENAI_API_KEY``
    3. For local endpoints only: a placeholder ``"local"``
    """
    env_base = os.getenv("OPENAI_BASE_URL") or None
    env_key = os.getenv("OPENAI_API_KEY") or None

    resolved_base = base_url
    if resolved_base is None and env_base:
        if (
            not explicit_base
            and _is_openai_compatible_local(env_base)
            and _looks_like_openai_cloud_model(model_name)
        ):
            logger.info(
                "Ignoring local OPENAI_BASE_URL={} for cloud model openai/{} "
                "(use omlx/… or LLMCaller.configure/llm_base_url for local)",
                env_base,
                model_name,
            )
            resolved_base = None
        else:
            resolved_base = env_base

    resolved_key = api_key or env_key
    is_local = _is_openai_compatible_local(resolved_base)
    if not resolved_key:
        if is_local:
            resolved_key = "local"
        elif _looks_like_openai_cloud_model(model_name) or resolved_base is None:
            raise ValueError(
                "OPENAI_API_KEY is required for openai/ cloud models "
                f"(model={model_name!r}). Set the env var or pass api_key=…"
            )
    return resolved_base, resolved_key


async def _call_openai(
    model_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
    timeout: float = 120.0,
    base_url: str | None = None,
    api_key: str | None = None,
    disable_thinking: bool | None = None,
    explicit_base: bool = False,
) -> str:
    """Call OpenAI or an OpenAI-compatible chat completions API.

    Honours explicit *base_url* / *api_key*, then ``LLMCaller`` defaults /
    env vars. Cloud model ids (``gpt-*``, ``o1``/``o3``/…) are not silently
    redirected to a local ``OPENAI_BASE_URL`` unless the caller set an
    explicit base (``LLMCaller.configure`` / per-call ``base_url`` /
    ``PromptFitter(llm_base_url=…)``).

    For local compatible servers (oMLX, vLLM, …) thinking is disabled via
    ``chat_template_kwargs`` and residual thinking preambles are stripped.
    """
    import openai

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # explicit_base=True when caller/configure supplied a base_url.
    resolved_base, resolved_key = _resolve_openai_endpoint(
        model_name,
        base_url=base_url,
        api_key=api_key,
        explicit_base=explicit_base or bool(base_url),
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if _openai_supports_temperature(model_name):
        kwargs["temperature"] = temperature
    if _openai_uses_max_completion_tokens(model_name):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if json_mode and _openai_supports_temperature(model_name):
        # json_object mode is unreliable / unsupported on some reasoning models
        kwargs["response_format"] = {"type": "json_object"}

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base
    if resolved_key:
        client_kwargs["api_key"] = resolved_key

    if disable_thinking is None:
        disable_thinking = _is_openai_compatible_local(resolved_base)
    if disable_thinking:
        # oMLX / Qwen3.x: top-level chat_template_kwargs disables CoT leakage.
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }

    client = openai.AsyncOpenAI(**client_kwargs)
    async with client as c:
        try:
            response = await c.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Retry once without temperature / response_format for picky models.
            msg = str(exc).lower()
            retried = False
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                retried = True
            if "response_format" in msg and "response_format" in kwargs:
                kwargs.pop("response_format", None)
                retried = True
            if "max_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                retried = True
            if not retried:
                raise
            response = await c.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""
        if content:
            return _normalize_llm_text(content)
        if reasoning:
            return _normalize_llm_text(str(reasoning))
        return ""


async def _call_gemini(
    model_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> str:
    """Call Google Gemini via the Generative Language REST API.

    Uses ``GEMINI_API_KEY`` (or *api_key* / ``GOOGLE_API_KEY``). Model specs
    look like ``gemini/gemini-3.1-flash-lite`` → ``gemini-3.1-flash-lite``.
    """
    import httpx

    key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
        or ""
    )
    if not key:
        raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for gemini/ models")

    # Allow bare ids or models/ prefix from env configs.
    model_id = model_name.removeprefix("models/")
    url = f"{_GEMINI_BASE_URL}/models/{model_id}:generateContent"
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, params={"key": key}, json=payload)
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        # Surface block reasons instead of a silent empty string upstream.
        feedback = data.get("promptFeedback") or {}
        raise RuntimeError(f"Gemini returned no candidates: {feedback or data}")

    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
    return _normalize_llm_text("\n".join(t for t in texts if t))


async def _call_litellm(
    model_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: float = 120.0,
) -> str:
    """Call any supported model via litellm (lazy import)."""
    import litellm

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = await litellm.acompletion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return _normalize_llm_text(response.choices[0].message.content or "")
