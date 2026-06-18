"""Unified LLM calling abstraction for the ``optimize`` module.

This module centralises all LLM interactions used during prompt
optimisation — strategy rewriting, metric evaluation, dataset
synthesis, etc.  Instead of each module importing *httpx* and
calling Ollama directly, every call goes through
:class:`LLMCaller` which handles:

* **Provider routing** — ``ollama/``, ``openai/``, ``litellm/`` prefixes.
* **Graceful degradation** — failures are logged and an empty string
  (or empty dict for JSON calls) is returned so the caller never crashes.
* **JSON extraction** — :meth:`LLMCaller.call_with_json` strips
  markdown fences and retries automatically on parse errors.

Example
-------
>>> text = await LLMCaller.call("ollama/mistral:7b", "Say hello")
>>> data = await LLMCaller.call_with_json("openai/gpt-4o-mini", "Return {\"ok\": true}")
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

# =====================================================================
# Constants
# =====================================================================

_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"

_SUPPORTED_PROVIDERS = ("ollama", "openai", "litellm")

# Regex for stripping markdown code fences (```json ... ``` or ``` ... ```)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


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
    """
    for provider in _SUPPORTED_PROVIDERS:
        prefix = f"{provider}/"
        if model.startswith(prefix):
            return provider, model[len(prefix):]
    return "ollama", model


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object extraction from free-form LLM text.

    Strategy:
    1. Strip markdown code fences.
    2. Locate the first ``{`` and the **last** ``}`` and try to parse
       the substring between them.
    3. Raise :exc:`ValueError` if nothing works.
    """
    # 1. Try stripping code fences first
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Find first '{' … last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object delimiters found in response")

    candidate = text[start : end + 1]
    return json.loads(candidate)  # type: ignore[no-any-return]


# =====================================================================
# LLMCaller
# =====================================================================


class LLMCaller:
    """Unified, provider-agnostic LLM caller for the optimise pipeline.

    All methods are **static / async** so that no instance state is
    required — just call ``await LLMCaller.call(model, prompt)``.
    """

    # -----------------------------------------------------------------
    # Core call
    # -----------------------------------------------------------------

    @staticmethod
    async def call(
        model: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
        timeout: float = 120.0,
    ) -> str:
        """Send a single prompt to *model* and return the generated text.

        Parameters
        ----------
        model:
            Model specification, optionally prefixed with the provider
            (e.g. ``"ollama/mistral:7b"``, ``"openai/gpt-4o-mini"``).
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

        Returns
        -------
        str
            Generated text, or ``""`` on failure.
        """
        provider, model_name = parse_model_spec(model)
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
        model: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Call the LLM expecting a JSON object response.

        The *prompt* is augmented with an instruction to reply in JSON.
        If the first attempt fails to parse, the call is retried up to
        *max_retries* times.  Returns an empty ``{}`` on total failure.

        Parameters
        ----------
        model:
            Model specification (see :meth:`call`).
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

        Returns
        -------
        dict[str, Any]
            Parsed JSON object, or ``{}`` on failure.
        """
        json_instruction = (
            "\n\nIMPORTANT: Reply with ONLY a valid JSON object. "
            "Do not include any other text, explanation, or markdown formatting."
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


async def _call_openai(
    model_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
    timeout: float = 120.0,
) -> str:
    """Call the OpenAI chat completions API (lazy import)."""
    import openai

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    client = openai.AsyncOpenAI(timeout=timeout)
    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    return (choice.message.content or "").strip()


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
    return (response.choices[0].message.content or "").strip()
