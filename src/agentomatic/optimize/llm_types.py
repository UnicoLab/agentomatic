"""Pluggable LLM type system for the ``optimize`` module.

Defines a :class:`LLMCallable` protocol and a :data:`LLMSpec` union type
so that every component in the optimisation pipeline can accept **either**
a model-spec string (``"ollama/mistral:7b"``, ``"openai/gpt-4o"``) **or**
a user-supplied async callable / LangChain model.

Quick-start
-----------
>>> from agentomatic.optimize.llm_types import LLMSpec, call_llm
>>>
>>> # String-based (existing behavior)
>>> text = await call_llm("ollama/mistral:7b", "Say hello")
>>>
>>> # Custom callable
>>> async def my_llm(prompt: str, *, system_prompt: str | None = None) -> str:
...     return "Hello from my model!"
>>> text = await call_llm(my_llm, "Say hello")
>>>
>>> # LangChain model (duck-typed via ainvoke/invoke)
>>> from langchain_ollama import ChatOllama
>>> text = await call_llm(ChatOllama(model="mistral:7b"), "Say hello")
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Protocol, runtime_checkable

# =====================================================================
# Protocol / Type Alias
# =====================================================================


@runtime_checkable
class LLMCallable(Protocol):
    """Protocol for custom LLM callables.

    Any async function or object with ``__call__`` matching this
    signature can be used wherever :data:`LLMSpec` is accepted.

    Example::

        async def my_custom_llm(
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            return await my_inference_engine(prompt, system=system_prompt)

        optimizer = PromptOptimizer(
            agent="my_agent",
            llm=my_custom_llm,  # ← works!
        )

    Objects implementing the LangChain chat-model protocol
    (``ainvoke`` / ``invoke``) are also accepted — see
    :func:`call_llm` for the full dispatch logic.
    """

    async def __call__(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str: ...


LLMSpec = str | LLMCallable
"""Union type for LLM model specifications.

Accepted everywhere a model is needed:

* **str** — provider/model-name string routed through
  :class:`~agentomatic.optimize.llm_caller.LLMCaller`
  (e.g. ``"ollama/mistral:7b"``, ``"openai/gpt-4o"``).
* **LLMCallable** — any async callable matching the
  :class:`LLMCallable` protocol.
* Objects with ``ainvoke`` or ``invoke`` (LangChain protocol)
  are also accepted at runtime.
"""


# =====================================================================
# Unified calling layer
# =====================================================================


async def call_llm(
    model: LLMSpec,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
    timeout: float = 120.0,
) -> str:
    """Call an LLM using any supported model specification.

    Dispatch logic (checked in order):

    1. **str** — delegate to :meth:`LLMCaller.call` for
       provider-routed HTTP calls.
    2. **LangChain protocol** — if *model* has ``ainvoke`` or
       ``invoke``, call with ``[SystemMessage, HumanMessage]``.
    3. **Async callable** — ``await model(prompt, system_prompt=…)``.
    4. **Sync callable** — run in executor with
       ``model(prompt, system_prompt=…)``.

    Args:
        model: Model specification — string, callable, or LangChain model.
        prompt: User prompt text.
        system_prompt: Optional system-level instruction.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        json_mode: Request JSON-formatted output (string models only).
        timeout: HTTP / client timeout in seconds (string models only).

    Returns:
        Generated text, or ``""`` on failure.

    Raises:
        TypeError: If *model* is not a string, callable, or LangChain model.
    """
    # ── String: delegate to LLMCaller ────────────────────────────
    if isinstance(model, str):
        from agentomatic.optimize.llm_caller import LLMCaller

        return await LLMCaller.call(
            model,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
        )

    # ── Non-string dispatch: all paths have graceful error handling ─
    from loguru import logger

    try:
        # ── LangChain protocol: ainvoke ──────────────────────────
        if hasattr(model, "ainvoke"):
            return await _call_langchain_async(
                model,
                prompt,
                system_prompt,
            )

        # ── LangChain protocol: invoke (sync) ────────────────────
        if hasattr(model, "invoke"):
            return await asyncio.to_thread(
                _call_langchain_sync,
                model,
                prompt,
                system_prompt,
            )

        # ── Async callable ───────────────────────────────────────
        if callable(model) and (
            inspect.iscoroutinefunction(model)
            or inspect.iscoroutinefunction(
                getattr(model, "__call__", None),
            )
        ):
            result = await model(
                prompt,
                system_prompt=system_prompt,
            )
            return str(result)

        # ── Sync callable ────────────────────────────────────────
        if callable(model):
            result = await asyncio.to_thread(
                model,  # type: ignore[arg-type]
                prompt,
                system_prompt=system_prompt,
            )
            return str(result)
    except Exception as exc:  # noqa: BLE001
        model_name = getattr(model, "__name__", None) or type(model).__name__
        logger.warning(
            f"LLM callable '{model_name}' failed: {exc}",
        )
        return ""

    raise TypeError(
        f"model must be a string, async callable, or LangChain-compatible "
        f"object (with ainvoke/invoke). Got: {type(model).__name__}"
    )


async def call_llm_json(
    model: LLMSpec,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.3,
    max_retries: int = 2,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call an LLM expecting a JSON object response.

    For **string** models, delegates to
    :meth:`LLMCaller.call_with_json` which handles JSON extraction
    and retries.  For **callable** models, augments the prompt with
    a JSON instruction and parses the response.

    Args:
        model: Model specification — string, callable, or LangChain model.
        prompt: User prompt — a JSON-return instruction is appended.
        system_prompt: Optional system instruction.
        temperature: Sampling temperature (default lower for JSON).
        max_retries: Extra attempts after a parse failure.
        timeout: HTTP timeout in seconds (string models only).

    Returns:
        Parsed JSON dict, or ``{}`` on failure.
    """
    # ── String: delegate to LLMCaller ────────────────────────────
    if isinstance(model, str):
        from agentomatic.optimize.llm_caller import LLMCaller

        return await LLMCaller.call_with_json(
            model,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )

    # ── Callable / LangChain: call + parse ───────────────────────
    import json
    import re

    from loguru import logger

    json_instruction = (
        "\n\nIMPORTANT: Reply with ONLY a valid JSON object. "
        "Do not include any other text, explanation, or markdown formatting."
    )
    augmented = prompt + json_instruction
    fence_re = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)

    last_error = ""
    for attempt in range(1 + max_retries):
        raw = await call_llm(
            model,
            augmented,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        if not raw:
            last_error = "empty response from LLM"
            continue
        try:
            # Strip code fences
            m = fence_re.search(raw)
            text = m.group(1).strip() if m else raw
            # Find JSON delimiters
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object delimiters found")
            return json.loads(text[start : end + 1])  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            logger.debug(f"JSON parse attempt {attempt + 1} failed: {exc}")

    logger.warning(f"call_llm_json failed after {1 + max_retries} attempts: {last_error}")
    return {}


# =====================================================================
# LangChain helpers (private)
# =====================================================================


async def _call_langchain_async(
    model: Any,
    prompt: str,
    system_prompt: str | None,
) -> str:
    """Call a LangChain-protocol model via ``ainvoke``."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
    except ImportError:
        # Fallback: create dicts if langchain-core isn't installed
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

    result = await model.ainvoke(messages)
    return result.content if hasattr(result, "content") else str(result)


def _call_langchain_sync(
    model: Any,
    prompt: str,
    system_prompt: str | None,
) -> str:
    """Call a LangChain-protocol model via ``invoke`` (sync)."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
    except ImportError:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

    result = model.invoke(messages)
    return result.content if hasattr(result, "content") else str(result)
