"""Ordered LLM fallback chain with configurable trigger conditions.

Wraps a primary chat model and zero or more fallback models. When the
primary fails (or returns an empty response), the next model in the chain
is tried. Successful fallbacks are logged with the model label that
answered.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

DEFAULT_FALLBACK_ON: tuple[str, ...] = (
    "timeout",
    "connection",
    "rate_limit",
    "empty_response",
)


class EmptyLLMResponseError(RuntimeError):
    """Raised when an LLM returns an empty or whitespace-only response."""


def normalize_fallback_on(fallback_on: list[str] | tuple[str, ...] | None) -> frozenset[str]:
    """Normalize configured fallback trigger names.

    Args:
        fallback_on: Trigger names, or ``None`` for the library defaults.

    Returns:
        A frozenset of lower-cased trigger names.
    """
    if fallback_on is None:
        return frozenset(DEFAULT_FALLBACK_ON)
    return frozenset(str(item).strip().lower() for item in fallback_on if str(item).strip())


def is_empty_llm_response(result: Any) -> bool:
    """Return ``True`` when *result* has no usable text content.

    Args:
        result: Raw LLM invoke/ainvoke result.

    Returns:
        Whether the response should be treated as empty.
    """
    from agentomatic.providers.message_utils import message_text

    text: str
    try:
        text = message_text(result)
    except Exception:  # noqa: BLE001
        content = getattr(result, "content", None)
        if content is None:
            text = str(result) if result is not None else ""
        else:
            text = str(content)
    return not str(text or "").strip()


def should_fallback(exc: BaseException, fallback_on: frozenset[str]) -> bool:
    """Decide whether *exc* should trigger the next fallback model.

    Args:
        exc: Exception raised by the current model attempt.
        fallback_on: Configured trigger set.

    Returns:
        ``True`` when the chain should advance to the next model.
    """
    if not fallback_on:
        return False
    if "any_error" in fallback_on or "any" in fallback_on or "exception" in fallback_on:
        return True
    if "empty_response" in fallback_on and isinstance(exc, EmptyLLMResponseError):
        return True

    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    combined = f"{name} {msg}"

    if "timeout" in fallback_on and (
        isinstance(exc, TimeoutError)
        or "timeout" in combined
        or "timed out" in combined
        or "deadline" in combined
    ):
        return True

    if "connection" in fallback_on and (
        isinstance(exc, (ConnectionError, OSError))
        or "connection" in combined
        or "connect" in combined
        or "unreachable" in combined
        or "name or service not known" in combined
        or "nodename nor servname" in combined
    ):
        return True

    if "rate_limit" in fallback_on and (
        "ratelimit" in name.replace("_", "")
        or "rate limit" in combined
        or "rate_limit" in combined
        or "too many requests" in combined
        or "429" in combined
    ):
        return True

    return False


def model_label(provider: str, model: str | None = None) -> str:
    """Build a human-readable ``provider/model`` label.

    Args:
        provider: Provider identifier.
        model: Optional model name.

    Returns:
        Label used in logs and telemetry.
    """
    provider = (provider or "unknown").strip() or "unknown"
    if model:
        return f"{provider}/{model}"
    return provider


class FallbackLLM:
    """Chat-model wrapper that retries with ordered fallback models.

    The wrapper exposes ``invoke`` / ``ainvoke`` / ``astream`` and a
    ``fallbacks`` attribute (list of backup models) for compatibility with
    LangChain's ``RunnableWithFallbacks`` shape used in existing tests.

    Args:
        primary: Primary chat model / runnable.
        fallbacks: Ordered backup models.
        labels: Labels aligned with ``[primary, *fallbacks]``.
        fallback_on: Trigger conditions (see :data:`DEFAULT_FALLBACK_ON`).
    """

    def __init__(
        self,
        primary: Any,
        fallbacks: list[Any] | None = None,
        *,
        labels: list[str] | None = None,
        fallback_on: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.primary = primary
        self.fallbacks: list[Any] = list(fallbacks or [])
        self._chain: list[Any] = [primary, *self.fallbacks]
        if labels is None:
            self._labels = [f"model-{idx}" for idx in range(len(self._chain))]
        else:
            self._labels = list(labels)
            while len(self._labels) < len(self._chain):
                self._labels.append(f"model-{len(self._labels)}")
        self._fallback_on = normalize_fallback_on(fallback_on)

    @property
    def fallback_on(self) -> frozenset[str]:
        """Configured fallback trigger names."""
        return self._fallback_on

    @property
    def labels(self) -> list[str]:
        """Labels for each model in the chain (primary first)."""
        return list(self._labels)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronously invoke the fallback chain.

        Args:
            *args: Forwarded to the underlying model.
            **kwargs: Forwarded to the underlying model.

        Returns:
            The first successful non-empty model response.

        Raises:
            Exception: The last error when every model fails.
        """
        return self._run_sync(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously invoke the fallback chain.

        Args:
            *args: Forwarded to the underlying model.
            **kwargs: Forwarded to the underlying model.

        Returns:
            The first successful non-empty model response.

        Raises:
            Exception: The last error when every model fails.
        """
        return await self._run_async(*args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        """Stream from the first model that starts successfully.

        Args:
            *args: Forwarded to the underlying model.
            **kwargs: Forwarded to the underlying model.

        Yields:
            Stream chunks from the successful model.
        """
        last_exc: BaseException | None = None
        for idx, llm in enumerate(self._chain):
            label = self._labels[idx]
            try:
                if not hasattr(llm, "astream"):
                    result = await self._invoke_one_async(llm, label, *args, **kwargs)
                    yield result
                    return
                stream = llm.astream(*args, **kwargs)
                first = True
                async for chunk in stream:
                    if first and idx > 0:
                        self._log_failover_success(idx, reason="stream_start")
                    first = False
                    yield chunk
                if first:
                    # Empty stream — treat like empty response when configured.
                    if "empty_response" in self._fallback_on:
                        raise EmptyLLMResponseError(f"Empty stream from {label}")
                    return
                if idx == 0:
                    logger.debug("LLM primary succeeded (stream): {}", label)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if idx >= len(self._chain) - 1 or not should_fallback(exc, self._fallback_on):
                    raise
                self._record_step_failure(idx, exc)

        if last_exc is not None:
            raise last_exc

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the primary model."""
        return getattr(self.primary, name)

    def _run_sync(self, *args: Any, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        for idx, llm in enumerate(self._chain):
            label = self._labels[idx]
            try:
                result = self._invoke_one_sync(llm, label, *args, **kwargs)
                if idx > 0:
                    self._log_failover_success(idx, reason=str(last_exc or "fallback"))
                else:
                    logger.debug("LLM primary succeeded: {}", label)
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if idx >= len(self._chain) - 1 or not should_fallback(exc, self._fallback_on):
                    raise
                self._record_step_failure(idx, exc)
        assert last_exc is not None
        raise last_exc

    async def _run_async(self, *args: Any, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        for idx, llm in enumerate(self._chain):
            label = self._labels[idx]
            try:
                result = await self._invoke_one_async(llm, label, *args, **kwargs)
                if idx > 0:
                    self._log_failover_success(idx, reason=str(last_exc or "fallback"))
                else:
                    logger.debug("LLM primary succeeded: {}", label)
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if idx >= len(self._chain) - 1 or not should_fallback(exc, self._fallback_on):
                    raise
                self._record_step_failure(idx, exc)
        assert last_exc is not None
        raise last_exc

    def _invoke_one_sync(self, llm: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        if callable(llm) and not hasattr(llm, "invoke"):
            result = llm(*args, **kwargs)
        else:
            result = llm.invoke(*args, **kwargs)
        return self._ensure_non_empty(result, label)

    async def _invoke_one_async(self, llm: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        if hasattr(llm, "ainvoke"):
            result = await llm.ainvoke(*args, **kwargs)
        elif callable(llm):
            result = llm(*args, **kwargs)
            if hasattr(result, "__await__"):
                result = await result
        else:
            result = llm.invoke(*args, **kwargs)
        return self._ensure_non_empty(result, label)

    def _ensure_non_empty(self, result: Any, label: str) -> Any:
        if "empty_response" in self._fallback_on and is_empty_llm_response(result):
            raise EmptyLLMResponseError(f"Empty response from {label}")
        return result

    def _record_step_failure(self, idx: int, exc: BaseException) -> None:
        from agentomatic.providers.llm import record_failover

        primary_label = self._labels[idx]
        next_label = self._labels[idx + 1]
        record_failover(primary_label, next_label, f"{type(exc).__name__}: {exc}")

    def _log_failover_success(self, idx: int, *, reason: str) -> None:
        logger.info(
            "LLM fallback succeeded with {} (after failure on {}); reason={}",
            self._labels[idx],
            self._labels[idx - 1],
            reason,
        )
