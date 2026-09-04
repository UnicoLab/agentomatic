"""Generic synchronous retry-with-backoff helper.

Small, dependency-free (stdlib only) retry primitive shared by anything that
needs "try, back off exponentially with jitter, try again" semantics —
token fetches, health checks, custom provider HTTP calls, etc. Intentionally
separate from :func:`agentomatic.providers.llm.invoke_with_retry`, which is
async and LLM-response-shaped (thinking/metadata/telemetry); this one is a
plain sync callable wrapper usable anywhere.

Example::

    from agentomatic.providers.retry import RetryConfig, retry_call

    def fetch() -> str:
        return httpx.get(url).raise_for_status().text

    result = retry_call(fetch, config=RetryConfig(max_attempts=3))
"""

from __future__ import annotations

import asyncio
import inspect
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from loguru import logger

T = TypeVar("T")

#: Default set of exceptions considered transient/worth retrying.
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Exponential backoff configuration.

    Attributes:
        max_attempts: Total attempts including the first (``1`` = no retry).
        base_delay: Delay (seconds) before the first retry.
        max_delay: Upper bound on any single delay.
        multiplier: Backoff growth factor per attempt.
        jitter: Fraction of the computed delay to randomise (``0.2`` = ±20%),
            avoiding thundering-herd retries across concurrent callers.
        retryable_exceptions: Exception types that trigger a retry; any other
            exception propagates immediately.
        retry_on: Optional predicate ``(exception) -> bool`` for finer-grained
            control (e.g. only retry HTTP 429/5xx). When set, takes priority
            over ``retryable_exceptions`` for the decision (but the exception
            still must be an instance of ``retryable_exceptions`` first).
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.2
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: DEFAULT_RETRYABLE_EXCEPTIONS
    )
    retry_on: Callable[[Exception], bool] | None = None

    def __post_init__(self) -> None:
        """Reject invalid policies before the first production request."""
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be an integer >= 1")
        for name, value in (
            ("base_delay", self.base_delay),
            ("max_delay", self.max_delay),
            ("multiplier", self.multiplier),
            ("jitter", self.jitter),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")
        if not isinstance(self.retryable_exceptions, tuple):
            raise TypeError("retryable_exceptions must be a tuple of Exception classes")
        if any(
            not isinstance(exc_type, type) or not issubclass(exc_type, Exception)
            for exc_type in self.retryable_exceptions
        ):
            raise TypeError("retryable_exceptions must contain only Exception classes")
        if self.retry_on is not None and not callable(self.retry_on):
            raise TypeError("retry_on must be callable")

    def delay_for(self, attempt: int) -> float:
        """Compute the backoff delay (seconds) before *attempt* (1-indexed retry)."""
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError("attempt must be an integer >= 1")
        if self.base_delay == 0 or self.max_delay == 0:
            return 0.0
        try:
            raw = min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        except OverflowError:
            raw = self.max_delay
        if self.jitter <= 0:
            return raw
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread))  # noqa: S311 - jitter, not crypto

    def should_retry(self, exc: Exception) -> bool:
        """Decide whether *exc* warrants a retry."""
        if not isinstance(exc, self.retryable_exceptions):
            return False
        if self.retry_on is not None:
            return self.retry_on(exc)
        return True


def retry_call(
    fn: Callable[[], T],
    *,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Call ``fn()`` with exponential backoff, retrying on transient errors.

    Args:
        fn: Zero-argument callable to invoke.
        config: Retry policy; defaults to :class:`RetryConfig` defaults.
        on_retry: Optional hook ``(attempt, exception, delay) -> None`` fired
            before each sleep (attempt is 1-indexed, the attempt that failed).

    Returns:
        ``fn()``'s return value on success.

    Raises:
        The last exception raised by ``fn`` once attempts are exhausted, or
        immediately if it isn't retryable.
    """
    cfg = config or RetryConfig()
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retried
            if attempt >= cfg.max_attempts or not cfg.should_retry(exc):
                raise
            delay = cfg.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            else:
                logger.debug(
                    "retry_call: attempt {}/{} failed ({}); retrying in {:.2f}s",
                    attempt,
                    cfg.max_attempts,
                    type(exc).__name__,
                    delay,
                )
            time.sleep(delay)

    raise RuntimeError("retry_call reached an unreachable state")  # pragma: no cover


async def async_retry_call(
    fn: Callable[[], Awaitable[T]],
    *,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None | Awaitable[None]] | None = None,
) -> T:
    """Await ``fn()`` with cancellation-safe exponential backoff.

    ``asyncio.CancelledError`` is deliberately never caught, so task
    cancellation immediately stops retries and pending sleeps.

    Args:
        fn: Zero-argument async callable to invoke.
        config: Retry policy; defaults to :class:`RetryConfig` defaults.
        on_retry: Optional sync or async hook fired before each sleep.

    Returns:
        The awaited callable's return value on success.

    Raises:
        The last exception raised by ``fn`` once attempts are exhausted, or
        immediately if it is not retryable.
    """
    cfg = config or RetryConfig()
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retried
            if attempt >= cfg.max_attempts or not cfg.should_retry(exc):
                raise
            delay = cfg.delay_for(attempt)
            if on_retry is not None:
                hook_result = on_retry(attempt, exc, delay)
                if inspect.isawaitable(hook_result):
                    await hook_result
            else:
                logger.debug(
                    "async_retry_call: attempt {}/{} failed ({}); retrying in {:.2f}s",
                    attempt,
                    cfg.max_attempts,
                    type(exc).__name__,
                    delay,
                )
            await asyncio.sleep(delay)

    raise RuntimeError("async_retry_call reached an unreachable state")  # pragma: no cover
