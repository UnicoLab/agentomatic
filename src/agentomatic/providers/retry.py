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

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

#: Default set of exceptions considered transient/worth retrying.
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass
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
    retryable_exceptions: tuple[type[BaseException], ...] = field(
        default_factory=lambda: DEFAULT_RETRYABLE_EXCEPTIONS
    )
    retry_on: Callable[[BaseException], bool] | None = None

    def delay_for(self, attempt: int) -> float:
        """Compute the backoff delay (seconds) before *attempt* (1-indexed retry)."""
        raw = min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        if self.jitter <= 0:
            return raw
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread))  # noqa: S311 - jitter, not crypto

    def should_retry(self, exc: BaseException) -> bool:
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
    on_retry: Callable[[int, BaseException, float], None] | None = None,
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
    last_exc: BaseException | None = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below when not retried
            last_exc = exc
            if attempt >= cfg.max_attempts or not cfg.should_retry(exc):
                raise
            delay = cfg.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            else:
                logger.debug(
                    f"retry_call: attempt {attempt}/{cfg.max_attempts} failed "
                    f"({type(exc).__name__}: {exc}); retrying in {delay:.2f}s"
                )
            time.sleep(delay)

    # Unreachable in practice (loop always returns or raises), but keeps
    # type-checkers happy and guards against a zero-attempt misconfiguration.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_call: max_attempts must be >= 1")
