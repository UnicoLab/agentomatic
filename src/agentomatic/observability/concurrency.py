"""Circuit breaker and concurrency control."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from loguru import logger


class CircuitState(Enum):
    """States for the circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""


class CircuitBreaker:
    """Async circuit breaker for protecting external services."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, auto-transitioning from OPEN to HALF_OPEN after timeout."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @asynccontextmanager
    async def __call__(self):
        """Use as an async context manager around protected calls."""
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpen(f"Circuit breaker '{self.name}' is open")

        try:
            yield
            self._on_success()
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Record a failed call and potentially open the circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' opened after {self._failure_count} failures")


class AgentSemaphore:
    """Global semaphore to limit concurrent agent invocations."""

    _instance: AgentSemaphore | None = None

    def __init__(self, max_concurrent: int = 10) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent
        self._active = 0

    @classmethod
    def get(cls, max_concurrent: int = 10) -> AgentSemaphore:
        """Get or create the singleton semaphore."""
        if cls._instance is None:
            cls._instance = cls(max_concurrent)
        return cls._instance

    @asynccontextmanager
    async def acquire(self):
        """Acquire the semaphore, limiting concurrency."""
        await self._semaphore.acquire()
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            self._semaphore.release()

    @property
    def active(self) -> int:
        """Number of currently active acquisitions."""
        return self._active
