# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
"""Tests for :mod:`agentomatic.providers.retry` (generic backoff helper)."""

from __future__ import annotations

import asyncio

import pytest

from agentomatic.providers.retry import RetryConfig, async_retry_call, retry_call


def test_succeeds_first_try_no_sleep(monkeypatch):
    monkeypatch.setattr(
        "agentomatic.providers.retry.time.sleep",
        lambda _: (_ for _ in ()).throw(AssertionError("should not sleep")),
    )
    assert retry_call(lambda: 42) == 42


def test_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", sleeps.append)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    result = retry_call(flaky, config=RetryConfig(max_attempts=5, base_delay=0.01, jitter=0))
    assert result == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_exhausts_attempts_and_raises(monkeypatch):
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", lambda _: None)

    def always_fails():
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        retry_call(always_fails, config=RetryConfig(max_attempts=3, base_delay=0.01, jitter=0))


def test_non_retryable_exception_propagates_immediately(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", sleeps.append)

    def raises_value_error():
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        retry_call(raises_value_error, config=RetryConfig(max_attempts=5))
    assert sleeps == []


def test_retry_on_predicate_overrides_type_check(monkeypatch):
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", lambda _: None)
    calls = {"n": 0}

    def flaky_connection_error():
        calls["n"] += 1
        raise ConnectionError("only retry even attempts")

    config = RetryConfig(
        max_attempts=3,
        base_delay=0.01,
        retryable_exceptions=(ConnectionError,),
        retry_on=lambda exc: False,
    )
    with pytest.raises(ConnectionError):
        retry_call(flaky_connection_error, config=config)
    assert calls["n"] == 1  # predicate says never retry


def test_delay_for_respects_max_delay_and_multiplier():
    config = RetryConfig(base_delay=1.0, multiplier=3.0, max_delay=5.0, jitter=0)
    assert config.delay_for(1) == 1.0
    assert config.delay_for(2) == 3.0
    assert config.delay_for(3) == 5.0  # capped (would be 9.0 uncapped)


def test_on_retry_hook_invoked_with_attempt_exception_delay(monkeypatch):
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", lambda _: None)
    events: list[tuple[int, str, float]] = []

    def flaky():
        if len(events) < 2:
            raise ConnectionError("x")
        return "done"

    def on_retry(attempt, exc, delay):
        events.append((attempt, type(exc).__name__, delay))

    result = retry_call(
        flaky,
        config=RetryConfig(max_attempts=5, base_delay=0.01, jitter=0),
        on_retry=on_retry,
    )
    assert result == "done"
    assert [e[0] for e in events] == [1, 2]
    assert all(e[1] == "ConnectionError" for e in events)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": 1.5}, "max_attempts"),
        ({"base_delay": -1}, "base_delay"),
        ({"max_delay": float("inf")}, "max_delay"),
        ({"multiplier": 0.5}, "multiplier"),
        ({"jitter": 1.01}, "jitter"),
    ],
)
def test_invalid_retry_configuration_fails_fast(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RetryConfig(**kwargs)


def test_retryable_exceptions_must_be_exception_classes():
    with pytest.raises(TypeError, match="Exception classes"):
        RetryConfig(retryable_exceptions=(KeyboardInterrupt,))


def test_public_retry_type_hints_are_runtime_resolvable():
    import typing

    assert typing.get_type_hints(RetryConfig)["max_attempts"] is int
    assert "fn" in typing.get_type_hints(retry_call)
    assert "fn" in typing.get_type_hints(async_retry_call)


def test_delay_for_validates_attempt_and_handles_numeric_overflow():
    config = RetryConfig(base_delay=1.0, multiplier=10.0, max_delay=5.0, jitter=0)
    with pytest.raises(ValueError, match="attempt"):
        config.delay_for(0)
    assert config.delay_for(1_000_000) == 5.0
    assert RetryConfig(base_delay=0, multiplier=10, jitter=0).delay_for(1_000_000) == 0


async def test_async_retry_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("agentomatic.providers.retry.asyncio.sleep", fake_sleep)
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("transient")
        return "ok"

    result = await async_retry_call(
        flaky,
        config=RetryConfig(max_attempts=3, base_delay=0.25, jitter=0),
    )
    assert result == "ok"
    assert sleeps == [0.25, 0.5]


async def test_async_retry_supports_async_hook(monkeypatch):
    async def fake_sleep(delay):
        return None

    monkeypatch.setattr("agentomatic.providers.retry.asyncio.sleep", fake_sleep)
    events = []
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("transient")
        return "done"

    async def on_retry(attempt, exc, delay):
        events.append((attempt, type(exc), delay))

    assert (
        await async_retry_call(
            flaky,
            config=RetryConfig(jitter=0),
            on_retry=on_retry,
        )
        == "done"
    )
    assert events == [(1, TimeoutError, 1.0)]


async def test_async_retry_propagates_cancellation_without_retry(monkeypatch):
    sleep_called = False

    async def fake_sleep(delay):
        nonlocal sleep_called
        sleep_called = True

    monkeypatch.setattr("agentomatic.providers.retry.asyncio.sleep", fake_sleep)

    async def cancelled():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await async_retry_call(cancelled)
    assert sleep_called is False
