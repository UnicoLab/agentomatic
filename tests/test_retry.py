# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
"""Tests for :mod:`agentomatic.providers.retry` (generic backoff helper)."""

from __future__ import annotations

import pytest

from agentomatic.providers.retry import RetryConfig, retry_call


def test_succeeds_first_try_no_sleep(monkeypatch):
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", lambda _: (_ for _ in ()).throw(AssertionError("should not sleep")))
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
