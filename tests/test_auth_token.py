# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
"""Tests for :mod:`agentomatic.providers.auth_token`."""

from __future__ import annotations

import httpx
import pytest

from agentomatic.providers.auth_token import (
    OAuth2ClientCredentialsTokenProvider,
    StaticTokenProvider,
    TokenResponseError,
)
from agentomatic.providers.retry import RetryConfig


def test_static_token_provider_returns_fixed_token():
    provider = StaticTokenProvider("sk-fixed")
    assert provider.get_token() == "sk-fixed"
    assert provider.get_token() == "sk-fixed"


@pytest.mark.parametrize("token", ["", "   "])
def test_static_token_provider_rejects_blank_tokens(token):
    with pytest.raises(ValueError, match="non-empty"):
        StaticTokenProvider(token)


async def test_static_token_provider_supports_async_callers():
    provider = StaticTokenProvider("fixed")
    assert await provider.aget_token() == "fixed"


def _response(status_code: int, json_body: dict) -> httpx.Response:
    return httpx.Response(
        status_code, json=json_body, request=httpx.Request("POST", "https://x/token")
    )


def test_oauth2_fetches_and_caches_token(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, *, data, verify, timeout):
        calls["n"] += 1
        assert data["grant_type"] == "client_credentials"
        assert data["client_id"] == "cid"
        return _response(200, {"access_token": "tok-1", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/as/token.oauth2",
        client_id="cid",
        client_secret="secret",
    )
    assert provider.get_token() == "tok-1"
    assert provider.get_token() == "tok-1"  # cached, no second fetch
    assert calls["n"] == 1


def test_oauth2_refreshes_after_expiry(monkeypatch):
    tokens = iter(["tok-1", "tok-2"])

    def fake_post(url, *, data, verify, timeout):
        return _response(200, {"access_token": next(tokens), "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/as/token.oauth2",
        client_id="cid",
        client_secret="secret",
        refresh_margin_seconds=0,
    )
    assert provider.get_token() == "tok-1"

    # Force expiry without sleeping in the test.
    provider._expires_at = 0.0  # noqa: SLF001 - white-box test of internal cache state
    assert provider.get_token() == "tok-2"


def test_oauth2_invalidate_forces_refetch(monkeypatch):
    tokens = iter(["tok-1", "tok-2"])
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _response(200, {"access_token": next(tokens), "expires_in": 3600}),
    )
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/as/token.oauth2", client_id="cid", client_secret="s"
    )
    assert provider.get_token() == "tok-1"
    provider.invalidate()
    assert provider.get_token() == "tok-2"


def test_oauth2_missing_access_token_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _response(200, {"foo": "bar"}))
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/as/token.oauth2",
        client_id="cid",
        client_secret="s",
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(ValueError, match="missing.*'access_token'"):
        provider.get_token()


def test_oauth2_retries_transient_failures_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def fake_post(url, *, data, verify, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("connection refused")
        return _response(200, {"access_token": "tok", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", lambda _: None)

    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/as/token.oauth2",
        client_id="cid",
        client_secret="s",
        retry=RetryConfig(max_attempts=5, base_delay=0.01, jitter=0),
    )
    assert provider.get_token() == "tok"
    assert attempts["n"] == 3


def test_oauth2_does_not_retry_on_4xx(monkeypatch):
    attempts = {"n": 0}

    def fake_post(url, *, data, verify, timeout):
        attempts["n"] += 1
        return _response(401, {"error": "invalid_client"})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("agentomatic.providers.retry.time.sleep", lambda _: None)

    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/as/token.oauth2",
        client_id="cid",
        client_secret="s",
        retry=RetryConfig(max_attempts=5, base_delay=0.01, jitter=0),
    )
    with pytest.raises(httpx.HTTPStatusError):
        provider.get_token()
    assert attempts["n"] == 1  # 401 is not retryable


def test_oauth2_thread_safety_single_fetch(monkeypatch):
    """Concurrent get_token() calls must only trigger one HTTP fetch."""
    import threading
    import time as time_module

    calls = {"n": 0}
    lock = threading.Lock()

    def fake_post(url, *, data, verify, timeout):
        with lock:
            calls["n"] += 1
        time_module.sleep(0.05)
        return _response(200, {"access_token": "tok", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)

    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/as/token.oauth2", client_id="cid", client_secret="s"
    )
    threads = [threading.Thread(target=provider.get_token) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1


async def test_oauth2_async_api_uses_the_same_cache(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, *, data, verify, timeout):
        calls["n"] += 1
        return _response(200, {"access_token": "async-token", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/token",
        client_id="cid",
        client_secret="secret",
    )
    assert await provider.aget_token() == "async-token"
    assert provider.get_token() == "async-token"
    assert calls["n"] == 1


def test_oauth2_supports_basic_client_auth_and_custom_token_params(monkeypatch):
    captured = {}

    def fake_post(url, *, data, verify, timeout, auth):
        captured.update(data=data, auth=auth)
        return _response(200, {"access_token": "tok", "expires_in": 3600})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/token",
        client_id="cid",
        client_secret="secret",
        client_auth_method="client_secret_basic",
        token_params={"audience": "https://api.example.com"},
    )
    assert provider.get_token() == "tok"
    assert captured["data"] == {
        "grant_type": "client_credentials",
        "audience": "https://api.example.com",
    }
    assert isinstance(captured["auth"], httpx.BasicAuth)


@pytest.mark.parametrize("reserved", ["client_id", "client_secret", "grant_type", "scope"])
def test_oauth2_custom_params_cannot_override_reserved_fields(reserved):
    with pytest.raises(ValueError, match="cannot override"):
        OAuth2ClientCredentialsTokenProvider(
            token_url="https://idp.example.com/token",
            client_id="cid",
            client_secret="secret",
            token_params={reserved: "override"},
        )


def test_oauth2_rejects_insecure_non_loopback_token_url():
    with pytest.raises(ValueError, match="HTTPS"):
        OAuth2ClientCredentialsTokenProvider(
            token_url="http://idp.example.com/token",
            client_id="cid",
            client_secret="secret",
        )


def test_oauth2_allows_loopback_http_for_local_development(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _response(200, {"access_token": "local", "expires_in": 60}),
    )
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="http://127.0.0.1/token",
        client_id="cid",
        client_secret="secret",
    )
    assert provider.get_token() == "local"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"client_id": ""}, "client_id"),
        ({"client_secret": ""}, "client_secret"),
        ({"timeout": 0}, "timeout"),
        ({"refresh_margin_seconds": -1}, "refresh_margin_seconds"),
        ({"default_expires_in": float("inf")}, "default_expires_in"),
        ({"client_auth_method": "private_key_jwt"}, "client_auth_method"),
    ],
)
def test_oauth2_invalid_configuration_fails_fast(kwargs, message):
    base = {
        "token_url": "https://idp.example.com/token",
        "client_id": "cid",
        "client_secret": "secret",
    }
    with pytest.raises(ValueError, match=message):
        OAuth2ClientCredentialsTokenProvider(**{**base, **kwargs})


def test_oauth2_short_lived_token_is_still_cached(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, *, data, verify, timeout):
        calls["n"] += 1
        return _response(200, {"access_token": "short", "expires_in": 10})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/token",
        client_id="cid",
        client_secret="secret",
        refresh_margin_seconds=30,
    )
    assert provider.get_token() == "short"
    assert provider.get_token() == "short"
    assert calls["n"] == 1


@pytest.mark.parametrize("expires_in", [0, -1, "invalid", "inf", None])
def test_oauth2_rejects_invalid_expiry_without_caching(monkeypatch, expires_in):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _response(
            200,
            {"access_token": "secret-token", "expires_in": expires_in},
        ),
    )
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/token",
        client_id="cid",
        client_secret="secret",
    )
    with pytest.raises(TokenResponseError, match="expires_in"):
        provider.get_token()


def test_oauth2_invalid_response_errors_never_echo_response_body(monkeypatch):
    sensitive = "do-not-leak-this-response-value"
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _response(200, {"error_description": sensitive}),
    )
    provider = OAuth2ClientCredentialsTokenProvider(
        token_url="https://idp.example.com/token",
        client_id="cid",
        client_secret="secret",
    )
    with pytest.raises(TokenResponseError) as exc_info:
        provider.get_token()
    assert sensitive not in str(exc_info.value)
