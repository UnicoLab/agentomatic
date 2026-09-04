"""Generic bearer-token providers for custom LLM/embedding backends.

Many internal "AI gateway" hubs (Azure-style model hubs, SecureGPT-like
proxies, etc.) sit behind OAuth2 client-credentials auth instead of a plain
static API key: you exchange a client id/secret for a short-lived bearer
token and must transparently refresh it before it expires.

This module provides small, dependency-light (``requests`` only) building
blocks so a custom :func:`agentomatic.providers.register_llm_provider`
builder doesn't need to reimplement token caching/refresh from scratch::

    from agentomatic.providers.auth_token import OAuth2ClientCredentialsTokenProvider

    token_provider = OAuth2ClientCredentialsTokenProvider(
        token_url=f"https://{one_account}/as/token.oauth2",
        client_id=client_id,
        client_secret=client_secret,
        scope="urn:grp:chatgpt",
    )
    headers = {"Authorization": f"Bearer {token_provider.get_token()}"}

:class:`StaticTokenProvider` is the trivial counterpart for hubs that hand
out a long-lived API key used directly as the bearer token (no exchange
step) — both implement the same ``get_token() -> str`` interface so callers
can accept either without branching.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

from loguru import logger

from agentomatic.providers.retry import RetryConfig, retry_call


class TokenProvider(Protocol):
    """Common interface for anything that can hand back a bearer token."""

    def get_token(self) -> str:
        """Return a currently-valid bearer token."""
        ...


class StaticTokenProvider:
    """Trivial :class:`TokenProvider` wrapping a fixed, pre-issued API key.

    Use this when the backend accepts a long-lived key directly as the
    bearer token (no OAuth2 exchange), so callers can share the same
    ``get_token()`` interface as :class:`OAuth2ClientCredentialsTokenProvider`.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    def get_token(self) -> str:
        """Return the static token."""
        return self._token


class OAuth2ClientCredentialsTokenProvider:
    """Thread-safe OAuth2 client-credentials token fetcher with auto-renewal.

    Fetches a bearer token from *token_url* using the ``client_credentials``
    grant, caches it, and transparently re-fetches it once it is within
    *refresh_margin_seconds* of expiring. Safe to share a single instance
    across threads/requests.

    Args:
        token_url: Full OAuth2 token endpoint
            (e.g. ``https://{ONE_ACCOUNT}/as/token.oauth2``).
        client_id: OAuth2 client id.
        client_secret: OAuth2 client secret.
        scope: Optional space-delimited scope string.
        refresh_margin_seconds: Refresh this many seconds before actual
            expiry to avoid racing a token that dies mid-request.
        verify_ssl: TLS verification for the token request (keep ``True``
            outside of local/dev environments with self-signed certs).
        timeout: Timeout (seconds) for the token request itself.
        retry: Backoff policy for transient token-endpoint failures
            (connection errors, timeouts, 5xx). Defaults to 3 attempts.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        *,
        scope: str | None = None,
        refresh_margin_seconds: float = 30.0,
        verify_ssl: bool = True,
        timeout: float = 30.0,
        retry: RetryConfig | None = None,
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._refresh_margin = refresh_margin_seconds
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._retry = retry or RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0)

        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        """Return a valid bearer token, fetching/renewing it if needed."""
        with self._lock:
            if self._token is None or time.monotonic() >= self._expires_at:
                self._fetch_token_locked()
            assert self._token is not None  # noqa: S101 - set by _fetch_token_locked
            return self._token

    def invalidate(self) -> None:
        """Force the next :meth:`get_token` call to fetch a fresh token."""
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def _fetch_token_locked(self) -> None:
        """Perform the OAuth2 token request with retry/backoff. Caller holds ``_lock``."""
        import httpx

        payload = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "client_credentials",
        }
        if self._scope:
            payload["scope"] = self._scope

        def _request() -> httpx.Response:
            response = httpx.post(
                self._token_url,
                data=payload,
                verify=self._verify_ssl,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response

        retry_cfg = RetryConfig(
            max_attempts=self._retry.max_attempts,
            base_delay=self._retry.base_delay,
            max_delay=self._retry.max_delay,
            multiplier=self._retry.multiplier,
            jitter=self._retry.jitter,
            retryable_exceptions=(httpx.TransportError, httpx.HTTPStatusError),
            retry_on=lambda exc: (
                not isinstance(exc, httpx.HTTPStatusError)
                or exc.response.status_code >= 500
                or exc.response.status_code == 429
            ),
        )
        response = retry_call(_request, config=retry_cfg)
        body = response.json()

        token = body.get("access_token")
        if not token:
            raise ValueError(f"OAuth2 token response missing 'access_token': {body}")

        expires_in = float(body.get("expires_in", 3600))
        self._token = token
        self._expires_at = time.monotonic() + max(expires_in - self._refresh_margin, 0.0)
        logger.debug(f"OAuth2 token refreshed from {self._token_url} (expires_in={expires_in}s)")
