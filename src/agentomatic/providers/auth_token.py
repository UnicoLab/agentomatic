"""Generic bearer-token providers for custom LLM/embedding backends.

Many internal "AI gateway" hubs (Azure-style model hubs, SecureGPT-like
proxies, etc.) sit behind OAuth2 client-credentials auth instead of a plain
static API key: you exchange a client id/secret for a short-lived bearer
token and must transparently refresh it before it expires.

This module provides small, dependency-light (``httpx`` only) building
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
from collections.abc import Mapping
from math import isfinite
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from loguru import logger

from agentomatic.providers.retry import RetryConfig, retry_call


class TokenProvider(Protocol):
    """Common interface for anything that can hand back a bearer token."""

    def get_token(self) -> str:
        """Return a currently-valid bearer token."""
        ...


class AsyncTokenProvider(Protocol):
    """Common interface for asynchronously obtaining a bearer token."""

    async def aget_token(self) -> str:
        """Return a currently-valid bearer token without blocking the event loop."""
        ...


class TokenResponseError(ValueError):
    """Raised when a successful token response has an invalid, safe-to-report shape."""


class StaticTokenProvider:
    """Trivial :class:`TokenProvider` wrapping a fixed, pre-issued API key.

    Use this when the backend accepts a long-lived key directly as the
    bearer token (no OAuth2 exchange), so callers can share the same
    ``get_token()`` interface as :class:`OAuth2ClientCredentialsTokenProvider`.
    """

    def __init__(self, token: str) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("token must be a non-empty string")
        self._token = token

    def get_token(self) -> str:
        """Return the static token."""
        return self._token

    async def aget_token(self) -> str:
        """Return the static token from asynchronous call sites."""
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
        client_auth_method: Send credentials in the form body
            (``client_secret_post``) or HTTP Basic auth
            (``client_secret_basic``).
        token_params: Additional form fields such as ``audience`` or
            ``resource``. OAuth-reserved fields cannot be overridden.
        default_expires_in: Cache lifetime used when the server omits
            ``expires_in``.
        allow_insecure_http: Permit a non-loopback ``http://`` token URL.
            Disabled by default to protect client credentials in transit.
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
        client_auth_method: Literal["client_secret_post", "client_secret_basic"] = (
            "client_secret_post"
        ),
        token_params: Mapping[str, str] | None = None,
        default_expires_in: float = 3600.0,
        allow_insecure_http: bool = False,
    ) -> None:
        self._validate_configuration(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            refresh_margin_seconds=refresh_margin_seconds,
            timeout=timeout,
            default_expires_in=default_expires_in,
            client_auth_method=client_auth_method,
            allow_insecure_http=allow_insecure_http,
        )
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._refresh_margin = refresh_margin_seconds
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._retry = retry or RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0)
        self._client_auth_method = client_auth_method
        self._token_params = dict(token_params or {})
        self._default_expires_in = default_expires_in
        reserved = {"client_id", "client_secret", "grant_type", "scope"}
        overlap = reserved.intersection(self._token_params)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"token_params cannot override OAuth fields: {names}")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self._token_params.items()
        ):
            raise TypeError("token_params keys and values must be strings")

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

    async def aget_token(self) -> str:
        """Return a valid token without blocking an asynchronous event loop."""
        import asyncio

        return await asyncio.to_thread(self.get_token)

    def invalidate(self) -> None:
        """Force the next :meth:`get_token` call to fetch a fresh token."""
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def _fetch_token_locked(self) -> None:
        """Perform the OAuth2 token request with retry/backoff. Caller holds ``_lock``."""
        import httpx

        payload = {"grant_type": "client_credentials", **self._token_params}
        request_kwargs: dict[str, Any] = {
            "data": payload,
            "verify": self._verify_ssl,
            "timeout": self._timeout,
        }
        if self._client_auth_method == "client_secret_basic":
            request_kwargs["auth"] = httpx.BasicAuth(self._client_id, self._client_secret)
        else:
            payload["client_id"] = self._client_id
            payload["client_secret"] = self._client_secret
        if self._scope:
            payload["scope"] = self._scope

        def _request() -> httpx.Response:
            response = httpx.post(self._token_url, **request_kwargs)
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
        try:
            body = response.json()
        except ValueError as exc:
            raise TokenResponseError("OAuth2 token response is not valid JSON") from exc
        if not isinstance(body, dict):
            raise TokenResponseError("OAuth2 token response must be a JSON object")

        token = body.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise TokenResponseError("OAuth2 token response missing a non-empty 'access_token'")

        expires_in_raw = body.get("expires_in", self._default_expires_in)
        try:
            expires_in = float(expires_in_raw)
        except (TypeError, ValueError) as exc:
            raise TokenResponseError("OAuth2 token response has an invalid 'expires_in'") from exc
        if not isfinite(expires_in) or expires_in <= 0:
            raise TokenResponseError(
                "OAuth2 token response 'expires_in' must be positive and finite"
            )
        effective_margin = min(self._refresh_margin, expires_in / 2)
        self._token = token
        self._expires_at = time.monotonic() + expires_in - effective_margin
        logger.debug(
            "OAuth2 token refreshed (expires_in={}s, refresh_margin={}s)",
            expires_in,
            effective_margin,
        )

    @staticmethod
    def _validate_configuration(
        *,
        token_url: str,
        client_id: str,
        client_secret: str,
        refresh_margin_seconds: float,
        timeout: float,
        default_expires_in: float,
        client_auth_method: str,
        allow_insecure_http: bool,
    ) -> None:
        """Validate security-sensitive configuration eagerly."""
        for credential_name, credential_value in (
            ("client_id", client_id),
            ("client_secret", client_secret),
        ):
            if not isinstance(credential_value, str) or not credential_value.strip():
                raise ValueError(f"{credential_name} must be a non-empty string")
        if not isinstance(token_url, str) or not token_url.strip():
            raise ValueError("token_url must be a non-empty URL")
        parsed = urlsplit(token_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("token_url must be an absolute HTTP(S) URL")
        loopback_hosts = {"localhost", "127.0.0.1", "::1"}
        if (
            parsed.scheme != "https"
            and parsed.hostname not in loopback_hosts
            and not allow_insecure_http
        ):
            raise ValueError(
                "token_url must use HTTPS outside loopback; set allow_insecure_http=True "
                "only for controlled development environments"
            )
        for numeric_name, numeric_value, allow_zero in (
            ("refresh_margin_seconds", refresh_margin_seconds, True),
            ("timeout", timeout, False),
            ("default_expires_in", default_expires_in, False),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, (int, float))
                or not isfinite(numeric_value)
            ):
                raise ValueError(f"{numeric_name} must be a finite number")
            if numeric_value < 0 or (not allow_zero and numeric_value == 0):
                comparator = ">= 0" if allow_zero else "> 0"
                raise ValueError(f"{numeric_name} must be {comparator}")
        if client_auth_method not in {"client_secret_post", "client_secret_basic"}:
            raise ValueError(
                "client_auth_method must be 'client_secret_post' or 'client_secret_basic'"
            )
