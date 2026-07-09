"""Upstream authentication for custom endpoints.

Resolves credentials (with ``${ENV}`` interpolation) into request headers
and implements the OAuth2 *client credentials* grant with token caching.
"""

from __future__ import annotations

import base64
import os
import re
import time
from typing import TYPE_CHECKING

from loguru import logger

from agentomatic.endpoints.models import AuthType, UpstreamAuthConfig

if TYPE_CHECKING:
    import httpx

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_env(value: str) -> str:
    """Interpolate ``${VAR}`` references from the environment.

    Args:
        value: A string that may contain ``${VAR}`` placeholders.

    Returns:
        The string with all placeholders replaced by their environment
        values (missing variables resolve to an empty string).
    """
    if not value or "${" not in value:
        return value

    def _replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_PATTERN.sub(_replace, value)


class UpstreamAuthenticator:
    """Compute authentication headers for an upstream service.

    A single instance is bound to one :class:`UpstreamAuthConfig`. For the
    OAuth2 client-credentials grant it caches the access token and refreshes
    it automatically shortly before expiry (controlled by ``token_leeway``).
    """

    def __init__(self, config: UpstreamAuthConfig) -> None:
        self._config = config
        self._token: str = ""
        self._token_expiry: float = 0.0

    @property
    def scheme(self) -> AuthType:
        """Return the configured authentication scheme."""
        return self._config.type

    async def headers(self, client: httpx.AsyncClient) -> dict[str, str]:
        """Return the auth headers to attach to an upstream request.

        Args:
            client: An ``httpx.AsyncClient`` used to fetch OAuth2 tokens
                when required.

        Returns:
            A mapping of header names to values (may be empty).
        """
        cfg = self._config
        if cfg.type == AuthType.NONE:
            return {}

        if cfg.type == AuthType.API_KEY:
            key = resolve_env(cfg.api_key)
            if not key:
                return {}
            prefix = f"{cfg.header_prefix} " if cfg.header_prefix else ""
            return {cfg.header_name: f"{prefix}{key}"}

        if cfg.type == AuthType.BEARER:
            token = resolve_env(cfg.api_key)
            return {cfg.header_name: f"Bearer {token}"} if token else {}

        if cfg.type == AuthType.BASIC:
            user = resolve_env(cfg.username)
            pwd = resolve_env(cfg.password)
            raw = f"{user}:{pwd}".encode()
            encoded = base64.b64encode(raw).decode("ascii")
            return {"Authorization": f"Basic {encoded}"}

        if cfg.type == AuthType.OAUTH2_CLIENT_CREDENTIALS:
            token = await self._get_oauth2_token(client)
            return {"Authorization": f"Bearer {token}"} if token else {}

        return {}

    async def _get_oauth2_token(self, client: httpx.AsyncClient) -> str:
        """Fetch (or reuse a cached) OAuth2 client-credentials token."""
        cfg = self._config
        now = time.time()
        if self._token and now < (self._token_expiry - cfg.token_leeway):
            return self._token

        token_url = resolve_env(cfg.token_url)
        if not token_url:
            logger.warning("OAuth2 auth configured without a token_url — skipping")
            return ""

        data = {
            "grant_type": "client_credentials",
            "client_id": resolve_env(cfg.client_id),
            "client_secret": resolve_env(cfg.client_secret),
        }
        if cfg.scope:
            data["scope"] = cfg.scope
        if cfg.audience:
            data["audience"] = resolve_env(cfg.audience)

        try:
            resp = await client.post(token_url, data=data)
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"OAuth2 token request to {token_url} failed: {exc}")
            return ""

        self._token = str(body.get("access_token", ""))
        expires_in = float(body.get("expires_in", 3600))
        self._token_expiry = now + expires_in
        logger.debug(f"Fetched OAuth2 token (expires in {expires_in:.0f}s)")
        return self._token
