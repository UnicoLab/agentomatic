"""Authenticated HTTP service connections for agents.

Reuses the endpoint :class:`UpstreamClient` so agents can call external
services (with API-key / bearer / basic / OAuth2 auth) with minimal code.
"""

from __future__ import annotations

from typing import Any

from agentomatic.connections.models import HttpConnectionConfig
from agentomatic.endpoints.client import UpstreamClient
from agentomatic.endpoints.models import UpstreamConfig, UpstreamResult


def _observe_connection(name: str, status: str) -> None:
    """Emit a best-effort connection-call metric."""
    try:
        from agentomatic.observability.metrics import CONNECTION_CALL_COUNT

        CONNECTION_CALL_COUNT.labels(connection=name, status=status).inc()
    except Exception:  # noqa: BLE001 - metrics are optional
        pass


class HttpConnection:
    """A reusable, authenticated HTTP client bound to one external service."""

    def __init__(self, config: HttpConnectionConfig) -> None:
        self.config = config
        self._client = UpstreamClient(
            UpstreamConfig(
                name=config.name,
                base_url=config.base_url,
                headers=config.headers,
                auth=config.auth,
                timeout=config.timeout,
                max_retries=config.max_retries,
                verify_ssl=config.verify_ssl,
            )
        )

    @property
    def name(self) -> str:
        """Return the connection's logical name."""
        return self.config.name

    async def initialize(self) -> None:
        """No-op initialisation (the client is created lazily)."""

    async def request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Any = None,
        params: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        """Perform an authenticated request against the service.

        Args:
            path: Request path relative to the configured ``base_url``.
            method: HTTP method (default ``GET``).
            payload: Optional JSON body for write methods.
            params: Optional query parameters.

        Returns:
            A structured :class:`UpstreamResult`.
        """
        try:
            result = await self._client.request(
                payload,
                path=path,
                method=method,
                params=params,
            )
        except Exception:
            _observe_connection(self.name, "error")
            raise
        _observe_connection(self.name, "ok" if result.ok else "error")
        return result

    async def get(self, path: str, **kwargs: Any) -> UpstreamResult:
        """Convenience GET request."""
        return await self.request(path, method="GET", **kwargs)

    async def post(self, path: str, payload: Any = None, **kwargs: Any) -> UpstreamResult:
        """Convenience POST request."""
        return await self.request(path, method="POST", payload=payload, **kwargs)

    async def health_check(self) -> dict[str, Any]:
        """Report configuration-level health (does not make a request)."""
        return {"connection": self.name, "kind": "http", "status": "configured"}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
