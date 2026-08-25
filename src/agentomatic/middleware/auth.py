"""API key authentication middleware.

Enabled via ``FEATURES__ENABLE_AUTH=true`` and ``AUTH__API_KEY=your-key``.
Skips health/readiness probes. Supports both header and query param.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from agentomatic.middleware.pathutils import PROBE_PATHS, path_is_skipped

_SKIP_PATHS = {
    # Probe endpoints (see PROBE_PATHS) are added below: an orchestrator has
    # no credentials, so a readiness probe that 401s keeps every pod out of
    # service and the Deployment never rolls out.
    *PROBE_PATHS,
    "/docs",
    "/openapi.json",
    "/redoc",
    "/",
    # Only the static UI shell is public (like the Swagger UI shell at
    # /docs) — NOT "/studio", which (via prefix matching in
    # path_is_skipped) would also exempt the entire Studio debug REST API
    # (/studio/agents/..., /studio/.../threads/{id}/state, etc.), letting
    # an unauthenticated caller read/mutate any agent's run state.
    "/studio/ui",
    "/status",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple API-key guard.

    Args:
        app: ASGI application.
        api_key: Expected key value.
        header_name: Header to check (default ``X-API-Key``).
        query_param: Query parameter alternative (default ``api_key``).
    """

    def __init__(
        self,
        app: Any,
        *,
        api_key: str,
        header_name: str = "X-API-Key",
        query_param: str = "api_key",
        skip_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._api_key = api_key
        self._header = header_name
        self._query = query_param
        self._skip_paths = skip_paths if skip_paths is not None else _SKIP_PATHS

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        if path_is_skipped(request.url.path, self._skip_paths):
            response: Response = await call_next(request)
            return response

        key = request.headers.get(self._header) or request.query_params.get(self._query)
        # Compare as bytes: hmac.compare_digest raises TypeError on a str
        # containing non-ASCII, which would turn a bad key into a 500
        # instead of a 401 (and is trivially reachable via ?api_key=…).
        if not key or not hmac.compare_digest(key.encode(), self._api_key.encode()):
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )
        # Record the authenticated principal so downstream authorization (the
        # zero-trust enforcer) can tell an API-key caller from an anonymous
        # one. Without this it looked for JWT claims, found none, and denied
        # a request that had just presented a valid key.
        request.state.api_key_authenticated = True
        request.state.auth_method = "api_key"
        response = await call_next(request)
        return response
