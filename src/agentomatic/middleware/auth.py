"""API key authentication middleware.

Enabled via ``FEATURES__ENABLE_AUTH=true`` and ``AUTH__API_KEY=your-key``.
Skips health/readiness probes. Supports both header and query param.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_SKIP_PATHS = {"/health", "/healthz", "/readiness", "/docs", "/openapi.json", "/redoc", "/"}


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
        app,
        *,
        api_key: str,
        header_name: str = "X-API-Key",
        query_param: str = "api_key",
    ) -> None:
        super().__init__(app)
        self._api_key = api_key
        self._header = header_name
        self._query = query_param

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        key = request.headers.get(self._header) or request.query_params.get(self._query)
        if not key or key != self._api_key:
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )
        return await call_next(request)
