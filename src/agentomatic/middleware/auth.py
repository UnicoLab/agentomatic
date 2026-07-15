"""API key authentication middleware.

Enabled via ``FEATURES__ENABLE_AUTH=true`` and ``AUTH__API_KEY=your-key``.
Skips health/readiness probes. Supports both header and query param.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from agentomatic.middleware.pathutils import path_is_skipped

_SKIP_PATHS = {
    "/health",
    "/healthz",
    "/readiness",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/",
    "/studio",
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
        app,
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

    async def dispatch(self, request: Request, call_next) -> Response:
        if path_is_skipped(request.url.path, self._skip_paths):
            response: Response = await call_next(request)
            return response

        key = request.headers.get(self._header) or request.query_params.get(self._query)
        if not key or key != self._api_key:
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )
        response = await call_next(request)
        return response
