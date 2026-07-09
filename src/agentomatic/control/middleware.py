"""Control-plane request gating middleware.

Enforces maintenance mode and per-agent disable toggles set via the control
API, returning ``503 Service Unavailable`` for affected agent routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

    from agentomatic.control.state import ControlPlaneState


class MaintenanceMiddleware(BaseHTTPMiddleware):
    """Reject requests to disabled agents or during maintenance mode."""

    def __init__(
        self,
        app: object,
        *,
        state: ControlPlaneState,
        api_prefix: str = "/api/v1",
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._state = state
        self._api_prefix = api_prefix.rstrip("/")

    def _agent_for_path(self, path: str) -> str | None:
        """Return the target agent name for a request path, if any."""
        prefix = f"{self._api_prefix}/"
        if not path.startswith(prefix):
            return None
        segment = path[len(prefix) :].split("/", 1)[0]
        return segment or None

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Gate the request based on control-plane state."""
        path = request.url.path

        # Never block the control API itself or platform probes.
        if f"{self._api_prefix}/control" in path or path in ("/health", "/readiness"):
            return await call_next(request)

        if self._state.maintenance_mode:
            return JSONResponse(
                status_code=503,
                content={"detail": "Platform is in maintenance mode", "error": "maintenance"},
            )

        agent_name = self._agent_for_path(path)
        if agent_name and self._state.is_agent_disabled(agent_name):
            return JSONResponse(
                status_code=503,
                content={
                    "detail": f"Agent '{agent_name}' is disabled",
                    "agent": agent_name,
                    "error": "agent_disabled",
                },
            )

        return await call_next(request)
