"""Control-plane request gating middleware.

Enforces maintenance mode and per-agent disable toggles set via the control
API, returning ``503 Service Unavailable`` for affected agent routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send

    from agentomatic.control.state import ControlPlaneState


class MaintenanceMiddleware:
    """Pure ASGI middleware rejecting requests to disabled agents or in maintenance mode."""

    def __init__(
        self,
        app: Any,
        *,
        state: ControlPlaneState,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.app = app
        self._state = state
        self._api_prefix = api_prefix.rstrip("/")

    def _agent_for_path(self, path: str) -> str | None:
        """Return the target agent name for a request path, if any."""
        prefix = f"{self._api_prefix}/"
        if not path.startswith(prefix):
            return None
        segment = path[len(prefix) :].split("/", 1)[0]
        return segment or None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Gate HTTP requests based on control-plane state without buffering SSE."""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")

        # Never block the control API itself, Studio debug API/UI, or probes.
        if (
            f"{self._api_prefix}/control" in path
            or path.startswith("/studio")
            or path in ("/health", "/healthz", "/ready", "/readiness", "/metrics")
        ):
            await self.app(scope, receive, send)
            return

        if self._state.maintenance_mode:
            resp = JSONResponse(
                status_code=503,
                content={"detail": "Platform is in maintenance mode", "error": "maintenance"},
            )
            await resp(scope, receive, send)
            return

        agent_name = self._agent_for_path(path)
        if agent_name and self._state.is_agent_disabled(agent_name):
            resp = JSONResponse(
                status_code=503,
                content={
                    "detail": f"Agent '{agent_name}' is disabled",
                    "agent": agent_name,
                    "error": "agent_disabled",
                },
            )
            await resp(scope, receive, send)
            return

        await self.app(scope, receive, send)
