"""Per-request connection context middleware.

Aligns per-agent connections with the middleware stack: for every request it
attaches the resolved :class:`~agentomatic.connections.manager.ConnectionManager`
to ``request.state`` so route handlers and downstream middleware can reach an
agent's databases, vector stores and HTTP services without global lookups::

    async def handler(request: Request):
        db = request.state.connections.database("main")
        kb = request.state.connections.vector("kb")

For requests that target an agent route (``{api_prefix}/{agent}/...``),
``request.state.connections`` is that agent's scope; otherwise it is the
shared platform scope.  The platform scope is always available at
``request.state.platform_connections``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

    from agentomatic.core.registry import AgentRegistry


class ConnectionsMiddleware(BaseHTTPMiddleware):
    """Expose the routed agent's connection manager on ``request.state``."""

    def __init__(
        self,
        app: object,
        *,
        registry: AgentRegistry,
        api_prefix: str = "/api/v1",
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._registry = registry
        self._api_prefix = api_prefix.rstrip("/")

    def _agent_for_path(self, path: str) -> str | None:
        """Return the target agent name for a request path, if any."""
        prefix = f"{self._api_prefix}/"
        if not path.startswith(prefix):
            return None
        segment = path[len(prefix) :].split("/", 1)[0]
        if not segment or self._registry.get(segment) is None:
            return None
        return segment

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Attach connection managers to ``request.state`` then continue."""
        from agentomatic.connections.manager import PLATFORM_SCOPE, get_connections

        platform_mgr = get_connections(PLATFORM_SCOPE)
        request.state.platform_connections = platform_mgr

        agent_name = self._agent_for_path(request.url.path)
        request.state.agent_name = agent_name
        request.state.connections = get_connections(agent_name) if agent_name else platform_mgr
        return await call_next(request)
