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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.types import Receive, Scope, Send

    from agentomatic.core.registry import AgentRegistry


class ConnectionsMiddleware:
    """Pure ASGI middleware exposing the routed agent's connection manager."""

    def __init__(
        self,
        app: Any,
        *,
        registry: AgentRegistry,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.app = app
        self._registry = registry
        self._api_prefix = api_prefix.rstrip("/")

    def _agent_for_path(self, path: str) -> str | None:
        """Return the target agent name for a request path, if any."""
        prefix = f"{self._api_prefix}/"
        if not path.startswith(prefix):
            return None
        segment = path[len(prefix) :].split("/", 1)[0]
        if not segment:
            return None
        # URLs may carry the slug; connection scopes are keyed by the
        # registered name (they differ when manifest.slug != folder name).
        resolve = getattr(self._registry, "resolve_name", None)
        if callable(resolve):
            name = resolve(segment)
            return name if isinstance(name, str) else None
        return segment if self._registry.get(segment) is not None else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Attach connection managers to ``scope['state']`` without buffering."""
        if scope.get("type") == "http":
            from agentomatic.connections.manager import PLATFORM_SCOPE, get_connections

            platform_mgr = get_connections(PLATFORM_SCOPE)
            state = scope.setdefault("state", {})
            state["platform_connections"] = platform_mgr

            agent_name = self._agent_for_path(str(scope.get("path") or ""))
            state["agent_name"] = agent_name
            state["connections"] = get_connections(agent_name) if agent_name else platform_mgr
        await self.app(scope, receive, send)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Legacy Starlette-style dispatch helper preserved for unit tests."""
        from agentomatic.connections.manager import PLATFORM_SCOPE, get_connections

        platform_mgr = get_connections(PLATFORM_SCOPE)
        request.state.platform_connections = platform_mgr

        agent_name = self._agent_for_path(request.url.path)
        request.state.agent_name = agent_name
        request.state.connections = get_connections(agent_name) if agent_name else platform_mgr
        return await call_next(request)
