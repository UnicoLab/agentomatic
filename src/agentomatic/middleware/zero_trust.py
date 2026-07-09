"""Per-agent zero-trust enforcement middleware.

Bridges the :class:`~agentomatic.security.zero_trust.ZeroTrustEnforcer` into
the request pipeline: for every request targeting an agent route
(``{api_prefix}/{agent}/...``) it evaluates the agent's
:class:`~agentomatic.security.policy.AgentSecurityPolicy` (auth required,
allowed roles, allowed scopes) and rejects unauthorised requests.

This middleware must run *after* the JWT middleware so that
``request.state.jwt_claims`` is populated.  The platform adds it in the
correct order automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

    from agentomatic.core.registry import AgentRegistry
    from agentomatic.security.zero_trust import ZeroTrustEnforcer


class ZeroTrustMiddleware(BaseHTTPMiddleware):
    """Enforce per-agent security policies on inbound agent requests."""

    def __init__(
        self,
        app: object,
        *,
        enforcer: ZeroTrustEnforcer,
        registry: AgentRegistry,
        api_prefix: str = "/api/v1",
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._enforcer = enforcer
        self._registry = registry
        self._api_prefix = api_prefix.rstrip("/")

    def _agent_for_path(self, path: str) -> str | None:
        """Return the target agent name for a request path, if any."""
        prefix = f"{self._api_prefix}/"
        if not path.startswith(prefix):
            return None
        remainder = path[len(prefix) :]
        segment = remainder.split("/", 1)[0]
        if not segment:
            return None
        if self._registry.get(segment) is None:
            return None
        return segment

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Enforce the agent's policy before forwarding the request."""
        agent_name = self._agent_for_path(request.url.path)
        if agent_name is not None:
            ok, reason = self._enforcer.verify_request(request, agent_name)
            if not ok:
                claims = getattr(request.state, "jwt_claims", None)
                status = 401 if not claims else 403
                return JSONResponse(
                    status_code=status,
                    content={
                        "detail": reason,
                        "agent": agent_name,
                        "error": "zero_trust_denied",
                    },
                )
        return await call_next(request)
