"""Control plane REST API.

A production-oriented admin surface for observing and operating the
platform at runtime: inspect agents / endpoints / connections, drain or
re-enable individual agents, toggle maintenance mode, and read a sanitised
configuration snapshot.

Mutating operations can be protected by a shared ``control_token`` supplied
via the ``X-Control-Token`` header.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Header, HTTPException
from loguru import logger

from agentomatic.control.models import (
    ControlAgentInfo,
    ControlConnectionInfo,
    ControlEndpointInfo,
    ControlInfo,
    ControlMetricsSummary,
    MaintenanceRequest,
    ToggleResponse,
)
from agentomatic.control.state import ControlPlaneState

if TYPE_CHECKING:
    from agentomatic.core.platform import AgentPlatform


def create_control_router(
    platform: AgentPlatform,
    state: ControlPlaneState,
    *,
    control_token: str = "",
) -> APIRouter:
    """Create the control plane router.

    Args:
        platform: The owning :class:`AgentPlatform` (read for live state).
        state: Shared :class:`ControlPlaneState`.
        control_token: Optional shared secret; when set, mutating endpoints
            require a matching ``X-Control-Token`` header.

    Returns:
        A configured :class:`~fastapi.APIRouter`.
    """
    router = APIRouter(tags=["Control Plane"])

    def _authorize(token: str | None) -> None:
        """Reject mutating requests lacking the configured control token."""
        if control_token and token != control_token:
            raise HTTPException(status_code=401, detail="Invalid or missing control token")

    def _agent_policy(agent: Any) -> tuple[bool, list[str], list[str]]:
        """Extract (requires_auth, roles, scopes) from an agent policy."""
        policy = getattr(agent, "security_policy", None)
        if policy is None:
            return False, [], []
        return (
            bool(getattr(policy, "require_auth", False)),
            list(getattr(policy, "allowed_roles", []) or []),
            list(getattr(policy, "allowed_scopes", []) or []),
        )

    def _connection_names(agent: Any) -> list[str]:
        """Return the connection names declared by an agent."""
        configs = getattr(agent, "connections", None) or []
        return [getattr(c, "name", "?") for c in configs]

    # -- introspection ---------------------------------------------------

    @router.get("", response_model=ControlInfo, summary="Platform overview")
    async def control_info() -> ControlInfo:
        """Return a high-level snapshot of the platform."""
        from agentomatic.connections.manager import all_managers

        return ControlInfo(
            platform=platform.title,
            version=platform.version,
            uptime_seconds=state.uptime_seconds,
            maintenance_mode=state.maintenance_mode,
            agent_count=platform._registry.count,
            endpoint_count=platform.endpoint_registry.count,
            plugin_count=platform._plugin_registry.count,
            pipeline_count=len(platform.pipelines),
            connection_scopes=len(all_managers()),
        )

    @router.get("/agents", response_model=list[ControlAgentInfo], summary="List agents")
    async def list_agents() -> list[ControlAgentInfo]:
        """List all agents with operational metadata."""
        infos: list[ControlAgentInfo] = []
        for name, agent in platform._registry.all().items():
            requires_auth, roles, scopes = _agent_policy(agent)
            try:
                health = await agent.health_check()
            except Exception as exc:  # noqa: BLE001
                health = {"status": "error", "error": str(exc)}
            infos.append(
                ControlAgentInfo(
                    name=name,
                    slug=agent.slug,
                    description=agent.manifest.description,
                    version=agent.manifest.version,
                    framework=agent.manifest.framework,
                    enabled=not state.is_agent_disabled(name),
                    requires_auth=requires_auth,
                    allowed_roles=roles,
                    allowed_scopes=scopes,
                    connections=_connection_names(agent),
                    health=health,
                )
            )
        return infos

    @router.get(
        "/agents/{name}",
        response_model=ControlAgentInfo,
        summary="Get agent detail",
    )
    async def get_agent(name: str) -> ControlAgentInfo:
        """Return operational detail for a single agent."""
        agent = platform._registry.get(name)
        if agent is None:
            raise HTTPException(404, f"Agent '{name}' not found")
        requires_auth, roles, scopes = _agent_policy(agent)
        try:
            health = await agent.health_check()
        except Exception as exc:  # noqa: BLE001
            health = {"status": "error", "error": str(exc)}
        return ControlAgentInfo(
            name=name,
            slug=agent.slug,
            description=agent.manifest.description,
            version=agent.manifest.version,
            framework=agent.manifest.framework,
            enabled=not state.is_agent_disabled(name),
            requires_auth=requires_auth,
            allowed_roles=roles,
            allowed_scopes=scopes,
            connections=_connection_names(agent),
            health=health,
        )

    @router.get(
        "/endpoints",
        response_model=list[ControlEndpointInfo],
        summary="List custom endpoints",
    )
    async def list_endpoints() -> list[ControlEndpointInfo]:
        """List all registered custom endpoints."""
        infos: list[ControlEndpointInfo] = []
        for ep in platform.endpoint_registry.list_endpoints().values():
            data = ep.info()
            infos.append(ControlEndpointInfo(**data))
        return infos

    @router.get(
        "/connections",
        response_model=list[ControlConnectionInfo],
        summary="Connection health by scope",
    )
    async def list_connections() -> list[ControlConnectionInfo]:
        """Report connection health for every scope."""
        from agentomatic.connections.manager import all_managers

        out: list[ControlConnectionInfo] = []
        for scope, manager in all_managers().items():
            out.append(
                ControlConnectionInfo(scope=scope, connections=await manager.health_check())
            )
        return out

    @router.get("/health", summary="Aggregate platform health")
    async def control_health() -> dict[str, Any]:
        """Aggregate health across agents, endpoints and connections."""
        from agentomatic.connections.manager import all_managers

        agents: dict[str, Any] = {}
        for name, agent in platform._registry.all().items():
            try:
                agents[name] = await agent.health_check()
            except Exception as exc:  # noqa: BLE001
                agents[name] = {"status": "error", "error": str(exc)}

        connections: dict[str, Any] = {}
        for scope, manager in all_managers().items():
            connections[scope] = await manager.health_check()

        overall = (
            "healthy" if all(a.get("status") == "healthy" for a in agents.values()) else "degraded"
        )
        return {
            "status": "maintenance" if state.maintenance_mode else overall,
            "uptime_seconds": state.uptime_seconds,
            "agents": agents,
            "endpoints": [
                e.endpoint_name for e in platform.endpoint_registry.list_endpoints().values()
            ],
            "connections": connections,
        }

    @router.get(
        "/metrics/summary",
        response_model=ControlMetricsSummary,
        summary="Coarse operational counters",
    )
    async def metrics_summary() -> ControlMetricsSummary:
        """Return coarse counters for dashboards / quick checks."""
        from agentomatic.connections.manager import all_managers

        try:
            from agentomatic.observability.metrics import HAS_PROMETHEUS
        except Exception:  # noqa: BLE001
            HAS_PROMETHEUS = False

        return ControlMetricsSummary(
            agents=platform._registry.count,
            disabled_agents=len(state.disabled_agents),
            endpoints=platform.endpoint_registry.count,
            plugins=platform._plugin_registry.count,
            pipelines=len(platform.pipelines),
            connection_scopes=len(all_managers()),
            prometheus_available=bool(HAS_PROMETHEUS),
        )

    @router.get("/config", summary="Sanitised configuration snapshot")
    async def control_config() -> dict[str, Any]:
        """Return a sanitised view of the effective platform configuration."""
        return {
            "title": platform.title,
            "version": platform.version,
            "api_prefix": platform.api_prefix,
            "agents_dir": str(platform.agents_dir),
            "features": {
                "auth": platform._enable_auth,
                "jwt_auth": platform._enable_jwt_auth,
                "zero_trust": platform._enable_zero_trust,
                "rate_limit": platform._enable_rate_limit,
                "metrics": platform._enable_metrics,
                "telemetry": platform._enable_telemetry,
                "studio": platform._enable_studio,
            },
        }

    # -- operations (mutating) ------------------------------------------

    @router.post(
        "/agents/{name}/disable",
        response_model=ToggleResponse,
        summary="Disable (drain) an agent",
    )
    async def disable_agent(
        name: str,
        x_control_token: str | None = Header(default=None),
    ) -> ToggleResponse:
        """Stop routing traffic to an agent (returns 503 for its routes)."""
        _authorize(x_control_token)
        if platform._registry.get(name) is None:
            raise HTTPException(404, f"Agent '{name}' not found")
        state.disable_agent(name)
        logger.warning(f"🛑 Control plane: agent '{name}' disabled")
        return ToggleResponse(target=name, state="disabled")

    @router.post(
        "/agents/{name}/enable",
        response_model=ToggleResponse,
        summary="Re-enable an agent",
    )
    async def enable_agent(
        name: str,
        x_control_token: str | None = Header(default=None),
    ) -> ToggleResponse:
        """Resume routing traffic to a previously disabled agent."""
        _authorize(x_control_token)
        if platform._registry.get(name) is None:
            raise HTTPException(404, f"Agent '{name}' not found")
        state.enable_agent(name)
        logger.info(f"✅ Control plane: agent '{name}' enabled")
        return ToggleResponse(target=name, state="enabled")

    @router.post(
        "/maintenance",
        response_model=ToggleResponse,
        summary="Toggle maintenance mode",
    )
    async def set_maintenance(
        request: MaintenanceRequest,
        x_control_token: str | None = Header(default=None),
    ) -> ToggleResponse:
        """Enable or disable platform-wide maintenance mode."""
        _authorize(x_control_token)
        state.maintenance_mode = request.enabled
        logger.warning(f"🔧 Control plane: maintenance_mode={request.enabled}")
        return ToggleResponse(
            target="platform",
            state="maintenance" if request.enabled else "active",
        )

    return router
