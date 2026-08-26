"""Pydantic response models for the control plane API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ControlInfo(BaseModel):
    """High-level platform information."""

    platform: str
    version: str
    uptime_seconds: float
    maintenance_mode: bool
    agent_count: int
    endpoint_count: int
    plugin_count: int
    pipeline_count: int
    connection_scopes: int
    control_token_required: bool = Field(
        default=False,
        description="Whether control-plane mutation endpoints require X-Control-Token",
    )


class ControlAgentInfo(BaseModel):
    """Operational view of a single agent."""

    name: str
    slug: str
    description: str = ""
    version: str = "1.0.0"
    framework: str = ""
    enabled: bool = True
    requires_auth: bool = False
    allowed_roles: list[str] = Field(default_factory=list)
    allowed_scopes: list[str] = Field(default_factory=list)
    connections: list[str] = Field(default_factory=list)
    health: dict[str, Any] = Field(default_factory=dict)


class ControlEndpointInfo(BaseModel):
    """Operational view of a custom endpoint."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    path: str = ""
    methods: list[str] = Field(default_factory=list)
    aggregation: str = ""
    upstreams: list[str] = Field(default_factory=list)
    ready: bool = False


class ControlConnectionInfo(BaseModel):
    """Health summary for a connection scope."""

    scope: str
    connections: dict[str, Any] = Field(default_factory=dict)


class ControlConnectionProbe(BaseModel):
    """Stable OpenAPI contract for a single live connection probe.

    Connection health implementations can contain arbitrary driver metadata,
    including URLs and credentials.  The control plane therefore exposes an
    explicit, safe operational contract rather than passing implementation
    details through to API clients and Studio.
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="unknown", description="Current health status")
    connection: str | None = Field(default=None, description="Registered connection name")
    kind: str | None = Field(
        default=None, description="Connection kind, such as database or vector"
    )
    purpose: str | None = Field(default=None, description="Declared application purpose")
    backend: str | None = Field(default=None, description="Active backend implementation")
    provider: str | None = Field(default=None, description="External provider, when applicable")
    detail: str | None = Field(
        default=None, description="Safe configuration guidance, when available"
    )
    error: str | None = Field(default=None, description="Sanitised diagnostic when unhealthy")


class ToggleResponse(BaseModel):
    """Result of an enable/disable/maintenance toggle."""

    ok: bool = True
    target: str
    state: str


class MaintenanceRequest(BaseModel):
    """Request body for toggling maintenance mode."""

    enabled: bool = True


class ControlMetricsSummary(BaseModel):
    """Coarse counters for a quick operational overview."""

    agents: int
    disabled_agents: int
    endpoints: int
    plugins: int
    pipelines: int
    connection_scopes: int
    prometheus_available: bool
    metrics_path: str = "/metrics"
