"""Pydantic response models for the control plane API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
