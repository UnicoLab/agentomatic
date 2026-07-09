"""Configuration models for custom endpoints and their upstream services.

These models describe how a custom endpoint authenticates to and calls
one or more deployed model services (or any HTTP API) via ``httpx``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AuthType(StrEnum):
    """Supported upstream authentication schemes."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"


class UpstreamAuthConfig(BaseModel):
    """Authentication configuration for an upstream service.

    The same model covers static credentials (API key / bearer / basic)
    and the OAuth2 *client credentials* grant, where a short-lived access
    token is fetched from ``token_url`` and cached until it expires.

    Environment interpolation: any string field may reference an
    environment variable using ``${VAR_NAME}`` and it is resolved lazily
    at request time (never persisted), which keeps secrets out of config
    files and logs.
    """

    type: AuthType = AuthType.NONE

    # api_key / bearer
    api_key: str = Field("", description="API key or token value (supports ${ENV}).")
    header_name: str = Field(
        "Authorization",
        description="Header used to carry the credential (api_key/bearer).",
    )
    header_prefix: str = Field(
        "Bearer",
        description="Prefix prepended to the credential value (e.g. 'Bearer').",
    )

    # basic
    username: str = Field("", description="Basic-auth username (supports ${ENV}).")
    password: str = Field("", description="Basic-auth password (supports ${ENV}).")

    # oauth2 client credentials
    token_url: str = Field("", description="OAuth2 token endpoint (supports ${ENV}).")
    client_id: str = Field("", description="OAuth2 client id (supports ${ENV}).")
    client_secret: str = Field("", description="OAuth2 client secret (supports ${ENV}).")
    scope: str = Field("", description="Optional OAuth2 scope(s), space-delimited.")
    audience: str = Field("", description="Optional OAuth2 audience claim.")
    token_leeway: int = Field(
        30,
        ge=0,
        description="Seconds subtracted from token expiry to trigger early refresh.",
    )


class UpstreamConfig(BaseModel):
    """Declarative configuration for a single upstream model service."""

    name: str = Field(..., min_length=1, description="Logical name for this upstream.")
    base_url: str = Field(..., description="Base URL of the service (supports ${ENV}).")
    path: str = Field("", description="Default path appended to base_url for calls.")
    method: str = Field("POST", description="Default HTTP method for calls.")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Static headers merged into every request (supports ${ENV}).",
    )
    auth: UpstreamAuthConfig = Field(default_factory=UpstreamAuthConfig)
    timeout: float = Field(30.0, gt=0, description="Per-request timeout in seconds.")
    max_retries: int = Field(2, ge=0, le=10, description="Retry attempts on transport errors.")
    verify_ssl: bool = Field(True, description="Verify TLS certificates.")
    weight: float = Field(1.0, gt=0, description="Relative weight for aggregation/voting.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregationStrategy(StrEnum):
    """How to combine responses from multiple upstreams."""

    ALL = "all"
    FIRST_SUCCESS = "first_success"
    MAJORITY = "majority"


class UpstreamResult(BaseModel):
    """Result of a single upstream call within a fan-out."""

    upstream: str
    ok: bool
    status_code: int | None = None
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class EndpointCallRequest(BaseModel):
    """Default request body for endpoints that fan out to upstreams.

    Endpoints may declare their own typed schemas instead; this generic
    model is the fallback used when no ``InputT`` is provided.
    """

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON payload forwarded to the upstream model(s).",
    )
    upstreams: list[str] | None = Field(
        default=None,
        description="Optional subset of upstream names to call (defaults to all).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EndpointCallResponse(BaseModel):
    """Default response body for fan-out endpoints."""

    endpoint: str
    strategy: str
    ok: bool
    aggregated: Any = None
    results: list[UpstreamResult] = Field(default_factory=list)
    duration_ms: float = 0.0
