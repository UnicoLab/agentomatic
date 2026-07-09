"""Custom endpoints API for Agentomatic.

Custom endpoints let you expose your own HTTP APIs (auto-mounted alongside
agents, plugins and pipelines) that call deployed model services with
authentication and aggregate their outputs.
"""

from __future__ import annotations

from agentomatic.endpoints.base import BaseEndpoint
from agentomatic.endpoints.client import MultiModelClient, UpstreamClient
from agentomatic.endpoints.models import (
    AggregationStrategy,
    AuthType,
    EndpointCallRequest,
    EndpointCallResponse,
    UpstreamAuthConfig,
    UpstreamConfig,
    UpstreamResult,
)
from agentomatic.endpoints.registry import EndpointRegistry
from agentomatic.endpoints.router import create_endpoint_router

__all__ = [
    "AggregationStrategy",
    "AuthType",
    "BaseEndpoint",
    "EndpointCallRequest",
    "EndpointCallResponse",
    "EndpointRegistry",
    "MultiModelClient",
    "UpstreamAuthConfig",
    "UpstreamClient",
    "UpstreamConfig",
    "UpstreamResult",
    "create_endpoint_router",
]
