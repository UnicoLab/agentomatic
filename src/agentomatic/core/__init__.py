"""Agentomatic core — platform, registry, manifest, state."""

from __future__ import annotations

from .lifespan import configure_logging, create_lifespan
from .manifest import AgentManifest, RegisteredAgent
from .platform import AgentPlatform
from .registry import AgentRegistry
from .router_factory import (
    A2ATaskRequest,
    AgentChatRequest,
    AgentInvokeRequest,
    AgentInvokeResponse,
    create_default_router,
)
from .state import BaseAgentState

__all__ = [
    "A2ATaskRequest",
    "AgentChatRequest",
    "AgentInvokeRequest",
    "AgentInvokeResponse",
    "AgentManifest",
    "AgentPlatform",
    "AgentRegistry",
    "BaseAgentState",
    "RegisteredAgent",
    "configure_logging",
    "create_default_router",
    "create_lifespan",
]
