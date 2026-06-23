"""Agentomatic core — platform, registry, manifest, state."""

from __future__ import annotations

from .lifespan import configure_logging, create_lifespan
from .manifest import AgentManifest, RegisteredAgent
from .memory_manager import ConversationMemoryManager
from .platform import AgentPlatform
from .registry import AgentRegistry
from .router_factory import (
    A2ATaskRequest,
    AgentChatRequest,
    AgentInvokeRequest,
    AgentInvokeResponse,
    AgentSuspendedException,
    ApproveSuspendedRequest,
    CreateThreadRequest,
    ForkThreadRequest,
    RejectSuspendedRequest,
    UpdateThreadRequest,
    create_default_router,
)
from .schemas import SchemaValidator
from .state import BaseAgentState

__all__ = [
    "A2ATaskRequest",
    "AgentChatRequest",
    "AgentInvokeRequest",
    "AgentInvokeResponse",
    "AgentManifest",
    "AgentPlatform",
    "AgentRegistry",
    "AgentSuspendedException",
    "ApproveSuspendedRequest",
    "BaseAgentState",
    "ConversationMemoryManager",
    "CreateThreadRequest",
    "ForkThreadRequest",
    "RegisteredAgent",
    "RejectSuspendedRequest",
    "SchemaValidator",
    "UpdateThreadRequest",
    "configure_logging",
    "create_default_router",
    "create_lifespan",
]
