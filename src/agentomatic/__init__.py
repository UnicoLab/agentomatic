"""
Agentomatic — Drop agents, not code.

Zero-code multi-agent API platform framework.

Usage::

    from agentomatic import AgentPlatform, AgentManifest

    platform = AgentPlatform.from_folder("agents/")
    app = platform.build()

    # Run: uvicorn main:app --reload

With storage::

    from agentomatic import AgentPlatform
    from agentomatic.storage import MemoryStore, SQLAlchemyStore

    platform = AgentPlatform.from_folder(
        "agents/",
        store=MemoryStore(),  # or SQLAlchemyStore("postgresql+asyncpg://...")
        enable_metrics=True,
        enable_auth=True,
        auth_api_key="secret",
    )

With stacks (v0.6)::

    from agentomatic import AgentPlatform

    platform = AgentPlatform.from_folder("agents/", stack="local")
    app = platform.build()
"""

from __future__ import annotations

# Version
from agentomatic._version import __version__

# Core public API
from agentomatic.core.manifest import AgentManifest, RegisteredAgent
from agentomatic.core.memory_manager import ConversationMemoryManager
from agentomatic.core.platform import AgentPlatform
from agentomatic.core.registry import AgentRegistry
from agentomatic.core.schemas import SchemaValidator
from agentomatic.core.state import BaseAgentState
from agentomatic.prompts import PromptManager

# Protocols
from agentomatic.protocols.decorators import APIResponse, handle_api_errors, log_api_call

# Studio
from agentomatic.studio import GraphInspector, RunTracker

# Pipelines (lazy — avoids hard failure if yaml not installed)
try:
    from agentomatic.pipelines import Pipeline, PipelineConfig, PipelineResult
except ImportError:
    Pipeline = None  # type: ignore[assignment,misc]
    PipelineConfig = None  # type: ignore[assignment,misc]
    PipelineResult = None  # type: ignore[assignment,misc]

# Class-owned graph agents (v0.7)
from agentomatic.agents import (
    AgentDataset,
    AgentExample,
    AgentGraph,
    BaseGraphAgent,
    GraphBuilder,
    agent_node,
)

# Per-agent connections
from agentomatic.connections import (
    ConnectionPurpose,
    CustomConnectionConfig,
    DatabaseConnectionConfig,
    HttpConnectionConfig,
    VectorConnectionConfig,
    get_connections,
    register_connection_type,
    register_vector_provider,
)

# Custom endpoints (httpx calls to deployed model services)
from agentomatic.endpoints import (
    AggregationStrategy,
    AuthType,
    BaseEndpoint,
    EndpointRegistry,
    UpstreamAuthConfig,
    UpstreamConfig,
)

__all__ = [
    # Core
    "AgentPlatform",
    "AgentManifest",
    "RegisteredAgent",
    "AgentRegistry",
    "BaseAgentState",
    "ConversationMemoryManager",
    "SchemaValidator",
    # Protocols
    "APIResponse",
    "handle_api_errors",
    "log_api_call",
    # Prompts
    "PromptManager",
    # Studio
    "GraphInspector",
    "RunTracker",
    # Pipelines
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    # Custom endpoints
    "BaseEndpoint",
    "EndpointRegistry",
    "UpstreamConfig",
    "UpstreamAuthConfig",
    "AuthType",
    "AggregationStrategy",
    # Connections
    "DatabaseConnectionConfig",
    "HttpConnectionConfig",
    "VectorConnectionConfig",
    "CustomConnectionConfig",
    "ConnectionPurpose",
    "get_connections",
    "register_connection_type",
    "register_vector_provider",
    # Class-owned graph agents (v0.7)
    "BaseGraphAgent",
    "AgentGraph",
    "GraphBuilder",
    "agent_node",
    "AgentDataset",
    "AgentExample",
    # Version
    "__version__",
]
