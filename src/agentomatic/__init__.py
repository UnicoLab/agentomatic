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

# Class-owned graph agents + Keras-style training (v0.7 / v1.2)
from agentomatic.agents import (
    AgentDataset,
    AgentExample,
    AgentGraph,
    BaseGraphAgent,
    Callback,
    EarlyStopping,
    ExactKeyMatchMetric,
    GraphBuilder,
    GridSearchOptimizer,
    History,
    Loss,
    NoOpOptimizer,
    PromptFitterBridge,
    WeightedMetric,
    agent_node,
)

# Per-agent connections
from agentomatic.connections import (
    ConnectionPurpose,
    CustomConnectionConfig,
    DatabaseConnectionConfig,
    HttpConnectionConfig,
    VectorConnectionConfig,
    VectorStore,
    get_connections,
    initialize_connections,
    register_connection_type,
    register_connections,
    register_store_provider,
    register_vector_provider,
    register_vector_store_adapter,
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

# First-class ingestion / RAG ops layer (v0.12)
from agentomatic.ingestion import (
    BaseIngestor,
    IngestionRegistry,
    IngestionRequest,
    IngestionResult,
)
from agentomatic.providers.embeddings import register_embedding_provider

# Unified task/execution subsystem (v0.12)
from agentomatic.tasks import (
    TargetType,
    TaskManager,
    TaskRecord,
    TaskStatus,
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
    # Ingestion (v0.12)
    "BaseIngestor",
    "IngestionRegistry",
    "IngestionRequest",
    "IngestionResult",
    # Tasks (v0.12)
    "TaskManager",
    "TaskRecord",
    "TaskStatus",
    "TargetType",
    # Connections
    "DatabaseConnectionConfig",
    "HttpConnectionConfig",
    "VectorConnectionConfig",
    "CustomConnectionConfig",
    "ConnectionPurpose",
    "VectorStore",
    "get_connections",
    "initialize_connections",
    "register_connection_type",
    "register_connections",
    "register_vector_provider",
    "register_vector_store_adapter",
    "register_store_provider",
    "register_embedding_provider",
    # Class-owned graph agents (v0.7)
    "BaseGraphAgent",
    "AgentGraph",
    "GraphBuilder",
    "agent_node",
    "AgentDataset",
    "AgentExample",
    # Training lifecycle (Keras-style)
    "History",
    "Callback",
    "EarlyStopping",
    "Loss",
    "ExactKeyMatchMetric",
    "WeightedMetric",
    "NoOpOptimizer",
    "GridSearchOptimizer",
    "PromptFitterBridge",
    # Version
    "__version__",
]
