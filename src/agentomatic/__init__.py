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

from typing import Any

# Version
from agentomatic._version import __version__

# Core public API
from agentomatic.artifacts import ArtifactRegistry
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
    pass  # exposed as None via module __getattr__ below

# Class-owned graph agents + Keras-style training (v0.7 / v1.2)
from agentomatic.agents import (
    AgentDataset,
    AgentExample,
    AgentGraph,
    BaseGraphAgent,
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
    GraphBuilder,
    GridSearchOptimizer,
    History,
    Loss,
    MetricLoss,
    NoOpOptimizer,
    PromptFitterBridge,
    ResponseSimilarityMetric,
    WeightedMetric,
    agent_node,
)

# Unified callbacks (v1.10) — canonical location for all callbacks
from agentomatic.callbacks import (
    CallbackContext,
    EpochDiffCallback,
    ModelCheckpoint,
    NaNStopping,
    OptimizeCallback,
    OptimizeEarlyStopping,
    PlateauStopping,
    ProgressLogger,
    ScoreThreshold,
    TemperatureScheduler,
    TrainingCallback,
    TrainingEarlyStopping,
    default_callbacks,
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

# First-class LangChain / LangGraph adapter (v1.10)
from agentomatic.langchain_adapter import (
    AgentAdapter,
    adapt_langgraph_agent,
    collect_stream,
    dict_to_messages,
    extract_system_prompt,
    inject_config,
    inject_system_prompt,
    is_chain,
    make_config,
    messages_to_dict,
    resolve_prompt,
    serialize_messages,
    tools_to_names,
    wrap_chain_as_async_node,
    wrap_chain_as_node,
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
    "ArtifactRegistry",
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
    # LangChain / LangGraph adapter (v1.10)
    "AgentAdapter",
    "adapt_langgraph_agent",
    "collect_stream",
    "dict_to_messages",
    "extract_system_prompt",
    "inject_config",
    "inject_system_prompt",
    "is_chain",
    "make_config",
    "messages_to_dict",
    "resolve_prompt",
    "serialize_messages",
    "tools_to_names",
    "wrap_chain_as_async_node",
    "wrap_chain_as_node",
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
    # Unified callbacks (v1.10)
    "TrainingCallback",
    "TrainingEarlyStopping",
    "EarlyStopping",
    "EpochDiffCallback",
    "CallableMetric",
    "ContainsTermsMetric",
    "ResponseSimilarityMetric",
    "MetricLoss",
    "OptimizeCallback",
    "OptimizeEarlyStopping",
    "ModelCheckpoint",
    "NaNStopping",
    "PlateauStopping",
    "ProgressLogger",
    "ScoreThreshold",
    "TemperatureScheduler",
    "default_callbacks",
    "CallbackContext",
    "Loss",
    "ExactKeyMatchMetric",
    "WeightedMetric",
    "NoOpOptimizer",
    "GridSearchOptimizer",
    "PromptFitterBridge",
    # Version
    "__version__",
]

# Keras-style canonical name for the training early-stopping callback.
EarlyStopping = TrainingEarlyStopping


def __getattr__(name: str) -> Any:
    """Return ``None`` for optional pipeline exports when pipelines are unavailable."""
    if name in ("Pipeline", "PipelineConfig", "PipelineResult"):
        return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
