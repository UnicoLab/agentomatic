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
"""

from __future__ import annotations

# Version
from agentomatic._version import __version__

# Core public API
from agentomatic.core.manifest import AgentManifest, RegisteredAgent
from agentomatic.core.platform import AgentPlatform
from agentomatic.core.registry import AgentRegistry
from agentomatic.core.state import BaseAgentState

# Protocols
from agentomatic.protocols.decorators import APIResponse, handle_api_errors, log_api_call

__all__ = [
    # Core
    "AgentPlatform",
    "AgentManifest",
    "RegisteredAgent",
    "AgentRegistry",
    "BaseAgentState",
    # Protocols
    "APIResponse",
    "handle_api_errors",
    "log_api_call",
    # Version
    "__version__",
]
