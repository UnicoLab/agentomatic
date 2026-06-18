"""Agentomatic Studio — Universal Debug & Inspection API.

Provides REST + SSE endpoints for graph visualization, agent execution
tracing, state inspection, and checkpoint browsing. Works with **any**
agent framework via the adapter system.

Usage::

    from agentomatic.studio import StudioAdapter, studio_graph, studio_state
    from agentomatic.studio.router import create_studio_router
    from agentomatic.studio.adapters import resolve_adapter

    router = create_studio_router(registry=registry, store=store)
"""

from __future__ import annotations

from agentomatic.studio.adapter import StudioAdapter
from agentomatic.studio.decorators import (
    register_studio_hooks,
    studio_graph,
    studio_state,
    studio_stream,
)
from agentomatic.studio.graph_inspector import GraphInspector
from agentomatic.studio.run_tracker import RunTracker

__all__ = [
    "GraphInspector",
    "RunTracker",
    "StudioAdapter",
    "register_studio_hooks",
    "studio_graph",
    "studio_state",
    "studio_stream",
]
