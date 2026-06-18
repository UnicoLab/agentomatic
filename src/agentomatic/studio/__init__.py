"""Agentomatic Studio — Debug & inspection API for frontend tooling.

Provides REST + SSE endpoints for graph visualization, agent execution
tracing, state inspection, and checkpoint browsing.

Usage::

    from agentomatic.studio import GraphInspector, RunTracker
    from agentomatic.studio.router import create_studio_router

    router = create_studio_router(registry=registry, store=store)
"""

from __future__ import annotations

from agentomatic.studio.graph_inspector import GraphInspector
from agentomatic.studio.run_tracker import RunTracker

__all__ = [
    "GraphInspector",
    "RunTracker",
]
