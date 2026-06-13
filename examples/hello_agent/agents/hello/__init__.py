"""Minimal example agent — hello world."""

from __future__ import annotations

from typing import Any

from agentomatic import AgentManifest

manifest = AgentManifest(
    name="hello",
    slug="agentomatic-hello",
    description="A friendly hello-world agent that echoes your query.",
    intent_keywords=["hello", "greet", "hi"],
    version="1.0.0",
)


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Simple echo node function."""
    from .graph import get_graph

    return await get_graph().ainvoke(state)
