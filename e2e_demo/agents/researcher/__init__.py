"""Deterministic research agent used by the production Docker matrix."""

from __future__ import annotations

from typing import Any

from agentomatic import AgentManifest

manifest = AgentManifest(
    name="researcher", slug="researcher", framework="custom", description="Research fixture"
)


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Return a stable research result for delegation and pipeline assertions."""
    query = str(state.get("current_query") or state.get("query") or "")
    return {"response": f"research:{query}", "agent_type": "researcher"}
