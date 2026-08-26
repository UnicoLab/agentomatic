"""Writer agent that verifies a scoped live HTTP connection."""

from __future__ import annotations

from typing import Any

from agentomatic import AgentManifest
from agentomatic.connections import get_connections

manifest = AgentManifest(
    name="writer", slug="writer", framework="custom", description="Writing fixture"
)


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Use the scoped oMLX HTTP connection before producing a result."""
    probe = await get_connections("writer").http("omlx").get("/models")
    if not probe.ok:
        raise RuntimeError(f"oMLX connection probe failed: {probe.status_code}")
    query = str(state.get("current_query") or state.get("query") or "")
    return {"response": f"writer:{query}", "agent_type": "writer", "connection_probe": "ok"}
