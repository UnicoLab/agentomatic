"""Weather Agent — demonstrates ALL agentomatic overwrite options."""

from __future__ import annotations

from typing import Any

from agentomatic import AgentManifest

# ============================================================================
# 1. MANIFEST (required) — The agent's identity card
# ============================================================================
manifest = AgentManifest(
    name="weather",
    slug="demo-weather-agent",
    description="Weather forecasting agent with full customization demo",
    intent_keywords=["weather", "forecast", "temperature", "rain"],
    version="2.0.0",
    is_subagent=True,
    framework="langgraph",  # or 'langchain' or 'custom'
    metadata={
        "department": "demo",
        "author": "agentomatic",
        "capabilities": ["forecast", "alerts", "history"],
    },
)


# ============================================================================
# 2. NODE FUNCTION (recommended) — Entry point for the agent
# ============================================================================
async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Main entry point. Delegates to the graph."""
    from .graph import get_graph

    return await get_graph().ainvoke(state)
