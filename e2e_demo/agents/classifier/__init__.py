"""Structured-input agent used to verify Studio does not require `query`."""

from __future__ import annotations

from typing import Any

from agentomatic import AgentManifest

manifest = AgentManifest(
    name="classifier",
    slug="classifier",
    framework="custom",
    description="Classifies structured deployment input without a query field.",
)


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Echo the live typed field to make schema transport assertions exact."""
    return {
        "label": str(state["label"]),
        "priority": int(state.get("priority", 0)),
        "response": f"classified:{state['label']}",
    }
