"""Root-schema agent used by the production Docker verification fixture."""

from __future__ import annotations

from typing import Any

from agentomatic import AgentManifest

manifest = AgentManifest(
    name="root_echo_agent",
    slug="root_echo_agent",
    framework="custom",
    description="Echoes a Pydantic RootModel request through native, task, and Studio routes.",
)


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Return the root value that Agentomatic preserved in mapping state."""
    value = state["__root__"]
    return {"response": f"root:{value}", "root": value, "agent_type": "root_echo_agent"}
