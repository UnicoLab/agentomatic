"""The single source of truth for an agent's A2A card.

The card was built in two places — ``GET /api/v1/{agent}/card`` and the
platform's ``/.well-known/agent.json`` — and they had drifted. The per-agent
card advertised capabilities and a streaming endpoint; the well-known one, the
canonical A2A discovery document, listed only ``invoke`` and ``chat`` and no
capabilities at all. A client doing proper discovery therefore learned *less*
about an agent than one that guessed the per-agent URL.

Both now render through :func:`build_agent_card`, so a capability can only be
advertised in one place: here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentomatic.core.manifest import AgentManifest


def build_agent_card(
    manifest: AgentManifest,
    agent_name: str,
    api_prefix: str,
    *,
    supports_tasks: bool,
) -> dict[str, Any]:
    """Render one agent's A2A card.

    Capabilities are reported from what the deployment actually wired up
    rather than asserted unconditionally: a card that claims a capability the
    platform lacks sends A2A clients down a path that then answers 501.

    Args:
        manifest: The agent's manifest.
        agent_name: Name the agent is mounted under.
        api_prefix: API prefix the endpoints hang off (e.g. ``/api/v1``).
        supports_tasks: Whether the task subsystem is configured. Resumable
            streams, task history and push callbacks all depend on it.

    Returns:
        The agent card.
    """
    base = f"{api_prefix}/{agent_name}"
    endpoints: dict[str, str] = {
        "invoke": f"{base}/invoke",
        "chat": f"{base}/chat",
        "stream": f"{base}/invoke/stream",
        "stream_replay": f"{base}/invoke/stream/{{stream_id}}",
        "health": f"{base}/health",
        "card": f"{base}/card",
    }
    if supports_tasks:
        endpoints["a2a_tasks"] = f"{base}/a2a/tasks"
        endpoints["a2a_events"] = f"{base}/a2a/tasks/{{task_id}}/events"

    return {
        "name": manifest.slug,
        "description": manifest.description,
        "version": manifest.version,
        "framework": manifest.framework,
        "capabilities": {
            "streaming": True,
            "chat": True,
            "invoke": True,
            "a2a": True,
            # These three are the task subsystem's, not the agent's.
            "stateTransitionHistory": supports_tasks,
            "pushNotifications": supports_tasks,
            "resumableStreams": supports_tasks,
        },
        "endpoints": endpoints,
        "metadata": manifest.metadata,
    }
