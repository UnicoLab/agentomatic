"""Runtime state for the control plane.

Holds soft, in-process operational state — disabled agents, maintenance
mode, and process start time — that the control API can inspect and mutate
without restarting the platform.
"""

from __future__ import annotations

import time


class ControlPlaneState:
    """Mutable operational state shared by the control API and middleware."""

    def __init__(self) -> None:
        self.started_at: float = time.time()
        self.maintenance_mode: bool = False
        self._disabled_agents: set[str] = set()

    @property
    def uptime_seconds(self) -> float:
        """Return the process uptime in seconds."""
        return time.time() - self.started_at

    @property
    def disabled_agents(self) -> set[str]:
        """Return the set of currently disabled agent names."""
        return set(self._disabled_agents)

    def disable_agent(self, name: str) -> None:
        """Mark an agent as disabled (requests will be rejected with 503)."""
        self._disabled_agents.add(name)

    def enable_agent(self, name: str) -> None:
        """Re-enable a previously disabled agent."""
        self._disabled_agents.discard(name)

    def is_agent_disabled(self, name: str) -> bool:
        """Return ``True`` if the agent is currently disabled."""
        return name in self._disabled_agents
