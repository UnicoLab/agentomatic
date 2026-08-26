"""Swarm orchestration for multi-agent coordination.

Supports handoff-based delegation (langgraph-swarm), supervisor patterns,
and round-robin distribution across registered agents.
"""

from __future__ import annotations

import inspect
from threading import Lock
from typing import Any

from loguru import logger

_VALID_PATTERNS = frozenset({"handoff", "supervisor", "round_robin"})


class SwarmOrchestrator:
    """Manages swarm-style multi-agent coordination.

    Supports:
    - Handoff-based delegation (langgraph-swarm)
    - Supervisor patterns (one agent coordinates others)
    - Round-robin distribution

    Args:
        platform_url: Base URL of the platform API.

    Example::

        orchestrator = SwarmOrchestrator()
        orchestrator.register_agent("researcher", researcher_graph)
        orchestrator.register_agent("writer", writer_graph)
        swarm = orchestrator.create_swarm(pattern="handoff")
    """

    def __init__(
        self,
        *,
        platform_url: str = "http://localhost:8000",
    ) -> None:
        self._platform_url = platform_url
        self._agents: dict[str, Any] = {}

    @property
    def registered_agents(self) -> list[str]:
        """Return names of all registered agents."""
        return list(self._agents.keys())

    def register_agent(self, name: str, agent: Any) -> None:
        """Register a compiled agent graph for swarm participation.

        Args:
            name: Unique name for the agent within this swarm.
            agent: A compiled LangGraph agent (or compatible runnable).

        Raises:
            ValueError: If an agent with the same name is already registered.
        """
        if name in self._agents:
            raise ValueError(
                f"Agent '{name}' is already registered. "
                "Use a unique name or unregister the existing agent first."
            )
        self._agents[name] = agent
        logger.info("Registered agent '{}' in swarm orchestrator", name)

    def unregister_agent(self, name: str) -> None:
        """Remove a registered agent from the swarm.

        Args:
            name: Name of the agent to remove.

        Raises:
            KeyError: If no agent with that name is registered.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered.")
        del self._agents[name]
        logger.info("Unregistered agent '{}' from swarm orchestrator", name)

    def create_swarm(
        self,
        agents: list[str] | None = None,
        *,
        pattern: str = "handoff",
    ) -> Any:
        """Create a swarm graph from registered agents.

        Args:
            agents: Agent names to include (None = all registered).
            pattern: Orchestration pattern
                ('handoff', 'supervisor', 'round_robin').

        Returns:
            A compiled graph implementing the swarm pattern.

        Raises:
            ValueError: If pattern is unknown or no agents are available.
            ImportError: If required dependencies are not installed.
        """
        if pattern not in _VALID_PATTERNS:
            raise ValueError(
                f"Unknown swarm pattern '{pattern}'. Valid patterns: {sorted(_VALID_PATTERNS)}"
            )

        selected = self._resolve_agents(agents)

        if pattern == "handoff":
            return self._create_handoff_swarm(selected)
        if pattern == "supervisor":
            return self._create_supervisor_swarm(selected)
        # pattern == "round_robin"
        return self._create_round_robin_swarm(selected)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_agents(self, names: list[str] | None) -> dict[str, Any]:
        """Resolve agent names to their registered graphs.

        Args:
            names: Specific names to resolve, or None for all.

        Returns:
            Dict mapping agent name to agent graph.

        Raises:
            ValueError: If no agents are registered or a name is not found.
        """
        if not self._agents:
            raise ValueError("No agents registered. Call register_agent() first.")

        if names is None:
            return dict(self._agents)

        missing = set(names) - set(self._agents)
        if missing:
            raise ValueError(
                f"Agents not registered: {sorted(missing)}. Available: {sorted(self._agents)}"
            )
        return {n: self._agents[n] for n in names}

    def _create_handoff_swarm(self, agents: dict[str, Any]) -> Any:
        """Create a handoff-based swarm using langgraph-swarm.

        Args:
            agents: Mapping of agent name to compiled graph.

        Returns:
            Compiled swarm graph.

        Raises:
            ImportError: If langgraph-swarm is not installed.
        """
        try:
            from langgraph_swarm import (
                create_swarm as _create_swarm,
            )
        except ImportError:
            raise ImportError(
                "langgraph-swarm is required for the 'handoff' pattern. "
                "Install it with: pip install langgraph-swarm"
            ) from None

        agent_list = list(agents.values())
        logger.info(
            "Creating handoff swarm with {} agents: {}",
            len(agents),
            sorted(agents.keys()),
        )
        return _create_swarm(agent_list, default_active_agent=agent_list[0].name).compile()

    def _create_supervisor_swarm(self, agents: dict[str, Any]) -> Any:
        """Create a supervisor-pattern swarm.

        The returned runnable dispatches to the agent selected by the input's
        ``swarm_agent``, ``route_to``, or ``agent`` key (in that order). When
        no routing hint is supplied, it uses the first registered agent.

        Args:
            agents: Mapping of agent name to compiled graph.

        Returns:
            A runnable supervisor swarm.
        """
        logger.info(
            "Creating supervisor swarm with {} agents: {}",
            len(agents),
            sorted(agents.keys()),
        )
        return _RunnableSwarm(pattern="supervisor", agents=agents)

    def _create_round_robin_swarm(self, agents: dict[str, Any]) -> Any:
        """Create a round-robin distribution swarm.

        The round-robin pattern cycles through agents in registration order.

        Args:
            agents: Mapping of agent name to compiled graph.

        Returns:
            A runnable round-robin swarm.
        """
        logger.info(
            "Creating round-robin swarm with {} agents: {}",
            len(agents),
            sorted(agents.keys()),
        )
        return _RunnableSwarm(pattern="round_robin", agents=agents)


class _RunnableSwarm:
    """Small runnable wrapper for the built-in swarm patterns.

    It deliberately accepts any LangGraph/LangChain-compatible runnable:
    objects exposing ``invoke`` / ``ainvoke`` as well as plain callables. The
    implementation is dependency-free, so teams that do not install
    ``langgraph-swarm`` still have production-safe supervisor and round-robin
    dispatching.

    Args:
        pattern: ``supervisor`` or ``round_robin``.
        agents: Mapping of stable agent names to compatible runnables.
    """

    def __init__(self, *, pattern: str, agents: dict[str, Any]) -> None:
        self.pattern = pattern
        self._agents = dict(agents)
        self.agents = list(self._agents)
        self._next_index = 0
        self._lock = Lock()

    def _select(self, input_data: Any) -> tuple[str, Any]:
        """Select the target runnable for one invocation."""
        if self.pattern == "round_robin":
            with self._lock:
                name = self.agents[self._next_index]
                self._next_index = (self._next_index + 1) % len(self.agents)
            return name, self._agents[name]

        selected = None
        if isinstance(input_data, dict):
            for key in ("swarm_agent", "route_to", "agent"):
                if input_data.get(key):
                    selected = str(input_data[key])
                    break
        name = selected or self.agents[0]
        agent = self._agents.get(name)
        if agent is None:
            raise ValueError(
                f"Supervisor selected unknown agent '{name}'. Available: {self.agents}"
            )
        return name, agent

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        """Synchronously dispatch one input to the selected swarm agent.

        Async-only runnables must be called through :meth:`ainvoke` so their
        coroutine is never leaked or run in an accidental event loop.
        """
        name, agent = self._select(input)
        call = getattr(agent, "invoke", agent)
        result = call(input, **kwargs)
        if inspect.isawaitable(result):
            raise TypeError(f"Swarm agent '{name}' is async-only; call swarm.ainvoke(...)")
        return result

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        """Asynchronously dispatch one input to the selected swarm agent."""
        _name, agent = self._select(input)
        call = getattr(agent, "ainvoke", None) or getattr(agent, "invoke", agent)
        result = call(input, **kwargs)
        return await result if inspect.isawaitable(result) else result

    def __call__(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        """Alias :meth:`invoke` for callable-style use."""
        return self.invoke(input, **kwargs)

    def __repr__(self) -> str:
        return f"RunnableSwarm(pattern={self.pattern!r}, agents={self.agents!r})"
