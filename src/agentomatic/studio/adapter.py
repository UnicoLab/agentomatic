"""Universal adapter protocol for Agentomatic Studio.

Any agent framework can implement this protocol to unlock the full
Studio debugging experience. See :class:`LangGraphAdapter` for the
reference implementation and :class:`GenericAdapter` for the fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentomatic.studio.models import (
        StudioCheckpoint,
        StudioGraphTopology,
        StudioRunEvent,
        StudioStateSnapshot,
    )


class StudioAdapter(ABC):
    """Abstract base class defining the universal Studio interface.

    Every adapter must implement these methods. Methods that are not
    supported by the underlying framework should return sensible
    defaults (empty lists, ``None``, etc.) rather than raising.

    Attributes:
        agent_name: The machine name of the agent this adapter wraps.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        """Return the list of Studio capabilities this adapter supports.

        Possible values: ``'graph'``, ``'streaming'``, ``'checkpoints'``,
        ``'state'``, ``'breakpoints'``, ``'hitl'``, ``'traces'``.
        """
        ...

    @property
    def supports_graph(self) -> bool:
        """Whether this adapter can provide a real execution graph."""
        return "graph" in self.capabilities

    @property
    def supports_checkpoints(self) -> bool:
        """Whether this adapter supports checkpoint-based time-travel."""
        return "checkpoints" in self.capabilities

    @property
    def supports_state_mutation(self) -> bool:
        """Whether this adapter supports live state editing."""
        return "state" in self.capabilities

    @property
    def supports_breakpoints(self) -> bool:
        """Whether this adapter supports conditional breakpoints."""
        return "breakpoints" in self.capabilities

    # ------------------------------------------------------------------
    # Graph topology
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_graph(self) -> StudioGraphTopology:
        """Return the agent's execution graph topology.

        Returns:
            A :class:`StudioGraphTopology` describing the nodes and edges.
            For adapters that don't support real graphs, return a synthetic
            linear topology.
        """
        ...

    # ------------------------------------------------------------------
    # Execution streaming
    # ------------------------------------------------------------------

    @abstractmethod
    async def stream_execution(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
        breakpoints: list[str] | None = None,
        checkpoint_id: str | None = None,
    ) -> AsyncGenerator[StudioRunEvent, None]:
        """Execute the agent and yield events as they occur.

        Args:
            state: Initial state dict to pass to the agent.
            config: Optional execution configuration (e.g. thread_id).
            breakpoints: Optional list of node names to pause before.
            checkpoint_id: Optional checkpoint to resume from.

        Yields:
            :class:`StudioRunEvent` instances for each notable step.
        """
        ...

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_state(self, thread_id: str) -> StudioStateSnapshot | None:
        """Retrieve the latest state for a thread.

        Args:
            thread_id: The conversation thread identifier.

        Returns:
            A :class:`StudioStateSnapshot`, or ``None`` if state
            inspection is not supported.
        """
        ...

    @abstractmethod
    async def update_state(
        self,
        thread_id: str,
        updates: dict[str, Any],
    ) -> StudioStateSnapshot | None:
        """Apply a partial state update to a thread.

        Args:
            thread_id: The conversation thread identifier.
            updates: Key-value pairs to merge into the current state.

        Returns:
            The updated :class:`StudioStateSnapshot`, or ``None`` if
            state mutation is not supported.
        """
        ...

    # ------------------------------------------------------------------
    # Checkpoint history
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_history(self, thread_id: str) -> list[StudioCheckpoint]:
        """Return the checkpoint history for a thread.

        Args:
            thread_id: The conversation thread identifier.

        Returns:
            A list of :class:`StudioCheckpoint` objects, newest first.
            Returns an empty list if checkpoints are not supported.
        """
        ...
