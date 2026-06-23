"""Observability mixin — trace recording and retrieval.

Manages a history of execution traces, where each trace is a
list of ``TraceEvent`` objects from a single graph invocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TraceEvent


class ObservabilityMixin:
    """Mixin for recording and accessing execution traces.

    Each invocation produces a trace (list of ``TraceEvent``).
    This mixin stores a rolling history of traces.

    Attributes:
        _traces: Internal list of trace histories.

    Example::

        agent.invoke_graph(state)
        trace = agent.get_last_trace()
        for event in trace:
            print(f"{event.node_name}: {event.duration_ms}ms")
    """

    _traces: list[list[TraceEvent]]

    def get_last_trace(self) -> list[TraceEvent]:
        """Return the most recent execution trace.

        Returns:
            List of ``TraceEvent`` from the last invocation,
            or an empty list if no traces exist.
        """
        traces = self._get_traces()
        if not traces:
            return []
        return list(traces[-1])

    def get_trace_history(self) -> list[list[TraceEvent]]:
        """Return the full history of execution traces.

        Returns:
            List of traces, each being a list of ``TraceEvent``.
        """
        return list(self._get_traces())

    def _record_trace(self, trace: list[TraceEvent]) -> None:
        """Record a new execution trace.

        Args:
            trace: List of ``TraceEvent`` from a graph invocation.
        """
        traces = self._get_traces()
        traces.append(list(trace))

    def _get_traces(self) -> list[list[TraceEvent]]:
        """Lazily initialize and return the traces list.

        Returns:
            The internal ``_traces`` list.
        """
        if not hasattr(self, "_traces") or self._traces is None:
            self._traces = []
        return self._traces
