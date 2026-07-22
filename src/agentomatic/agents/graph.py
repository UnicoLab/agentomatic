"""Internal graph runtime — lightweight, no LangGraph dependency.

Provides:
- ``GraphNode`` — a callable node in the graph
- ``AgentGraph`` — the execution runtime (invoke, validate, visualize)

The graph is intentionally simple for MVP. Complex orchestration
(streaming, checkpoints, parallel) can be handled by adapting to
LangGraph via ``BaseGraphAgent.as_langgraph()``.
"""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any, Generic

from loguru import logger

from .types import StateT, TraceEvent

# Sentinel for "end of graph"
END = "__END__"


def _run_coro_sync(coro: Any) -> Any:
    """Run *coro* to completion from a sync caller.

    Used by :meth:`AgentGraph.invoke` so async node handlers work under the
    sync ``transform`` / ``evaluate`` / ``fit`` path.

    Uses a persistent thread-local loop (not ``asyncio.run``) so LangChain /
    OpenAI async HTTP clients survive across fit → evaluate calls.
    """
    from agentomatic.async_utils import run_sync

    return run_sync(coro)


# ---------------------------------------------------------------------------
# Graph Node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphNode(Generic[StateT]):
    """A single node in the agent graph.

    Attributes:
        name: Unique name of this node.
        handler: Callable that takes state and returns state.
        description: Optional human-readable description.
        metadata: Arbitrary metadata for introspection.
    """

    name: str
    handler: Callable[[StateT], StateT]
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __call__(self, state: StateT) -> StateT:
        """Invoke the node handler."""
        return self.handler(state)


# ---------------------------------------------------------------------------
# Agent Graph
# ---------------------------------------------------------------------------


@dataclass
class AgentGraph(Generic[StateT]):
    """Lightweight graph runtime for agent execution.

    Supports linear and conditional edges. Nodes are executed
    sequentially from entrypoint to finish, following edges.

    Example::

        graph = AgentGraph(
            nodes={"a": node_a, "b": node_b},
            edges={"a": "b"},
            entrypoint="a",
            finish="b",
        )
        final_state = graph.invoke(initial_state)
    """

    nodes: dict[str, GraphNode[StateT]] = field(
        default_factory=dict,
    )
    edges: dict[str, str | Callable[..., str]] = field(
        default_factory=dict,
    )
    entrypoint: str = ""
    finish: str = ""

    # Runtime trace (populated during invoke)
    _last_trace: list[TraceEvent] = field(
        default_factory=list,
        repr=False,
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate graph consistency.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []

        if not self.entrypoint:
            errors.append("No entrypoint defined")
        elif self.entrypoint not in self.nodes:
            errors.append(f"Entrypoint '{self.entrypoint}' not in nodes")

        if not self.finish:
            errors.append("No finish node defined")
        elif self.finish not in self.nodes:
            errors.append(f"Finish node '{self.finish}' not in nodes")

        # Check edges reference existing nodes
        for source, target in self.edges.items():
            if source not in self.nodes:
                errors.append(f"Edge source '{source}' not in nodes")
            if isinstance(target, str) and target != END:
                if target not in self.nodes:
                    errors.append(f"Edge target '{target}' not in nodes")

        # Check path exists from entrypoint to finish
        if not errors and self.entrypoint and self.finish:
            if not self._has_path(self.entrypoint, self.finish):
                errors.append(f"No path from '{self.entrypoint}' to '{self.finish}'")

        return errors

    def _has_path(self, start: str, end: str) -> bool:
        """Check if a path exists between two nodes."""
        visited: set[str] = set()
        queue: deque[str] = deque([start])

        while queue:
            current = queue.popleft()
            if current == end:
                return True
            if current in visited:
                continue
            visited.add(current)

            target = self.edges.get(current)
            if target is None:
                continue
            if isinstance(target, str):
                if target != END:
                    queue.append(target)
            else:
                # Conditional edge — we can't statically resolve,
                # so assume all reachable targets
                # (conditional edges store possible values
                # in metadata or we just assume path exists)
                return True

        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def invoke(self, state: StateT) -> StateT:
        """Execute the graph synchronously.

        Args:
            state: Initial state.

        Returns:
            Final state after graph execution.

        Raises:
            ValueError: If graph is invalid.
            RuntimeError: If a node execution fails.
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        self._last_trace = []
        current_node_name = self.entrypoint

        while current_node_name and current_node_name != END:
            node = self.nodes[current_node_name]
            trace = TraceEvent(node_name=current_node_name)

            try:
                logger.debug(f"🔄 Executing node: {current_node_name}")
                if inspect.iscoroutinefunction(node.handler):
                    result = _run_coro_sync(node.handler(state))
                else:
                    result = node(state)
                    if inspect.iscoroutine(result):
                        result = _run_coro_sync(result)
                if result is not None:
                    state = result
                trace.finish(status="success")
            except Exception as exc:
                trace.finish(status="error", error=str(exc))
                self._last_trace.append(trace)
                raise RuntimeError(f"Node '{current_node_name}' failed: {exc}") from exc

            self._last_trace.append(trace)

            # Check if we've reached the finish node
            if current_node_name == self.finish:
                break

            # Follow edge
            edge = self.edges.get(current_node_name)
            if edge is None:
                break
            elif isinstance(edge, str):
                current_node_name = edge
            else:
                # Conditional edge — call the function
                next_name = edge(state)
                if inspect.iscoroutine(next_name):
                    next_name = _run_coro_sync(next_name)
                if next_name == END:
                    break
                current_node_name = next_name

        return state

    async def ainvoke(self, state: StateT) -> StateT:
        """Execute the graph asynchronously.

        Async node handlers are awaited; sync handlers are
        called directly.

        Args:
            state: Initial state.

        Returns:
            Final state after graph execution.
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        self._last_trace = []
        current_node_name = self.entrypoint

        while current_node_name and current_node_name != END:
            node = self.nodes[current_node_name]
            trace = TraceEvent(node_name=current_node_name)

            try:
                logger.debug(f"🔄 Executing node: {current_node_name}")
                if inspect.iscoroutinefunction(node.handler):
                    result = await node.handler(state)
                else:
                    result = node(state)
                if result is not None:
                    state = result
                trace.finish(status="success")
            except Exception as exc:
                trace.finish(status="error", error=str(exc))
                self._last_trace.append(trace)
                raise RuntimeError(f"Node '{current_node_name}' failed: {exc}") from exc

            self._last_trace.append(trace)

            if current_node_name == self.finish:
                break

            edge = self.edges.get(current_node_name)
            if edge is None:
                break
            elif isinstance(edge, str):
                current_node_name = edge
            else:
                result_name = edge(state)
                if inspect.iscoroutine(result_name):
                    result_name = await result_name
                if result_name == END:
                    break
                current_node_name = result_name

        return state

    async def astream(self, state: StateT) -> AsyncGenerator[dict[str, Any], None]:
        """Stream the execution of the graph asynchronously.

        Yields state updates after each node executes, in a format
        compatible with LangGraph (``{node_name: state_dict}``).

        Args:
            state: Initial state.

        Yields:
            Dictionary containing the node name and its output state.
        """
        import copy

        def _state_to_dict(s: Any) -> dict[str, Any]:
            if hasattr(s, "model_dump"):
                return s.model_dump()
            if hasattr(s, "__dict__"):
                return vars(s)
            return dict(s) if isinstance(s, dict) else {}

        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        self._last_trace = []
        current_node_name = self.entrypoint

        while current_node_name and current_node_name != END:
            node = self.nodes[current_node_name]
            trace = TraceEvent(node_name=current_node_name)

            try:
                logger.debug(f"🔄 Executing node: {current_node_name}")
                if inspect.iscoroutinefunction(node.handler):
                    result = await node.handler(state)
                else:
                    result = node(state)
                if result is not None:
                    state = result
                trace.finish(status="success")
            except Exception as exc:
                trace.finish(status="error", error=str(exc))
                self._last_trace.append(trace)
                raise RuntimeError(f"Node '{current_node_name}' failed: {exc}") from exc

            self._last_trace.append(trace)

            # Yield the state after this node finishes
            yield {current_node_name: copy.deepcopy(_state_to_dict(state))}

            if current_node_name == self.finish:
                break

            edge = self.edges.get(current_node_name)
            if edge is None:
                break
            elif isinstance(edge, str):
                current_node_name = edge
            else:
                result_name = edge(state)
                if inspect.iscoroutine(result_name):
                    result_name = await result_name
                if result_name == END:
                    break
                current_node_name = result_name

    async def astream_studio_events(
        self, state: StateT, run_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream execution emitting fine-grained events for Agentomatic Studio.

        Args:
            state: Initial state.
            run_id: The run ID for the events.

        Yields:
            Event dictionaries matching StudioRunEvent format.
        """
        import copy
        import time
        from datetime import UTC, datetime

        def _now_iso() -> str:
            return datetime.now(UTC).isoformat()

        def _state_to_dict(s: Any) -> dict[str, Any]:
            if hasattr(s, "model_dump"):
                return s.model_dump()
            if hasattr(s, "__dict__"):
                return vars(s)
            return dict(s) if isinstance(s, dict) else {}

        yield {
            "event": "run_start",
            "run_id": run_id,
            "timestamp": _now_iso(),
            "data": {"input": copy.deepcopy(_state_to_dict(state))},
        }

        errors = self.validate()
        if errors:
            yield {
                "event": "run_error",
                "run_id": run_id,
                "timestamp": _now_iso(),
                "data": {"error": f"Invalid graph: {'; '.join(errors)}"},
            }
            return

        self._last_trace = []
        current_node_name = self.entrypoint

        while current_node_name and current_node_name != END:
            node = self.nodes[current_node_name]
            trace = TraceEvent(node_name=current_node_name)
            t0 = time.perf_counter()

            yield {
                "event": "node_start",
                "run_id": run_id,
                "node": current_node_name,
                "timestamp": _now_iso(),
                "data": {},
            }

            try:
                if inspect.iscoroutinefunction(node.handler):
                    result = await node.handler(state)
                else:
                    result = node(state)
                if result is not None:
                    state = result
                trace.finish(status="success")
            except Exception as exc:
                trace.finish(status="error", error=str(exc))
                self._last_trace.append(trace)
                yield {
                    "event": "run_error",
                    "run_id": run_id,
                    "node": current_node_name,
                    "timestamp": _now_iso(),
                    "data": {"error": str(exc)},
                }
                raise RuntimeError(f"Node '{current_node_name}' failed: {exc}") from exc

            self._last_trace.append(trace)
            duration_ms = (time.perf_counter() - t0) * 1000

            state_dict = _state_to_dict(state)

            yield {
                "event": "node_end",
                "run_id": run_id,
                "node": current_node_name,
                "timestamp": _now_iso(),
                "duration_ms": duration_ms,
                "data": {"output": copy.deepcopy(state_dict)},
            }

            yield {
                "event": "state_update",
                "run_id": run_id,
                "timestamp": _now_iso(),
                "data": copy.deepcopy(state_dict),
            }

            if current_node_name == self.finish:
                break

            edge = self.edges.get(current_node_name)
            if edge is None:
                break
            elif isinstance(edge, str):
                current_node_name = edge
            else:
                result_name = edge(state)
                if inspect.iscoroutine(result_name):
                    result_name = await result_name
                if result_name == END:
                    break
                current_node_name = result_name

        yield {
            "event": "run_complete",
            "run_id": run_id,
            "timestamp": _now_iso(),
            "data": {"output": copy.deepcopy(_state_to_dict(state))},
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def last_trace(self) -> list[TraceEvent]:
        """Return the trace from the last invocation."""
        return list(self._last_trace)

    @property
    def node_names(self) -> list[str]:
        """Return all node names."""
        return list(self.nodes.keys())

    def visualize(self) -> str:
        """Generate a Mermaid diagram of the graph.

        Returns:
            Mermaid-syntax string.
        """
        lines = ["graph TD"]
        lines.append(f'    START(["▶ Start"]) --> {self.entrypoint}')

        for name, node in self.nodes.items():
            desc = node.description or name
            lines.append(f'    {name}["{desc}"]')

        for source, target in self.edges.items():
            if isinstance(target, str):
                if target == END:
                    lines.append(f"    {source} --> DONE")
                else:
                    lines.append(f"    {source} --> {target}")
            else:
                # Conditional edge
                lines.append(f'    {source} -->|"conditional"| {source}_router')
                lines.append(f'    {source}_router{{"route?"}}')

        finish_label = "✅ Done"
        lines.append(f'    DONE(["{finish_label}"])')
        if self.finish and self.finish not in self.edges:
            lines.append(f"    {self.finish} --> DONE")

        return "\n".join(lines)
