"""Track agent execution as a series of events for debugging and SSE streaming."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.studio.models import StudioRunEvent, StudioRunInfo

if TYPE_CHECKING:
    from agentomatic.core.manifest import RegisteredAgent


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class RunTracker:
    """Track agent execution as a series of events for debugging.

    Maintains an in-memory store of :class:`StudioRunInfo` objects and
    provides an async generator (``execute_and_stream``) that yields
    SSE-formatted events during agent execution.
    """

    def __init__(self) -> None:
        self._runs: dict[str, StudioRunInfo] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_run(
        self,
        agent_name: str,
        thread_id: str | None,
        request_data: dict[str, Any],
    ) -> StudioRunInfo:
        """Create a new tracked run.

        Args:
            agent_name: Name of the agent being invoked.
            thread_id: Optional thread ID for the conversation.
            request_data: Serialized request payload.

        Returns:
            A new :class:`StudioRunInfo` in ``'pending'`` status.
        """
        run = StudioRunInfo(
            id=f"run_{uuid.uuid4().hex[:12]}",
            agent_name=agent_name,
            thread_id=thread_id,
            status="pending",
            created_at=_now_iso(),
            input=request_data,
        )
        self._runs[run.id] = run
        logger.debug(f"Studio run created: {run.id} for agent={agent_name}")
        return run

    def get_run(self, run_id: str) -> StudioRunInfo | None:
        """Retrieve a run by its ID."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        agent_name: str | None = None,
        limit: int = 50,
    ) -> list[StudioRunInfo]:
        """List recent runs, optionally filtered by agent.

        Args:
            agent_name: If provided, only return runs for this agent.
            limit: Maximum number of runs to return.

        Returns:
            Runs sorted newest-first, capped at *limit*.
        """
        runs = list(self._runs.values())
        if agent_name:
            runs = [r for r in runs if r.agent_name == agent_name]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)[:limit]

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def add_event(self, run_id: str, event: StudioRunEvent) -> None:
        """Append an event to a tracked run."""
        run = self._runs.get(run_id)
        if run:
            run.events.append(event)

    def complete_run(
        self,
        run_id: str,
        output: dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Mark a run as completed successfully."""
        run = self._runs.get(run_id)
        if run:
            run.status = "completed"
            run.completed_at = _now_iso()
            run.output = output
            run.duration_ms = round(duration_ms, 2)

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed."""
        run = self._runs.get(run_id)
        if run:
            run.status = "failed"
            run.completed_at = _now_iso()
            run.error = error

    # ------------------------------------------------------------------
    # Streaming execution
    # ------------------------------------------------------------------

    async def execute_and_stream(
        self,
        agent: RegisteredAgent,
        state: dict[str, Any],
        run_id: str,
        thread_id: str | None = None,
        checkpoint_id: str | None = None,
        breakpoints: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute an agent with full event tracking, yielding SSE frames.

        For LangGraph agents this uses ``graph.astream_events()``.
        For simple ``node_fn`` agents it wraps execution with start/end events.

        Args:
            agent: The registered agent to execute.
            state: Initial agent state dict.
            run_id: The run ID to attach events to.

        Yields:
            SSE-formatted strings (``data: ...\\n\\n``).
        """
        run = self._runs.get(run_id)
        if not run:
            return

        run.status = "running"
        start_time = time.monotonic()

        # -- Run start event --
        start_event = StudioRunEvent(
            event="run_start",
            run_id=run_id,
            timestamp=_now_iso(),
            data={"agent": run.agent_name, "input": run.input},
        )
        self.add_event(run_id, start_event)
        yield f"data: {start_event.model_dump_json()}\n\n"

        try:
            if agent.graph_fn:
                async for sse_frame in self._stream_graph(
                    agent, state, run_id, start_time, thread_id, checkpoint_id, breakpoints
                ):
                    yield sse_frame
            elif agent.node_fn:
                async for sse_frame in self._stream_node_fn(agent, state, run_id, start_time):
                    yield sse_frame
            else:
                raise RuntimeError(f"Agent '{agent.name}' has no callable (graph_fn or node_fn)")

            # -- Run complete event --
            run = self._runs.get(run_id)
            complete_event = StudioRunEvent(
                event="run_complete",
                run_id=run_id,
                timestamp=_now_iso(),
                data={"output": run.output if run else {}},
                duration_ms=run.duration_ms if run else None,
            )
            self.add_event(run_id, complete_event)
            yield f"data: {complete_event.model_dump_json()}\n\n"

        except Exception as exc:
            duration = (time.monotonic() - start_time) * 1000
            self.fail_run(run_id, str(exc))
            logger.error(f"Studio run {run_id} failed: {exc}")

            error_event = StudioRunEvent(
                event="run_error",
                run_id=run_id,
                timestamp=_now_iso(),
                data={"error": str(exc), "type": type(exc).__name__},
                duration_ms=round(duration, 2),
            )
            self.add_event(run_id, error_event)
            yield f"data: {error_event.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Internal — LangGraph streaming
    # ------------------------------------------------------------------

    async def _stream_graph(
        self,
        agent: RegisteredAgent,
        state: dict[str, Any],
        run_id: str,
        start_time: float,
        thread_id: str | None = None,
        checkpoint_id: str | None = None,
        breakpoints: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream events from a LangGraph compiled graph."""
        graph = agent.graph_fn()  # type: ignore[misc]

        config: dict[str, Any] = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
            if checkpoint_id:
                config["configurable"]["checkpoint_id"] = checkpoint_id

        if breakpoints:
            try:
                graph.interrupt_before_nodes = frozenset(breakpoints)
            except Exception:
                pass

        async for lg_event in graph.astream_events(state, config=config, version="v2"):
            studio_event = self._map_langgraph_event(run_id, lg_event)
            if studio_event:
                self.add_event(run_id, studio_event)
                yield f"data: {studio_event.model_dump_json()}\n\n"

        duration = (time.monotonic() - start_time) * 1000
        self.complete_run(run_id, state, duration)

    # ------------------------------------------------------------------
    # Internal — node_fn streaming
    # ------------------------------------------------------------------

    async def _stream_node_fn(
        self,
        agent: RegisteredAgent,
        state: dict[str, Any],
        run_id: str,
        start_time: float,
    ) -> AsyncGenerator[str, None]:
        """Stream events for a simple node_fn agent."""
        # Node start
        node_start = StudioRunEvent(
            event="node_start",
            run_id=run_id,
            timestamp=_now_iso(),
            node=agent.name,
        )
        self.add_event(run_id, node_start)
        yield f"data: {node_start.model_dump_json()}\n\n"

        # Execute
        result = await agent.node_fn(state)  # type: ignore[misc]

        # Node end
        node_end = StudioRunEvent(
            event="node_end",
            run_id=run_id,
            timestamp=_now_iso(),
            node=agent.name,
            data={"output": result if isinstance(result, dict) else {"result": str(result)}},
            duration_ms=round((time.monotonic() - start_time) * 1000, 2),
        )
        self.add_event(run_id, node_end)
        yield f"data: {node_end.model_dump_json()}\n\n"

        duration = (time.monotonic() - start_time) * 1000
        output = result if isinstance(result, dict) else {"response": str(result)}
        self.complete_run(run_id, output, duration)

    # ------------------------------------------------------------------
    # Internal — LangGraph event mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_langgraph_event(
        run_id: str,
        lg_event: dict[str, Any],
    ) -> StudioRunEvent | None:
        """Map a LangGraph streaming event to a :class:`StudioRunEvent`.

        Only a subset of LangGraph events are mapped — unmapped events
        return ``None`` and are silently skipped.
        """
        event_type = lg_event.get("event", "")
        name = lg_event.get("name", "")
        data = lg_event.get("data", {})

        if event_type == "on_chain_start" and name != "LangGraph":
            return StudioRunEvent(
                event="node_start",
                run_id=run_id,
                timestamp=_now_iso(),
                node=name,
                data={"tags": lg_event.get("tags", [])},
            )

        if event_type == "on_chain_end" and name != "LangGraph":
            output = data.get("output", {})
            # Ensure output is JSON-serializable
            if not isinstance(output, (dict, list, str, int, float, bool, type(None))):
                output = str(output)
            return StudioRunEvent(
                event="node_end",
                run_id=run_id,
                timestamp=_now_iso(),
                node=name,
                data={"output": output},
            )

        if event_type == "on_chat_model_stream":
            chunk = data.get("chunk", {})
            content = ""
            if hasattr(chunk, "content"):
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get("content", "")
            if content:
                return StudioRunEvent(
                    event="message_chunk",
                    run_id=run_id,
                    timestamp=_now_iso(),
                    node=name,
                    data={"content": content},
                )

        if event_type == "on_tool_start":
            tool_input = data.get("input", {})
            if not isinstance(tool_input, (dict, list, str, int, float, bool, type(None))):
                tool_input = str(tool_input)
            return StudioRunEvent(
                event="node_start",
                run_id=run_id,
                timestamp=_now_iso(),
                node=f"tool:{name}",
                data={"tool_input": tool_input},
            )

        if event_type == "on_tool_end":
            tool_output = data.get("output", "")
            if not isinstance(tool_output, (dict, list, str, int, float, bool, type(None))):
                tool_output = str(tool_output)
            return StudioRunEvent(
                event="node_end",
                run_id=run_id,
                timestamp=_now_iso(),
                node=f"tool:{name}",
                data={"tool_output": tool_output},
            )

        return None  # Skip unmapped events
