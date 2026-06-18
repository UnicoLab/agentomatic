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
    from agentomatic.studio.adapter import StudioAdapter


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class RunTracker:
    """Track agent execution as a series of events for debugging.

    Maintains an in-memory store of :class:`StudioRunInfo` objects and
    provides an async generator (``execute_with_adapter``) that yields
    SSE-formatted events during agent execution.

    This class is **framework-agnostic** — all framework-specific logic
    is delegated to the :class:`~agentomatic.studio.adapter.StudioAdapter`.
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
    # Streaming execution (adapter-based)
    # ------------------------------------------------------------------

    async def execute_with_adapter(
        self,
        adapter: StudioAdapter,
        state: dict[str, Any],
        run_id: str,
        thread_id: str | None = None,
        checkpoint_id: str | None = None,
        breakpoints: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute an agent via its adapter with full event tracking.

        Delegates all framework-specific logic to the provided
        :class:`~agentomatic.studio.adapter.StudioAdapter`, wrapping
        the execution with run lifecycle events.

        Args:
            adapter: The Studio adapter for the target agent.
            state: Initial agent state dict.
            run_id: The run ID to attach events to.
            thread_id: Optional thread identifier for config.
            checkpoint_id: Optional checkpoint to resume from.
            breakpoints: Optional node names to pause before.

        Yields:
            SSE-formatted strings (``data: ...\\n\\n``).
        """
        run = self._runs.get(run_id)
        if not run:
            return

        run.status = "running"
        start_time = time.monotonic()

        # Build config for the adapter
        config: dict[str, Any] = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        # -- Run start event --
        start_event = StudioRunEvent(
            event="run_start",
            run_id=run_id,
            timestamp=_now_iso(),
            data={
                "agent": run.agent_name,
                "input": run.input,
                "capabilities": adapter.capabilities,
            },
        )
        self.add_event(run_id, start_event)
        yield f"data: {start_event.model_dump_json()}\n\n"

        try:
            async for event in adapter.stream_execution(
                state, config, breakpoints, checkpoint_id
            ):
                # Stamp the run_id onto adapter events
                event.run_id = run_id
                self.add_event(run_id, event)
                yield f"data: {event.model_dump_json()}\n\n"

            # -- Run complete --
            duration = (time.monotonic() - start_time) * 1000
            self.complete_run(run_id, state, duration)

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
