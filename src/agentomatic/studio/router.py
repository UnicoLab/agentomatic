"""FastAPI router for the Agentomatic Studio debug API."""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from agentomatic.studio.graph_inspector import GraphInspector
from agentomatic.studio.models import (
    StudioAgentInfo,
    StudioAgentSchemas,
    StudioCheckpoint,
    StudioGraphTopology,
    StudioRunInfo,
    StudioRunRequest,
    StudioServerInfo,
    StudioStateSnapshot,
    StudioStateUpdate,
)
from agentomatic.studio.run_tracker import RunTracker

if TYPE_CHECKING:
    from agentomatic.core.registry import AgentRegistry
    from agentomatic.storage.base import BaseStore


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def create_studio_router(
    registry: AgentRegistry,
    store: BaseStore | None = None,
    platform_title: str = "Agentomatic Platform",
    platform_version: str = "1.0.0",
) -> APIRouter:
    """Create the Studio debug API router.

    Provides endpoints for agent discovery, graph inspection, execution
    tracing (with SSE streaming), state inspection, and checkpoint
    browsing.

    Args:
        registry: The platform's agent registry.
        store: Optional storage backend for checkpoints and state.
        platform_title: Human-readable platform title.
        platform_version: Platform version string.

    Returns:
        A fully-wired :class:`~fastapi.APIRouter`.
    """
    router = APIRouter(prefix="/studio", tags=["Studio Debug API"])
    inspector = GraphInspector()
    tracker = RunTracker()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_agent(name: str):
        """Look up an agent by name or raise 404."""
        agent = registry.get(name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
        return agent

    # ==================================================================
    # Discovery endpoints
    # ==================================================================

    @router.get("/info", response_model=StudioServerInfo, summary="Platform info")
    async def get_info() -> StudioServerInfo:
        """Return platform-level metadata and capabilities."""
        capabilities = ["studio"]
        if store is not None:
            capabilities.append("storage")
        capabilities.append("streaming")
        return StudioServerInfo(
            version=platform_version,
            platform_title=platform_title,
            agent_count=registry.count,
            capabilities=capabilities,
        )

    @router.get("/agents", response_model=list[StudioAgentInfo], summary="List agents")
    async def list_agents() -> list[StudioAgentInfo]:
        """List all registered agents with their debugging capabilities."""
        infos: list[StudioAgentInfo] = []
        for _name, agent in registry.all().items():
            infos.append(
                StudioAgentInfo(
                    name=agent.name,
                    slug=agent.slug,
                    description=agent.manifest.description,
                    version=agent.manifest.version,
                    framework=agent.manifest.framework,
                    capabilities=inspector.get_capabilities(agent),
                    has_graph=agent.graph_fn is not None,
                    has_config=agent.config is not None,
                    has_prompts=agent.prompt_manager is not None,
                )
            )
        return infos

    # ==================================================================
    # Graph inspection endpoints
    # ==================================================================

    @router.get(
        "/agents/{name}/graph",
        response_model=StudioGraphTopology,
        summary="Get agent graph topology",
    )
    async def get_graph(name: str) -> StudioGraphTopology:
        """Return the execution graph topology for an agent."""
        agent = _resolve_agent(name)
        return inspector.inspect(agent)

    @router.get(
        "/agents/{name}/schemas",
        response_model=StudioAgentSchemas,
        summary="Get agent input/output schemas",
    )
    async def get_schemas(name: str) -> StudioAgentSchemas:
        """Return JSON schemas for agent input and output models.

        Attempts to discover custom schemas from the agent's ``schemas``
        module, falling back to the default platform request/response
        models.
        """
        agent = _resolve_agent(name)
        input_schema: dict[str, Any] = {}
        output_schema: dict[str, Any] = {}

        # Try to find custom schemas from the agent module
        if agent.module_path:
            try:
                schemas_mod = importlib.import_module(f"{agent.module_path}.schemas")
                title_camel = name.title().replace("_", "")

                # Input schema
                for candidate in [
                    "CustomInvokeRequest",
                    f"{title_camel}Request",
                    "AgentInvokeRequest",
                ]:
                    cls = getattr(schemas_mod, candidate, None)
                    if cls and hasattr(cls, "model_json_schema"):
                        input_schema = cls.model_json_schema()
                        break

                # Output schema
                for candidate in [
                    "CustomInvokeResponse",
                    f"{title_camel}Response",
                    "AgentInvokeResponse",
                ]:
                    cls = getattr(schemas_mod, candidate, None)
                    if cls and hasattr(cls, "model_json_schema"):
                        output_schema = cls.model_json_schema()
                        break
            except ImportError:
                pass

        # Fall back to default models
        if not input_schema:
            from agentomatic.core.router_factory import AgentInvokeRequest

            input_schema = AgentInvokeRequest.model_json_schema()
        if not output_schema:
            from agentomatic.core.router_factory import AgentInvokeResponse

            output_schema = AgentInvokeResponse.model_json_schema()

        return StudioAgentSchemas(
            input_schema=input_schema,
            output_schema=output_schema,
        )

    @router.get(
        "/agents/{name}/config",
        summary="Get agent configuration",
    )
    async def get_config(name: str) -> dict[str, Any]:
        """Return the agent's configuration object (if any)."""
        agent = _resolve_agent(name)
        if agent.config:
            if hasattr(agent.config, "model_dump"):
                return {"agent": name, "config": agent.config.model_dump()}
            return {"agent": name, "config": vars(agent.config)}
        return {"agent": name, "config": {}}

    # ==================================================================
    # Run endpoints
    # ==================================================================

    @router.post(
        "/agents/{name}/runs",
        response_model=StudioRunInfo,
        summary="Create and execute a run",
    )
    async def create_run(name: str, request: StudioRunRequest) -> StudioRunInfo:
        """Create a new run, execute the agent synchronously, and return the result."""
        agent = _resolve_agent(name)
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"

        run = tracker.create_run(
            agent_name=name,
            thread_id=thread_id,
            request_data=request.model_dump(),
        )

        state = _build_studio_state(request, thread_id)

        # Execute — consume the stream to completion, discard SSE frames
        async for _frame in tracker.execute_and_stream(
            agent, state, run.id, thread_id, request.checkpoint_id, request.breakpoints
        ):
            pass

        # Return the final run info
        updated_run = tracker.get_run(run.id)
        if not updated_run:
            raise HTTPException(status_code=500, detail="Run vanished unexpectedly")
        return updated_run

    @router.post(
        "/agents/{name}/runs/stream",
        summary="Create and stream a run via SSE",
    )
    async def stream_run(name: str, request: StudioRunRequest) -> StreamingResponse:
        """Create a new run and stream execution events as Server-Sent Events."""
        agent = _resolve_agent(name)
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"

        run = tracker.create_run(
            agent_name=name,
            thread_id=thread_id,
            request_data=request.model_dump(),
        )

        state = _build_studio_state(request, thread_id)

        return StreamingResponse(
            tracker.execute_and_stream(
                agent, state, run.id, thread_id, request.checkpoint_id, request.breakpoints
            ),
            media_type="text/event-stream",
            headers={
                "X-Studio-Run-Id": run.id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get(
        "/agents/{name}/runs",
        response_model=list[StudioRunInfo],
        summary="List runs for an agent",
    )
    async def list_runs(name: str, limit: int = 50) -> list[StudioRunInfo]:
        """List recent runs for a specific agent."""
        _resolve_agent(name)  # verify agent exists
        return tracker.list_runs(agent_name=name, limit=limit)

    @router.get(
        "/agents/{name}/runs/{run_id}",
        response_model=StudioRunInfo,
        summary="Get a specific run",
    )
    async def get_run(name: str, run_id: str) -> StudioRunInfo:
        """Retrieve full details and events for a specific run."""
        _resolve_agent(name)
        run = tracker.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        if run.agent_name != name:
            raise HTTPException(
                status_code=404,
                detail=f"Run '{run_id}' does not belong to agent '{name}'",
            )
        return run

    # ==================================================================
    # State & Checkpoint endpoints
    # ==================================================================

    @router.get(
        "/agents/{name}/threads/{thread_id}/state",
        response_model=StudioStateSnapshot,
        summary="Get thread state",
    )
    async def get_state(name: str, thread_id: str) -> StudioStateSnapshot:
        """Retrieve the latest state for a thread.

        If the agent has a checkpointer, loads the most recent checkpoint.
        Otherwise returns an empty state snapshot.
        """
        agent = _resolve_agent(name)
        state_data: dict[str, Any] = {}
        checkpoint_id: str | None = None

        # Try to get state from graph checkpointer
        if agent.graph_fn:
            try:
                graph = agent.graph_fn()
                checkpointer = getattr(graph, "checkpointer", None)
                if checkpointer is not None:
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": "",
                        }
                    }
                    if hasattr(checkpointer, "aget_tuple"):
                        cp_tuple = await checkpointer.aget_tuple(config)
                    elif hasattr(checkpointer, "get_tuple"):
                        cp_tuple = checkpointer.get_tuple(config)
                    else:
                        cp_tuple = None

                    if cp_tuple:
                        state_data = cp_tuple.checkpoint or {}
                        checkpoint_id = (
                            cp_tuple.config.get("configurable", {}).get("checkpoint_id")
                            if cp_tuple.config
                            else None
                        )
            except Exception as exc:
                logger.warning(f"Failed to load state from checkpointer: {exc}")

        # Fall back to store if available
        if not state_data and store is not None:
            try:
                cps = await store.list_checkpoints(thread_id, "", limit=1)
                if cps:
                    latest = cps[0]
                    state_data = latest.get("checkpoint", {})
                    checkpoint_id = latest.get("checkpoint_id")
            except Exception as exc:
                logger.warning(f"Failed to load state from store: {exc}")

        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=name,
            state=state_data,
            timestamp=_now_iso(),
            checkpoint_id=checkpoint_id,
        )

    @router.post(
        "/agents/{name}/threads/{thread_id}/state",
        response_model=StudioStateSnapshot,
        summary="Update thread state",
    )
    async def update_state(
        name: str,
        thread_id: str,
        body: StudioStateUpdate,
    ) -> StudioStateSnapshot:
        """Apply a partial state update to a thread.

        Loads the current state, merges the provided updates, and if a
        checkpointer is available saves a new checkpoint.
        """
        agent = _resolve_agent(name)

        # Get current state first
        current_state = await get_state(name, thread_id)
        merged = {**current_state.state, **body.updates}

        # Try to persist via graph checkpointer
        if agent.graph_fn:
            try:
                graph = agent.graph_fn()
                if hasattr(graph, "update_state"):
                    config = {"configurable": {"thread_id": thread_id}}
                    await graph.aupdate_state(config, body.updates)
                    logger.debug(f"State updated via graph checkpointer for {thread_id}")
            except Exception as exc:
                logger.warning(f"Failed to update state via checkpointer: {exc}")

        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=name,
            state=merged,
            timestamp=_now_iso(),
            checkpoint_id=current_state.checkpoint_id,
        )

    @router.get(
        "/agents/{name}/threads/{thread_id}/history",
        response_model=list[StudioCheckpoint],
        summary="Get checkpoint history",
    )
    async def get_history(name: str, thread_id: str) -> list[StudioCheckpoint]:
        """List checkpoint history for a thread.

        Uses the storage backend's ``list_checkpoints`` method to retrieve
        the execution history.
        """
        _resolve_agent(name)
        checkpoints: list[StudioCheckpoint] = []

        if store is not None:
            try:
                raw_cps = await store.list_checkpoints(thread_id, "")
                for idx, cp in enumerate(raw_cps):
                    checkpoints.append(
                        StudioCheckpoint(
                            id=cp.get("checkpoint_id", f"cp_{idx}"),
                            thread_id=thread_id,
                            step=idx,
                            state=cp.get("checkpoint", {}),
                            metadata=cp.get("metadata", {}),
                            parent_id=cp.get("parent_checkpoint_id"),
                            timestamp=cp.get("timestamp", _now_iso()),
                        )
                    )
            except Exception as exc:
                logger.warning(f"Failed to list checkpoints: {exc}")

        return checkpoints

    # ------------------------------------------------------------------
    # Helpers (module-level closures)
    # ------------------------------------------------------------------

    def _build_studio_state(
        request: StudioRunRequest,
        thread_id: str,
    ) -> dict[str, Any]:
        """Build the initial state dict for a studio run."""
        return {
            "current_query": request.query,
            "user_id": request.user_id,
            "thread_id": thread_id,
            "messages": [],
            "context": request.context,
            "metadata": request.metadata,
            "steps_taken": [],
            "response": "",
            "suggestions": [],
            "citations": [],
            "prompt_version": request.prompt_version,
        }

    logger.info("🎨 Studio debug API router created")
    return router
