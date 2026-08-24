"""FastAPI router for the Agentomatic Studio debug API."""

from __future__ import annotations

import json as json_mod
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from agentomatic.core.errors import client_safe_detail
from agentomatic.core.schemas import SchemaValidator, load_schema_models
from agentomatic.studio.adapters import resolve_adapter
from agentomatic.studio.models import (
    StudioAgentInfo,
    StudioAgentSchemas,
    StudioCheckpoint,
    StudioGraphTopology,
    StudioRunInfo,
    StudioRunRequest,
    StudioSchemaSource,
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
    return datetime.now(UTC).isoformat()


class StudioResumeRequest(BaseModel):
    """Request to resume a paused/interrupted execution.

    Defined at module scope so FastAPI/Pydantic can resolve the type when
    generating the OpenAPI schema (a nested ForwardRef inside
    :func:`create_studio_router` previously blanked the full schema).
    """

    value: Any = Field(default=None, description="Human response or approval value")
    action: str = Field("approve", description="'approve' or 'reject'")


StudioResumeRequest.model_rebuild()


def _resolve_agent_schemas(
    agent: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, StudioSchemaSource]]:
    """Resolve JSON Schemas and their provenance for an agent.

    Priority:
      1. :class:`SchemaValidator` discovered at registration time (runtime
         validation is active for these models).
      2. Pydantic models declared in the agent's ``schemas.py`` module
         (discovered here on demand — covers any naming convention).
      3. Default platform models (``AgentInvokeRequest`` /
         ``AgentInvokeResponse``).

    Returns:
        ``(input_schema, output_schema, schema_source)``.
    """
    from agentomatic.core.router_factory import AgentInvokeRequest, AgentInvokeResponse

    def _schema_for(
        model: type[BaseModel] | None,
        default_model: type[BaseModel],
    ) -> tuple[dict[str, Any], StudioSchemaSource]:
        if model is None:
            default_schema = default_model.model_json_schema()
            return default_schema, StudioSchemaSource(
                source="default",
                model=default_model.__name__,
                convention="default",
            )
        return model.model_json_schema(), StudioSchemaSource(
            source="custom",
            model=model.__name__,
            convention=f"schemas.{model.__module__}.{model.__name__}",
        )

    # Priority 1: SchemaValidator from registry discovery.
    validator: SchemaValidator | None = getattr(agent, "schema_validator", None)
    if isinstance(validator, SchemaValidator):
        input_model = validator.request_model
        output_model = validator.response_model
        input_schema, input_source = _schema_for(input_model, AgentInvokeRequest)
        output_schema, output_source = _schema_for(output_model, AgentInvokeResponse)
        if input_model is not None:
            input_source.convention = f"schema_validator.{input_model.__name__}"
        if output_model is not None:
            output_source.convention = f"schema_validator.{output_model.__name__}"
        return (
            input_schema,
            output_schema,
            {"input": input_source, "output": output_source},
        )

    # Priority 2: scan the agent's schemas.py module on demand.
    input_model, output_model = load_schema_models(
        getattr(agent, "module_path", "") or "",
        agent.name,
    )
    input_schema, input_source = _schema_for(input_model, AgentInvokeRequest)
    output_schema, output_source = _schema_for(output_model, AgentInvokeResponse)
    return (
        input_schema,
        output_schema,
        {"input": input_source, "output": output_source},
    )


def create_studio_router(
    registry: AgentRegistry,
    store: BaseStore | None = None,
    platform_title: str = "Agentomatic Platform",
    platform_version: str = "1.0.0",
) -> APIRouter:
    """Create the Studio debug API router.

    Uses the universal :func:`~agentomatic.studio.adapters.resolve_adapter`
    factory to provide the best debugging experience for every agent,
    regardless of framework.

    Args:
        registry: The platform's agent registry.
        store: Optional storage backend for checkpoints and state.
        platform_title: Human-readable platform title.
        platform_version: Platform version string.

    Returns:
        A fully-wired :class:`~fastapi.APIRouter`.
    """
    router = APIRouter(prefix="/studio", tags=["Studio Debug API"])
    tracker = RunTracker()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_agent(name: str):
        """Look up an agent by folder name or slug, or raise 404."""
        agent = registry.get(name)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{name}' not found (tried name and slug)",
            )
        return agent

    def _get_adapter(name: str):
        """Resolve the agent and its Studio adapter."""
        agent = _resolve_agent(name)
        return agent, resolve_adapter(agent, store)

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
            adapter = resolve_adapter(agent, store)
            infos.append(
                StudioAgentInfo(
                    name=agent.name,
                    slug=agent.slug,
                    description=agent.manifest.description,
                    version=agent.manifest.version,
                    framework=agent.manifest.framework,
                    capabilities=adapter.capabilities,
                    has_graph=adapter.supports_graph or agent.graph_fn is not None,
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
        """Return the execution graph topology for an agent.

        Works for all agent frameworks — LangGraph agents get their real
        graph topology extracted, while other frameworks receive a
        synthetic or user-defined topology.
        """
        import asyncio

        _agent, adapter = _get_adapter(name)
        try:
            return await asyncio.wait_for(adapter.get_graph(), timeout=8.0)
        except TimeoutError:
            return StudioGraphTopology(
                agent_name=name,
                metadata={"error": "get_graph timed out"},
            )
        except Exception as exc:
            # Graph introspection runs user code (graph_fn touches imports and
            # connections). A debug view failing to draw must not 500 — degrade
            # to an empty topology carrying the reason.
            logger.warning(f"Studio get_graph failed for '{name}': {exc}")
            return StudioGraphTopology(
                agent_name=name,
                metadata=client_safe_detail(exc, context="get_graph failed"),
            )

    @router.get(
        "/agents/{name}/schemas",
        response_model=StudioAgentSchemas,
        summary="Get agent input/output schemas",
    )
    async def get_schemas(name: str) -> StudioAgentSchemas:
        """Return JSON schemas for agent input and output models.

        Custom schemas come from the agent's ``schemas.py`` module (any of the
        supported naming conventions) or its registered
        ``SchemaValidator``. Agents without custom schemas fall back to the
        default platform request/response models — the response's
        ``schema_source`` tells clients which model was used.
        """
        agent = _resolve_agent(name)
        input_schema, output_schema, schema_source = _resolve_agent_schemas(agent)
        return StudioAgentSchemas(
            input_schema=input_schema,
            output_schema=output_schema,
            schema_source=schema_source,
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
        _agent, adapter = _get_adapter(name)
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"

        run = tracker.create_run(
            agent_name=name,
            thread_id=thread_id,
            request_data=request.model_dump(),
        )

        state = _build_studio_state(request, thread_id)

        # Execute — consume the stream to completion, discard SSE frames
        async for _frame in tracker.execute_with_adapter(
            adapter, state, run.id, thread_id, request.checkpoint_id, request.breakpoints
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
        """Create a new run and stream execution events as Server-Sent Events.

        Works with all agent frameworks. LangGraph agents stream rich
        node-level events; other frameworks stream trace events with
        timing data and input/output payloads.
        """
        _agent, adapter = _get_adapter(name)
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"

        run = tracker.create_run(
            agent_name=name,
            thread_id=thread_id,
            request_data=request.model_dump(),
        )

        state = _build_studio_state(request, thread_id)

        return StreamingResponse(
            tracker.execute_with_adapter(
                adapter, state, run.id, thread_id, request.checkpoint_id, request.breakpoints
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

        Delegates to the agent's Studio adapter, which uses the best
        available method (checkpointer, in-memory store, or custom
        provider) to retrieve state.
        """
        _agent, adapter = _get_adapter(name)
        result = await adapter.get_state(thread_id)
        if result is None:
            return StudioStateSnapshot(
                thread_id=thread_id,
                agent_name=name,
                state={},
                timestamp=_now_iso(),
            )
        return result

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

        Delegates to the agent's Studio adapter. LangGraph agents
        persist changes via the graph checkpointer; other agents
        update the in-memory trace store.
        """
        _agent, adapter = _get_adapter(name)
        result = await adapter.update_state(thread_id, body.updates)
        if result is None:
            return StudioStateSnapshot(
                thread_id=thread_id,
                agent_name=name,
                state=body.updates,
                timestamp=_now_iso(),
            )
        return result

    # ==================================================================
    # Resume endpoint (deep_agent / HITL interrupt support)
    # ==================================================================

    @router.post(
        "/agents/{name}/threads/{thread_id}/resume",
        summary="Resume interrupted execution",
    )
    async def resume_execution(
        name: str,
        thread_id: str,
        body: StudioResumeRequest,
    ) -> StreamingResponse:
        """Resume a LangGraph execution that was paused by an interrupt.

        Passes the human's response to the graph via ``Command(resume=value)``
        and streams the continued execution.
        """
        agent, adapter = _get_adapter(name)

        if agent.graph_fn is None:
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{name}' does not support interrupt/resume (no graph_fn)",
            )

        # Resume is a LangGraph feature: it needs ``astream_events`` and
        # ``Command(resume=...)``. Agentomatic's own lightweight AgentGraph has
        # neither, so without this guard the call raised AttributeError *inside*
        # the SSE body — surfacing a raw internal error with a 200 status.
        try:
            resume_graph = agent.graph_fn()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=client_safe_detail(exc, context="Failed to build agent graph"),
            ) from exc
        if not hasattr(resume_graph, "astream_events"):
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Agent '{name}' does not support interrupt/resume: its graph is not "
                    "a LangGraph runnable (no 'astream_events'). Compile the agent with "
                    "LangGraph to use interrupts."
                ),
            )

        async def _stream() -> AsyncGenerator[str, None]:
            try:
                # Guard re-check for the closure (the route already 400s above).
                if agent.graph_fn is None:
                    yield 'data: {"event": "run_error", "data": {"error": "no graph_fn"}}\n\n'
                    return
                graph = resume_graph  # already built (and validated) above
                config = {"configurable": {"thread_id": thread_id}}

                # Use LangGraph's Command to resume from interrupt
                resume_input: Any
                try:
                    from langgraph.types import Command

                    resume_input = Command(resume=body.value)
                except ImportError:
                    resume_input = {"__resume__": body.value}

                async for lg_event in graph.astream_events(
                    resume_input, config=config, version="v2"
                ):
                    mapped = adapter._map_event(lg_event)
                    if mapped:
                        yield f"data: {mapped.model_dump_json()}\n\n"

                yield 'data: {"event": "done"}\n\n'
            except Exception as exc:
                # Don't echo raw internal exception text to the client — it can
                # carry credentials/paths. Full detail goes to the server log,
                # correlated by error_id.
                error_data = json_mod.dumps(
                    {
                        "event": "run_error",
                        "data": client_safe_detail(exc, context="Resume failed"),
                    }
                )
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get(
        "/agents/{name}/threads/{thread_id}/history",
        response_model=list[StudioCheckpoint],
        summary="Get checkpoint history",
    )
    async def get_history(name: str, thread_id: str) -> list[StudioCheckpoint]:
        """List checkpoint or execution history for a thread.

        LangGraph agents return real checkpoint history. Other agents
        return execution trace history from the in-memory store.
        """
        _agent, adapter = _get_adapter(name)
        return await adapter.get_history(thread_id)

    # ------------------------------------------------------------------
    # Helpers (module-level closures)
    # ------------------------------------------------------------------

    def _build_studio_state(
        request: StudioRunRequest,
        thread_id: str,
    ) -> dict[str, Any]:
        """Build the initial state dict for a studio run.

        Preserves the full client payload (including unknown top-level
        extras) so class agents receive everything in ``input_to_state``.
        """
        from agentomatic.core.agent_invoke import build_invoke_state

        payload = request.model_dump()
        # Studio-only controls should not leak into agent input.
        for studio_only in ("breakpoints", "checkpoint_id"):
            payload.pop(studio_only, None)
        return build_invoke_state(
            payload,
            default_thread_id=thread_id,
            prompt_version=request.prompt_version,
        )

    logger.info("🎨 Studio debug API router created")
    return router
