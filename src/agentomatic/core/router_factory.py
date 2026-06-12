"""Auto-generate REST + A2A endpoints for every registered agent."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class AgentInvokeRequest(BaseModel):
    """Standard invocation request."""

    query: str = Field(..., description="User query or input")
    user_id: str = Field("default-user", description="User identifier")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    thread_id: str | None = Field(None, description="Thread ID for conversation continuity")
    prompt_version: str = Field("v1", description="Prompt version to use")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature override")
    max_tokens: int | None = Field(None, ge=1, description="Max tokens override")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class AgentInvokeResponse(BaseModel):
    """Standard invocation response."""

    response: str = Field("", description="Agent response text")
    agent_type: str = Field("", description="Agent slug")
    thread_id: str | None = Field(None, description="Thread ID")
    suggestions: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    steps_taken: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(0.0, description="Processing time in ms")


class AgentChatRequest(BaseModel):
    """Session-aware chat request."""

    content: str = Field(..., description="User message")
    user_id: str = Field("default-user", description="User identifier")
    thread_id: str | None = Field(None, description="Existing thread ID")
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATaskRequest(BaseModel):
    """A2A protocol task submission."""

    message: dict[str, Any] = Field(..., description="A2A message")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------


def create_default_router(
    agent_name: str,
    registry: Any,
    thread_store: Any | None = None,
) -> APIRouter:
    """Create a full set of auto-generated endpoints for an agent.

    Generates:
      POST /invoke          — sync invocation
      POST /invoke/stream   — SSE streaming
      POST /chat            — session-aware chat
      GET  /health          — per-agent health
      GET  /config          — agent configuration
      GET  /prompts         — available prompt versions
      GET  /card            — A2A agent card
      POST /a2a/tasks       — A2A task submission
      GET  /a2a/tasks/{id}  — A2A task status
      GET  /threads         — list threads
      GET  /threads/{id}    — get thread
      GET  /threads/{id}/messages — get messages

    Args:
        agent_name: Unique name of the agent.
        registry: The :class:`AgentRegistry` instance.
        thread_store: Optional thread-storage backend.

    Returns:
        A fully-wired :class:`~fastapi.APIRouter`.
    """
    router = APIRouter()

    def _get_agent():
        """Retrieve the agent from the registry or raise 404."""
        agent = registry.get(agent_name)
        if not agent:
            raise HTTPException(404, f"Agent '{agent_name}' not found")
        return agent

    def _build_initial_state(request: AgentInvokeRequest) -> dict[str, Any]:
        """Build the initial state dict for graph invocation."""
        return {
            "current_query": request.query,
            "user_id": request.user_id,
            "thread_id": request.thread_id or f"thread_{uuid.uuid4().hex[:12]}",
            "messages": [],
            "metadata": {**request.context, **request.metadata},
            "steps_taken": [],
            "response": "",
            "suggestions": [],
            "citations": [],
        }

    def _extract_response(
        result: dict[str, Any],
        agent_slug: str,
        duration_ms: float,
    ) -> AgentInvokeResponse:
        """Extract a standardised response from raw graph output."""
        return AgentInvokeResponse(
            response=result.get("response", str(result)),
            agent_type=result.get("agent_type", agent_slug),
            thread_id=result.get("thread_id"),
            suggestions=result.get("suggestions", []),
            citations=result.get("citations", []),
            steps_taken=result.get("steps_taken", []),
            metadata=result.get("metadata", {}),
            duration_ms=duration_ms,
        )

    # ── POST /invoke ──────────────────────────────────────────────
    @router.post("/invoke", response_model=AgentInvokeResponse)
    async def invoke(request: AgentInvokeRequest) -> AgentInvokeResponse:
        """Invoke agent synchronously."""
        agent = _get_agent()
        state = _build_initial_state(request)
        t0 = time.perf_counter()

        try:
            if agent.graph_fn:
                graph = agent.graph_fn()
                result = await graph.ainvoke(state)
            elif agent.node_fn:
                result = await agent.node_fn(state)
            else:
                raise HTTPException(500, f"Agent '{agent_name}' has no callable")

            duration_ms = (time.perf_counter() - t0) * 1000
            return _extract_response(result, agent.slug, duration_ms)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Agent {agent_name} invocation failed: {exc}")
            raise HTTPException(500, f"Agent invocation failed: {exc}") from exc

    # ── POST /invoke/stream ───────────────────────────────────────
    @router.post("/invoke/stream")
    async def invoke_stream(request: AgentInvokeRequest) -> StreamingResponse:
        """Invoke agent with SSE streaming."""
        agent = _get_agent()
        state = _build_initial_state(request)

        async def event_stream():
            """Yield SSE frames from graph or node execution."""
            try:
                if agent.graph_fn:
                    graph = agent.graph_fn()
                    async for event in graph.astream(state):
                        yield f"data: {json.dumps(event, default=str)}\n\n"
                elif agent.node_fn:
                    result = await agent.node_fn(state)
                    yield f"data: {json.dumps(result, default=str)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"X-Agent": agent_name, "Cache-Control": "no-cache"},
        )

    # ── POST /chat ────────────────────────────────────────────────
    @router.post("/chat")
    async def chat(request: AgentChatRequest) -> dict[str, Any]:
        """Session-aware chat with auto-thread management."""
        agent = _get_agent()
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"

        state: dict[str, Any] = {
            "current_query": request.content,
            "user_id": request.user_id,
            "thread_id": thread_id,
            "messages": [],
            "metadata": request.metadata,
            "steps_taken": [],
        }

        t0 = time.perf_counter()
        try:
            if agent.graph_fn:
                result = await agent.graph_fn().ainvoke(state)
            elif agent.node_fn:
                result = await agent.node_fn(state)
            else:
                raise HTTPException(500, f"Agent '{agent_name}' has no callable")

            duration_ms = (time.perf_counter() - t0) * 1000
            return {
                "response": result.get("response", str(result)),
                "thread_id": thread_id,
                "agent_type": result.get("agent_type", agent.slug),
                "suggestions": result.get("suggestions", []),
                "citations": result.get("citations", []),
                "duration_ms": round(duration_ms, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Chat with {agent_name} failed: {exc}")
            raise HTTPException(500, f"Chat failed: {exc}") from exc

    # ── GET /health ───────────────────────────────────────────────
    @router.get("/health")
    async def health() -> dict[str, Any]:
        """Per-agent health check."""
        agent = _get_agent()
        return await agent.health_check()

    # ── GET /config ───────────────────────────────────────────────
    @router.get("/config")
    async def get_config() -> dict[str, Any]:
        """Get agent configuration."""
        agent = _get_agent()
        if agent.config:
            if hasattr(agent.config, "model_dump"):
                return {"agent": agent_name, "config": agent.config.model_dump()}
            return {"agent": agent_name, "config": vars(agent.config)}
        return {"agent": agent_name, "config": {}}

    # ── GET /prompts ──────────────────────────────────────────────
    @router.get("/prompts")
    async def get_prompts() -> dict[str, Any]:
        """List available prompt versions."""
        agent = _get_agent()
        if agent.prompt_manager:
            return {
                "agent": agent_name,
                "versions": agent.prompt_manager.list_versions(),
                "active": (
                    agent.config.prompt_version
                    if agent.config and hasattr(agent.config, "prompt_version")
                    else "v1"
                ),
            }
        return {"agent": agent_name, "versions": [], "active": "v1"}

    # ── GET /card ─────────────────────────────────────────────────
    @router.get("/card")
    async def get_card() -> dict[str, Any]:
        """A2A Agent Card."""
        agent = _get_agent()
        m = agent.manifest
        return {
            "name": m.slug,
            "description": m.description,
            "version": m.version,
            "framework": m.framework,
            "capabilities": {
                "streaming": True,
                "chat": True,
                "invoke": True,
                "a2a": True,
            },
            "endpoints": {
                "invoke": f"/api/v1/{agent_name}/invoke",
                "chat": f"/api/v1/{agent_name}/chat",
                "stream": f"/api/v1/{agent_name}/invoke/stream",
                "health": f"/api/v1/{agent_name}/health",
            },
            "metadata": m.metadata,
        }

    # ── POST /a2a/tasks ───────────────────────────────────────────
    @router.post("/a2a/tasks")
    async def submit_a2a_task(request: A2ATaskRequest) -> dict[str, Any]:
        """Submit an A2A task."""
        agent = _get_agent()
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        query = request.message.get("content", "")

        state: dict[str, Any] = {
            "current_query": query,
            "user_id": "a2a",
            "thread_id": task_id,
            "messages": [],
            "metadata": {"a2a": True, **request.metadata},
        }

        try:
            if agent.graph_fn:
                result = await agent.graph_fn().ainvoke(state)
            elif agent.node_fn:
                result = await agent.node_fn(state)
            else:
                raise HTTPException(500, "No callable")

            return {
                "task_id": task_id,
                "status": "completed",
                "result": result.get("response", str(result)),
            }
        except Exception as exc:
            return {"task_id": task_id, "status": "failed", "error": str(exc)}

    # ── GET /a2a/tasks/{task_id} ──────────────────────────────────
    @router.get("/a2a/tasks/{task_id}")
    async def get_a2a_task(task_id: str) -> dict[str, Any]:
        """Get A2A task status."""
        return {
            "task_id": task_id,
            "status": "completed",
            "message": "Task tracking requires storage backend",
        }

    # ── GET /threads ──────────────────────────────────────────────
    @router.get("/threads")
    async def list_threads(user_id: str | None = None) -> dict[str, Any]:
        """List conversation threads."""
        if thread_store:
            threads = await thread_store.list_threads(
                agent_name=agent_name,
                user_id=user_id,
            )
            return {"threads": threads, "count": len(threads)}
        return {"threads": [], "count": 0, "message": "Thread storage not configured"}

    # ── GET /threads/{thread_id} ──────────────────────────────────
    @router.get("/threads/{thread_id}")
    async def get_thread(thread_id: str) -> dict[str, Any]:
        """Get a conversation thread."""
        if thread_store:
            thread = await thread_store.get_thread(thread_id)
            if thread:
                return thread
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        return {"thread_id": thread_id, "message": "Thread storage not configured"}

    # ── GET /threads/{thread_id}/messages ─────────────────────────
    @router.get("/threads/{thread_id}/messages")
    async def get_messages(thread_id: str) -> dict[str, Any]:
        """Get messages in a thread."""
        if thread_store:
            messages = await thread_store.get_messages(thread_id)
            return {
                "thread_id": thread_id,
                "messages": messages,
                "count": len(messages),
            }
        return {
            "thread_id": thread_id,
            "messages": [],
            "message": "Thread storage not configured",
        }

    return router
