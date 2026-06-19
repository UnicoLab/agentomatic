"""Auto-generate REST + A2A endpoints for every registered agent."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
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
    context: Any = Field(
        default_factory=dict,
        description="Context data returned by the agent (retrieved documents, search results, etc.)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(0.0, description="Processing time in ms")


class AgentChatRequest(BaseModel):
    """Session-aware chat request.

    All fields are optional except ``content``. Frontends can:
    - Supply their own ``messages`` to override automatic history loading
    - Pass ``context`` dict that the agent code can consume
    - Disable auto-persistence with ``persist=False``
    - Control history loading with ``include_history`` and ``max_history``
    """

    content: str = Field(..., description="User message")
    user_id: str = Field("default-user", description="User identifier")
    thread_id: str | None = Field(None, description="Existing thread ID")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context dict passed into state for agent code to consume",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Override: supply your own message history. "
            "Each dict should have 'role' and 'content'. "
            "When set, automatic history loading is skipped."
        ),
    )
    include_history: bool = Field(
        True, description="Load conversation history from store (ignored if messages is set)"
    )
    max_history: int | None = Field(
        None, description="Max messages to load (overrides agent default)"
    )
    persist: bool = Field(
        True, description="Auto-save user/assistant messages to the store after invocation"
    )
    prompt_version: str = Field("v1", description="Prompt version to use")


class CreateThreadRequest(BaseModel):
    """Request to explicitly create a thread."""

    thread_id: str | None = Field(None, description="Custom thread ID (auto-generated if omitted)")
    user_id: str = Field("default-user", description="User identifier")
    title: str | None = Field(None, description="Thread title")
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateThreadRequest(BaseModel):
    """Request to update thread fields."""

    title: str | None = None
    metadata: dict[str, Any] | None = None


class A2ATaskRequest(BaseModel):
    """A2A protocol task submission."""

    message: dict[str, Any] = Field(..., description="A2A message")
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizeInvokeRequest(BaseModel):
    """Optimization-specific invocation — returns full pipeline context."""

    query: str = Field(..., description="User query")
    system_prompt_override: str | None = Field(None, description="System prompt to inject")
    user_id: str = Field("optimizer", description="User ID")
    context: dict[str, Any] = Field(default_factory=dict)
    include_retrieval_context: bool = Field(True, description="Return RAG context")
    include_steps: bool = Field(True, description="Return execution steps")


class OptimizeInvokeResponse(BaseModel):
    """Full pipeline response for optimization."""

    response: str = Field("", description="Final agent response")
    retrieval_context: list[str] = Field(default_factory=list, description="RAG documents used")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tool invocations")
    steps_taken: list[str] = Field(default_factory=list, description="Execution steps")
    reasoning: str = Field("", description="Chain-of-thought")
    citations: list[dict[str, Any]] = Field(default_factory=list, description="Source citations")
    duration_ms: float = Field(0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    """User feedback on an agent response."""

    user_id: str = Field("anonymous")
    rating: int | None = Field(None, ge=1, le=5, description="1-5 star rating")
    comment: str | None = Field(None)
    correction: str | None = Field(None, description="Correct answer")
    feedback_type: str = Field("thumbs", description="thumbs|rating|correction|comment")
    query: str = Field("", description="Original query")
    response: str = Field("", description="Agent response being rated")
    thread_id: str | None = Field(None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSuspendedException(Exception):
    """Raised when an agent execution needs to be suspended for Human-in-the-Loop approval."""

    def __init__(
        self,
        approval_id: str,
        node_name: str,
        state_snapshot: dict[str, Any],
        message: str = "Execution suspended for approval",
    ) -> None:
        super().__init__(message)
        self.approval_id = approval_id
        self.node_name = node_name
        self.state_snapshot = state_snapshot
        self.message = message


class ApproveSuspendedRequest(BaseModel):
    """Request schema for approving suspended state."""

    approval_id: str
    approved: bool = True
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context to merge into state"
    )


class RejectSuspendedRequest(BaseModel):
    """Request schema for rejecting suspended state."""

    approval_id: str
    reason: str | None = None


class ForkThreadRequest(BaseModel):
    """Request schema for forking a thread."""

    message_index: int
    new_thread_id: str | None = None
    title: str | None = None


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------


def create_default_router(
    agent_name: str,
    registry: Any,
    thread_store: Any | None = None,
    state_factory: Callable[[Any], dict[str, Any]] | None = None,
    response_extractor: Callable[[dict[str, Any], str, float], Any] | None = None,
    api_prefix: str = "/api/v1",
    max_history_messages: int = 50,
    summarize_after: int = 30,
) -> APIRouter:
    """Create a full set of auto-generated endpoints for an agent.

    Generates:
      POST /invoke              — sync invocation
      POST /invoke/stream       — SSE streaming
      POST /chat                — session-aware chat
      GET  /health              — per-agent health
      GET  /config              — agent configuration
      GET  /prompts             — available prompt versions
      GET  /card                — A2A agent card
      POST /a2a/tasks           — A2A task submission
      GET  /a2a/tasks/{id}      — A2A task status
      GET  /threads             — list threads
      GET  /threads/{id}        — get thread
      GET  /threads/{id}/messages — get messages
      POST /optimize/invoke     — optimization invocation (full context)
      POST /feedback            — submit user feedback
      GET  /feedback            — list agent feedback
      GET  /feedback/export     — export feedback as JSONL

    Args:
      agent_name: Unique name of the agent.
      registry: The :class:`AgentRegistry` instance.
      thread_store: Optional thread-storage backend.
      state_factory: Custom state builder callable.
      response_extractor: Custom response extractor callable.

    Returns:
      A fully-wired :class:`~fastapi.APIRouter`.
    """
    # Check for custom schemas in the agent's package
    input_model: type[BaseModel] = AgentInvokeRequest
    output_model: type[BaseModel] = AgentInvokeResponse

    agent = registry.get(agent_name)
    if agent and agent.module_path:
        try:
            import importlib

            schemas_mod = importlib.import_module(f"{agent.module_path}.schemas")

            # Look for CustomInvokeRequest or {AgentName}Request
            title_camel = agent_name.title().replace("_", "")
            for name_candidate in [
                "CustomInvokeRequest",
                f"{title_camel}Request",
                "AgentInvokeRequest",
            ]:
                if hasattr(schemas_mod, name_candidate):
                    input_model = getattr(schemas_mod, name_candidate)
                    break

            # Look for CustomInvokeResponse or {AgentName}Response
            for name_candidate in [
                "CustomInvokeResponse",
                f"{title_camel}Response",
                "AgentInvokeResponse",
            ]:
                if hasattr(schemas_mod, name_candidate):
                    output_model = getattr(schemas_mod, name_candidate)
                    break
        except ImportError:
            pass

    router = APIRouter()

    # ── Memory Manager ────────────────────────────────────────────
    memory_mgr = None
    if thread_store:
        from agentomatic.core.memory_manager import ConversationMemoryManager

        memory_mgr = ConversationMemoryManager(
            store=thread_store,
            max_messages=max_history_messages,
            summarize_after=summarize_after,
        )

    def _get_agent() -> Any:
        """Resolve the agent from the registry or raise 404."""
        agent = registry.get(agent_name)
        if not agent:
            raise HTTPException(404, f"Agent '{agent_name}' not found")
        return agent

    def _build_initial_state(request: Any) -> dict[str, Any]:
        """Build the initial state dict for graph invocation."""
        agent = _get_agent()
        if state_factory:
            state = state_factory(request)
            if "prompt_version" not in state:
                chosen_version = None
                if hasattr(request, "prompt_version") and request.prompt_version != "v1":
                    chosen_version = request.prompt_version
                else:
                    ab_tests = (
                        getattr(agent.config, "prompt_ab_tests", None)
                        if (agent and agent.config)
                        else None
                    )
                    if ab_tests and isinstance(ab_tests, dict):
                        import random

                        versions = list(ab_tests.keys())
                        weights = [float(w) for w in ab_tests.values()]
                        chosen_version = random.choices(versions, weights=weights, k=1)[0]
                    else:
                        chosen_version = getattr(request, "prompt_version", "v1")
                state["prompt_version"] = chosen_version
            return state

        # Extract fields from model or dictionary
        request_dict: dict[str, Any] = {}
        if hasattr(request, "model_dump"):
            request_dict = request.model_dump()
        elif hasattr(request, "__dict__"):
            request_dict = request.__dict__
        elif isinstance(request, dict):
            request_dict = request

        # Resolve query/current_query or any field ending in _query
        query = request_dict.get("query") or request_dict.get("current_query") or ""
        if not query:
            for k, v in request_dict.items():
                if k.endswith("_query") and isinstance(v, str):
                    query = v
                    break

        user_id = request_dict.get("user_id") or "default-user"
        thread_id = request_dict.get("thread_id") or f"thread_{uuid.uuid4().hex[:12]}"

        # Resolve context and metadata
        context = request_dict.get("context") or {}
        metadata = request_dict.get("metadata") or {}

        # Capture other custom request parameters and merge them into metadata
        standard_keys = {"query", "current_query", "user_id", "thread_id", "context", "metadata"}
        extra_metadata = {k: v for k, v in request_dict.items() if k not in standard_keys}

        # Resolve prompt version
        chosen_version = None
        if hasattr(request, "prompt_version") and request.prompt_version != "v1":
            chosen_version = request.prompt_version
        else:
            ab_tests = (
                getattr(agent.config, "prompt_ab_tests", None)
                if (agent and agent.config)
                else None
            )
            if ab_tests and isinstance(ab_tests, dict):
                import random

                versions = list(ab_tests.keys())
                weights = [float(w) for w in ab_tests.values()]
                chosen_version = random.choices(versions, weights=weights, k=1)[0]
            else:
                chosen_version = getattr(request, "prompt_version", "v1")

        return {
            "current_query": query,
            "user_id": user_id,
            "thread_id": thread_id,
            "messages": [],  # populated later by memory manager if available
            "context": context,  # preserved as a separate state key for agent code
            "metadata": {**metadata, **extra_metadata},
            "steps_taken": [],
            "response": "",
            "suggestions": [],
            "citations": [],
            "prompt_version": chosen_version,
        }

    def _extract_response(
        result: dict[str, Any],
        agent_slug: str,
        duration_ms: float,
        prompt_version: str | None = None,
    ) -> Any:
        """Extract a standardised response from raw graph output."""
        res_metadata = result.get("metadata") or {}
        if prompt_version:
            res_metadata["prompt_version"] = prompt_version

        if response_extractor:
            return response_extractor(result, agent_slug, duration_ms)
        if output_model != AgentInvokeResponse:
            # If it's a custom schema and has a metadata field, set prompt_version there
            if hasattr(output_model, "metadata") or "metadata" in output_model.model_fields:
                result["metadata"] = res_metadata
            return output_model(**result)
        return AgentInvokeResponse(
            response=result.get("response", str(result)),
            agent_type=result.get("agent_type", agent_slug),
            thread_id=result.get("thread_id"),
            suggestions=result.get("suggestions", []),
            citations=result.get("citations", []),
            steps_taken=result.get("steps_taken", []),
            context=result.get("context", result.get("retrieved_documents", {})),
            metadata=res_metadata,
            duration_ms=duration_ms,
        )

    async def invoke(request: Any) -> Any:
        """Invoke agent synchronously."""
        agent = _get_agent()
        state = _build_initial_state(request)
        thread_id = state.get("thread_id", "")
        query = state.get("current_query", "")

        # ── Load conversation history ────────────────────────────
        if memory_mgr and thread_id:
            try:
                thread_id = await memory_mgr.get_or_create_thread(
                    thread_id, state.get("user_id", "default-user"), agent_name
                )
                state["thread_id"] = thread_id
                messages = await memory_mgr.load_history(thread_id, query)
                state["messages"] = messages
            except Exception as exc:
                logger.warning(f"History loading failed: {exc}")

        t0 = time.perf_counter()

        # Run before_node hooks
        for hook in registry.before_node_hooks:
            try:
                hook(agent_name, state)
            except Exception as hook_exc:
                logger.warning(f"Error executing before_node hook: {hook_exc}")

        try:
            if agent.graph_fn:
                graph = agent.graph_fn()
                result = await graph.ainvoke(state)
            elif agent.node_fn:
                result = await agent.node_fn(state)
            else:
                raise HTTPException(500, f"Agent '{agent_name}' has no callable")

            duration_ms = (time.perf_counter() - t0) * 1000

            # Run after_node hooks
            for hook in registry.after_node_hooks:
                try:
                    hook(agent_name, result)
                except Exception as hook_exc:
                    logger.warning(f"Error executing after_node hook: {hook_exc}")

            # ── Persist the turn ──────────────────────────────────
            response_text = result.get("response", str(result))
            if memory_mgr and thread_id and query:
                await memory_mgr.save_turn(
                    thread_id,
                    query,
                    response_text,
                    agent_name=agent.slug,
                    assistant_metadata={"prompt_version": state.get("prompt_version")},
                )

            return _extract_response(
                result, agent.slug, duration_ms, prompt_version=state.get("prompt_version")
            )
        except AgentSuspendedException as exc:
            if thread_store:
                await thread_store.save_suspended_state(
                    approval_id=exc.approval_id,
                    thread_id=thread_id or "default_thread",
                    agent_name=agent_name,
                    node_name=exc.node_name,
                    state_json=exc.state_snapshot,
                )
            raise HTTPException(
                status_code=202,
                detail={
                    "status": "suspended",
                    "approval_id": exc.approval_id,
                    "node_name": exc.node_name,
                    "message": exc.message,
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Agent {agent_name} invocation failed: {exc}")
            raise HTTPException(500, f"Agent invocation failed: {exc}") from exc

    async def invoke_stream(request: Any) -> StreamingResponse:
        """Invoke agent with SSE streaming."""
        agent = _get_agent()
        state = _build_initial_state(request)
        thread_id = state.get("thread_id", "")
        query = state.get("current_query", "")

        # ── Load conversation history ────────────────────────────
        if memory_mgr and thread_id:
            try:
                thread_id = await memory_mgr.get_or_create_thread(
                    thread_id, state.get("user_id", "default-user"), agent_name
                )
                state["thread_id"] = thread_id
                messages = await memory_mgr.load_history(thread_id, query)
                state["messages"] = messages
            except Exception as exc:
                logger.warning(f"History loading failed for stream: {exc}")

        # Run before_node hooks
        for hook in registry.before_node_hooks:
            try:
                hook(agent_name, state)
            except Exception as hook_exc:
                logger.warning(f"Error executing before_node hook: {hook_exc}")

        async def event_stream():
            """Yield SSE frames from graph or node execution."""
            collected_response = ""
            try:
                if agent.graph_fn:
                    graph = agent.graph_fn()
                    async for event in graph.astream(state):
                        yield f"data: {json.dumps(event, default=str)}\n\n"
                        # Collect response for persistence
                        if isinstance(event, dict) and "response" in event:
                            collected_response = event["response"]
                elif agent.node_fn:
                    result = await agent.node_fn(state)
                    yield f"data: {json.dumps(result, default=str)}\n\n"
                    if isinstance(result, dict):
                        collected_response = result.get("response", "")
                yield "data: [DONE]\n\n"

                # Persist the turn after streaming completes
                if memory_mgr and thread_id and query and collected_response:
                    await memory_mgr.save_turn(
                        thread_id,
                        query,
                        collected_response,
                        agent_name=agent.slug,
                    )
            except AgentSuspendedException as exc:
                if thread_store:
                    await thread_store.save_suspended_state(
                        approval_id=exc.approval_id,
                        thread_id=thread_id or "default_thread",
                        agent_name=agent_name,
                        node_name=exc.node_name,
                        state_json=exc.state_snapshot,
                    )
                yield f"data: {json.dumps({'status': 'suspended', 'approval_id': exc.approval_id, 'node_name': exc.node_name, 'message': exc.message})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"X-Agent": agent_name, "Cache-Control": "no-cache"},
        )

    # Apply annotations dynamically
    invoke.__annotations__["request"] = input_model
    invoke_stream.__annotations__["request"] = input_model

    router.add_api_route(
        "/invoke",
        invoke,
        methods=["POST"],
        response_model=output_model,
        summary="Invoke agent synchronously",
    )
    router.add_api_route(
        "/invoke/stream",
        invoke_stream,
        methods=["POST"],
        summary="Invoke agent with SSE streaming",
    )

    # ── POST /chat ────────────────────────────────────────────────
    @router.post("/chat")
    async def chat(request: AgentChatRequest) -> dict[str, Any]:
        """Session-aware chat with auto-thread management and conversation memory.

        When a ``thread_id`` is provided and a thread store is configured,
        this endpoint automatically:
        1. Loads prior conversation history into the agent's message context
        2. Invokes the agent with full conversational awareness
        3. Persists both user and assistant messages to the store

        If the conversation exceeds the configured threshold, older messages
        are automatically summarised and compressed.
        """
        agent = _get_agent()
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"

        state: dict[str, Any] = {
            "current_query": request.content,
            "user_id": request.user_id,
            "thread_id": thread_id,
            "messages": [],
            "context": request.context,
            "metadata": request.metadata,
            "steps_taken": [],
        }

        # Handle A/B Test Prompt Routing for chat
        ab_tests = (
            getattr(agent.config, "prompt_ab_tests", None) if (agent and agent.config) else None
        )
        if ab_tests and isinstance(ab_tests, dict):
            import random

            versions = list(ab_tests.keys())
            weights = [float(w) for w in ab_tests.values()]
            state["prompt_version"] = random.choices(versions, weights=weights, k=1)[0]
        else:
            state["prompt_version"] = "v1"

        # ── Load conversation history ────────────────────────────
        history_loaded = 0
        if request.messages is not None:
            # User supplied their own messages — use them directly
            try:
                from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

                lc_messages: list[Any] = []
                for msg in request.messages:
                    role = msg.get("role", "user")
                    content_val = msg.get("content", "")
                    if role == "assistant":
                        lc_messages.append(AIMessage(content=content_val))
                    elif role == "system":
                        lc_messages.append(SystemMessage(content=content_val))
                    else:
                        lc_messages.append(HumanMessage(content=content_val))
                lc_messages.append(HumanMessage(content=request.content))
                state["messages"] = lc_messages
            except ImportError:
                # langchain_core not installed — use plain dicts
                lc_messages_plain: list[dict[str, str]] = []
                for msg in request.messages:
                    lc_messages_plain.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )
                lc_messages_plain.append({"role": "user", "content": request.content})
                state["messages"] = lc_messages_plain
            history_loaded = len(request.messages)
        elif memory_mgr and request.include_history:
            try:
                thread_id = await memory_mgr.get_or_create_thread(
                    thread_id,
                    request.user_id,
                    agent_name,
                    title=request.content[:60],
                )
                state["thread_id"] = thread_id
                max_hist = request.max_history or max_history_messages
                messages = await memory_mgr.load_history(
                    thread_id,
                    request.content,
                    max_messages=max_hist,
                )
                state["messages"] = messages
                history_loaded = max(0, len(messages) - 1)
            except Exception as exc:
                logger.warning(f"History loading failed for chat: {exc}")
                state["metadata"]["_history_error"] = str(exc)

        # Run before_node hooks
        for hook in registry.before_node_hooks:
            try:
                hook(agent_name, state)
            except Exception as hook_exc:
                logger.warning(f"Error executing before_node hook: {hook_exc}")

        t0 = time.perf_counter()
        try:
            if agent.graph_fn:
                result = await agent.graph_fn().ainvoke(state)
            elif agent.node_fn:
                result = await agent.node_fn(state)
            else:
                raise HTTPException(500, f"Agent '{agent_name}' has no callable")

            duration_ms = (time.perf_counter() - t0) * 1000

            # Run after_node hooks
            for hook in registry.after_node_hooks:
                try:
                    hook(agent_name, result)
                except Exception as hook_exc:
                    logger.warning(f"Error executing after_node hook: {hook_exc}")

            res_meta = result.get("metadata") or {}
            res_meta["prompt_version"] = state["prompt_version"]

            # ── Persist the turn ──────────────────────────────────
            response_text = result.get("response", str(result))
            if memory_mgr and thread_id and request.persist:
                await memory_mgr.save_turn(
                    thread_id,
                    request.content,
                    response_text,
                    agent_name=agent.slug,
                    assistant_metadata={
                        "prompt_version": state.get("prompt_version"),
                        "agent_type": result.get("agent_type", agent.slug),
                    },
                )

            return {
                "response": response_text,
                "thread_id": thread_id,
                "agent_type": result.get("agent_type", agent.slug),
                "suggestions": result.get("suggestions", []),
                "citations": result.get("citations", []),
                "steps_taken": result.get("steps_taken", []),
                "context": result.get("context", result.get("retrieved_documents", [])),
                "duration_ms": round(duration_ms, 2),
                "metadata": res_meta,
                "history_loaded": history_loaded,
            }
        except AgentSuspendedException as exc:
            if thread_store:
                await thread_store.save_suspended_state(
                    approval_id=exc.approval_id,
                    thread_id=thread_id,
                    agent_name=agent_name,
                    node_name=exc.node_name,
                    state_json=exc.state_snapshot,
                )
            raise HTTPException(
                status_code=202,
                detail={
                    "status": "suspended",
                    "approval_id": exc.approval_id,
                    "node_name": exc.node_name,
                    "message": exc.message,
                },
            )
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
        return dict(await agent.health_check())

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
                "invoke": f"{api_prefix}/{agent_name}/invoke",
                "chat": f"{api_prefix}/{agent_name}/chat",
                "stream": f"{api_prefix}/{agent_name}/invoke/stream",
                "health": f"{api_prefix}/{agent_name}/health",
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

    # ── POST /threads ─────────────────────────────────────────────
    @router.post("/threads")
    async def create_thread_endpoint(request: CreateThreadRequest) -> dict[str, Any]:
        """Create a new conversation thread explicitly."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        tid = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"
        thread = await thread_store.create_thread(
            thread_id=tid,
            user_id=request.user_id,
            agent_name=agent_name,
            title=request.title,
            metadata=request.metadata,
        )
        return thread  # type: ignore[no-any-return]

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
                return dict(thread)
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        return {"thread_id": thread_id, "message": "Thread storage not configured"}

    # ── PATCH /threads/{thread_id} ────────────────────────────────
    @router.patch("/threads/{thread_id}")
    async def update_thread_endpoint(
        thread_id: str, request: UpdateThreadRequest
    ) -> dict[str, Any]:
        """Update thread title or metadata."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        updates: dict[str, Any] = {}
        if request.title is not None:
            updates["title"] = request.title
        if request.metadata is not None:
            updates["metadata"] = request.metadata
        if not updates:
            raise HTTPException(400, "No fields to update")
        result = await thread_store.update_thread(thread_id, **updates)
        if result is None:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        return dict(result)

    # ── DELETE /threads/{thread_id} ───────────────────────────────
    @router.delete("/threads/{thread_id}")
    async def delete_thread_endpoint(thread_id: str) -> dict[str, Any]:
        """Delete a thread and all its messages."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        deleted = await thread_store.delete_thread(thread_id)
        if not deleted:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        return {"status": "deleted", "thread_id": thread_id}

    # ── GET /threads/{thread_id}/messages ─────────────────────────
    @router.get("/threads/{thread_id}/messages")
    async def get_messages(
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get messages in a thread with pagination."""
        if thread_store:
            messages = await thread_store.get_messages(
                thread_id,
                limit=limit,
                offset=offset,
            )
            return {
                "thread_id": thread_id,
                "messages": messages,
                "count": len(messages),
                "limit": limit,
                "offset": offset,
            }
        return {
            "thread_id": thread_id,
            "messages": [],
            "message": "Thread storage not configured",
        }

    # ── DELETE /threads/{thread_id}/messages ──────────────────────
    @router.delete("/threads/{thread_id}/messages")
    async def clear_thread_messages(thread_id: str) -> dict[str, Any]:
        """Clear all messages in a thread (keeps the thread itself)."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        thread = await thread_store.get_thread(thread_id)
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        # Delete and recreate thread to clear messages
        await thread_store.delete_thread(thread_id)
        new_thread = await thread_store.create_thread(
            thread_id=thread_id,
            user_id=thread.get("user_id", "unknown"),
            agent_name=thread.get("agent_name", agent_name),
            title=thread.get("title"),
        )
        return {"status": "cleared", "thread_id": thread_id, "thread": new_thread}

    # ── GET /threads/{thread_id}/summary ──────────────────────────
    @router.get("/threads/{thread_id}/summary")
    async def get_thread_summary(thread_id: str) -> dict[str, Any]:
        """Get or generate a conversation summary for a thread."""
        if not memory_mgr:
            raise HTTPException(400, "Memory manager not configured (requires thread storage)")
        try:
            summary = await memory_mgr.get_conversation_summary(thread_id)
            return {"thread_id": thread_id, "summary": summary}
        except Exception as exc:
            raise HTTPException(500, f"Failed to generate summary: {exc}") from exc

    # ── POST /optimize/invoke ─────────────────────────────────────
    @router.post("/optimize/invoke", response_model=OptimizeInvokeResponse)
    async def optimize_invoke(request: OptimizeInvokeRequest) -> OptimizeInvokeResponse:
        """Invoke agent for optimization — returns full pipeline context.

        Unlike /invoke, this endpoint returns retrieval context,
        tool calls, reasoning steps, and citations — everything
        needed for DeepEval metrics (faithfulness, contextual_relevancy, etc.).
        """
        agent = _get_agent()
        state = {
            "current_query": request.query,
            "user_id": request.user_id,
            "thread_id": f"opt_{uuid.uuid4().hex[:12]}",
            "messages": [],
            "metadata": {
                **request.context,
                "_optimize": True,
                "_include_retrieval_context": request.include_retrieval_context,
                "_include_steps": request.include_steps,
            },
            "steps_taken": [],
            "response": "",
            "retrieval_context": [],
            "tool_calls": [],
            "reasoning": "",
        }

        # Inject prompt override
        if request.system_prompt_override:
            state["metadata"]["system_prompt_override"] = request.system_prompt_override  # type: ignore[index]

        t0 = time.perf_counter()
        try:
            if agent.graph_fn:
                result = await agent.graph_fn().ainvoke(state)
            elif agent.node_fn:
                result = await agent.node_fn(state)
            else:
                raise HTTPException(500, f"Agent '{agent_name}' has no callable")

            duration_ms = (time.perf_counter() - t0) * 1000

            return OptimizeInvokeResponse(
                response=result.get("response", str(result)),
                retrieval_context=result.get(
                    "retrieval_context", result.get("context_documents", [])
                ),
                tool_calls=result.get("tool_calls", result.get("tools_used", [])),
                steps_taken=result.get("steps_taken", []),
                reasoning=result.get("reasoning", result.get("chain_of_thought", "")),
                citations=result.get("citations", []),
                duration_ms=duration_ms,
                metadata={
                    k: v
                    for k, v in result.items()
                    if k
                    not in (
                        "response",
                        "retrieval_context",
                        "tool_calls",
                        "steps_taken",
                        "reasoning",
                        "citations",
                    )
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Optimize invoke for {agent_name} failed: {exc}")
            raise HTTPException(500, f"Optimize invoke failed: {exc}") from exc

    # ── POST /feedback ────────────────────────────────────────────
    @router.post("/feedback")
    async def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
        """Submit user feedback on an agent response."""
        from agentomatic.middleware.feedback import get_collector

        collector = get_collector()
        record = await collector.record(
            agent_name=agent_name,
            user_id=request.user_id,
            query=request.query,
            response=request.response,
            rating=request.rating,
            comment=request.comment,
            correction=request.correction,
            feedback_type=request.feedback_type,
            thread_id=request.thread_id,
            metadata=request.metadata,
        )
        return {"status": "recorded", "feedback_id": record.feedback_id}

    # ── GET /feedback ─────────────────────────────────────────────
    @router.get("/feedback")
    async def list_feedback(limit: int = 50) -> dict[str, Any]:
        """List feedback for this agent."""
        from agentomatic.middleware.feedback import get_collector

        collector = get_collector()
        records = await collector.get_feedback(agent_name=agent_name, limit=limit)
        return {"agent": agent_name, "feedback": records, "count": len(records)}

    # ── GET /feedback/export ──────────────────────────────────────
    @router.get("/feedback/export")
    async def export_feedback() -> dict[str, Any]:
        """Export feedback as JSONL for optimization datasets."""
        from agentomatic.middleware.feedback import get_collector

        collector = get_collector()
        jsonl = await collector.export_jsonl(agent_name=agent_name)
        return {
            "agent": agent_name,
            "format": "jsonl",
            "data": jsonl,
            "count": len(jsonl.strip().split("\n")) if jsonl.strip() else 0,
        }

    # ── GET /threads/{thread_id}/pending ──────────────────────────
    @router.get("/threads/{thread_id}/pending")
    async def get_pending_approvals(thread_id: str) -> dict[str, Any]:
        """Get pending human-in-the-loop approvals for a thread."""
        if thread_store:
            states = await thread_store.list_suspended_states(
                thread_id=thread_id, agent_name=agent_name
            )
            return {"thread_id": thread_id, "pending": states, "count": len(states)}
        return {"thread_id": thread_id, "pending": [], "message": "Thread storage not configured"}

    # ── POST /threads/{thread_id}/approve ─────────────────────────
    @router.post("/threads/{thread_id}/approve")
    async def approve_suspended_state(thread_id: str, request: ApproveSuspendedRequest) -> Any:
        """Approve a suspended state and resume graph execution."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        suspended = await thread_store.get_suspended_state(request.approval_id)
        if not suspended:
            raise HTTPException(
                404, f"Suspended state with approval_id '{request.approval_id}' not found"
            )

        # Delete the state now that we are resuming
        await thread_store.delete_suspended_state(request.approval_id)

        # Reconstruct the execution context
        state = suspended.get("state_snapshot") or {}
        # Apply human context updates (only into metadata to prevent state key corruption)
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"].update(request.context)
        # Mark as approved in metadata
        state["metadata"]["hitl_approved"] = True
        state["metadata"]["approval_id"] = request.approval_id

        # Resume execution
        agent = _get_agent()
        t0 = time.perf_counter()
        try:
            if agent.graph_fn:
                result = await agent.graph_fn().ainvoke(state)
            elif agent.node_fn:
                result = await agent.node_fn(state)
            else:
                raise HTTPException(500, f"Agent '{agent_name}' has no callable")

            duration_ms = (time.perf_counter() - t0) * 1000
            return _extract_response(
                result, agent.slug, duration_ms, prompt_version=state.get("prompt_version")
            )
        except AgentSuspendedException as exc:
            # Re-suspend if another HITL node is encountered
            await thread_store.save_suspended_state(
                approval_id=exc.approval_id,
                thread_id=thread_id,
                agent_name=agent_name,
                node_name=exc.node_name,
                state_json=exc.state_snapshot,
            )
            raise HTTPException(
                status_code=202,
                detail={
                    "status": "suspended",
                    "approval_id": exc.approval_id,
                    "node_name": exc.node_name,
                    "message": exc.message,
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error resuming execution: {exc}",
            ) from exc

    # ── POST /threads/{thread_id}/reject ──────────────────────────
    @router.post("/threads/{thread_id}/reject")
    async def reject_suspended_state(
        thread_id: str, request: RejectSuspendedRequest
    ) -> dict[str, Any]:
        """Reject suspended state and discard execution context."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        suspended = await thread_store.get_suspended_state(request.approval_id)
        if not suspended:
            raise HTTPException(
                404, f"Suspended state with approval_id '{request.approval_id}' not found"
            )
        await thread_store.delete_suspended_state(request.approval_id)
        return {"status": "rejected", "approval_id": request.approval_id, "reason": request.reason}

    # ── POST /threads/{thread_id}/fork ────────────────────────────
    @router.post("/threads/{thread_id}/fork")
    async def fork_thread(thread_id: str, request: ForkThreadRequest) -> dict[str, Any]:
        """Fork a conversation thread up to a specific message index."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        new_id = request.new_thread_id or f"thread_{uuid.uuid4().hex[:12]}"
        try:
            forked = await thread_store.fork_thread(
                parent_thread_id=thread_id,
                message_index=request.message_index,
                new_thread_id=new_id,
                title=request.title,
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to fork thread: {exc}") from exc
        if not forked:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        return forked  # type: ignore[no-any-return]

    # ── GET /threads/{thread_id}/lineage ──────────────────────────
    @router.get("/threads/{thread_id}/lineage")
    async def get_thread_lineage(thread_id: str) -> dict[str, Any]:
        """Get the full lineage tree (ancestors and descendants) for a thread."""
        if not thread_store:
            raise HTTPException(400, "Thread storage not configured")
        try:
            lineage = await thread_store.get_thread_lineage(thread_id)
            return lineage  # type: ignore[no-any-return]
        except Exception as exc:
            raise HTTPException(500, f"Failed to retrieve lineage: {exc}") from exc

    return router
