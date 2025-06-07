"""API router with comprehensive agent endpoints and streaming support."""

import importlib
import pkgutil
from typing import Any, Dict, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path, status, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from .settings import config
from .dependencies import agent_registry
from ..common.api_decorators import (
    handle_api_errors, log_api_call, rate_limit, validate_streaming_support,
    create_streaming_response, agent_context, agent_queue, APIResponse
)
from ..common.llm_factory import LLMProvider


class AgentRequest(BaseModel):
    """Base request model for agent interactions."""
    input: str
    context: Optional[str] = ""
    streaming: bool = False
    prompt_version: str = "v1"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Base response model for agent interactions."""
    output: str
    agent: str
    prompt_version: str
    streaming: bool = False
    metadata: Optional[Dict[str, Any]] = None


class PromptInfo(BaseModel):
    """Information about a prompt version."""
    version: str
    description: str
    author: str
    tags: list[str]
    created_at: str


def create_api_router() -> APIRouter:
    """Create the main API router with comprehensive agent endpoints."""
    router = APIRouter()

    # === SYSTEM ENDPOINTS ===

    @router.get("/api/{config.api_version}/agents",
                summary="List all agents",
                tags=["Agents Management"])
    @handle_api_errors
    @log_api_call
    async def list_agents():
        """List all registered agents with their information."""
        agents_info = agent_registry.list_agents()
        return APIResponse(
            data=agents_info,
            message=f"Found {len(agents_info)} registered agents"
        )

    @router.get("/api/{config.api_version}/agents/{agent_name}/health",
                summary="Agent health check",
                tags=["Agents Management"])
    @handle_api_errors
    @log_api_call
    async def agent_health_check(agent_name: str = Path(..., description="Agent name")):
        """Check the health of a specific agent."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )

        health_info = await agent.health_check()
        return APIResponse(data=health_info, message="Health check completed")

    @router.get("/api/{config.api_version}/agents/{agent_name}/prompts",
                summary="List agent prompts",
                tags=["Prompts Management"])
    @handle_api_errors
    @log_api_call
    async def list_agent_prompts(agent_name: str = Path(..., description="Agent name")):
        """List all available prompt versions for an agent."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )

        versions = agent.prompt_manager.list_versions()
        prompts_info = []

        for version in versions:
            prompt_info = agent.prompt_manager.get_prompt_info(version)
            if prompt_info:
                prompts_info.append(PromptInfo(
                    version=prompt_info.version,
                    description=prompt_info.description,
                    author=prompt_info.author,
                    tags=prompt_info.tags,
                    created_at=prompt_info.created_at.isoformat()
                ))

        return APIResponse(
            data=prompts_info,
            message=f"Found {len(prompts_info)} prompt versions"
        )

    @router.get("/api/{config.api_version}/agents/{agent_name}/prompts/{version}",
                summary="Get specific prompt",
                tags=["Prompts Management"])
    @handle_api_errors
    @log_api_call
    async def get_agent_prompt(
        agent_name: str = Path(..., description="Agent name"),
        version: str = Path(..., description="Prompt version")
    ):
        """Get a specific prompt version content."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )

        prompt_content = agent.get_prompt(version)
        if not prompt_content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt version '{version}' not found for agent '{agent_name}'"
            )

        prompt_info = agent.prompt_manager.get_prompt_info(version)
        return APIResponse(
            data={
                "content": prompt_content,
                "info": PromptInfo(
                    version=prompt_info.version,
                    description=prompt_info.description,
                    author=prompt_info.author,
                    tags=prompt_info.tags,
                    created_at=prompt_info.created_at.isoformat()
                ) if prompt_info else None
            },
            message=f"Retrieved prompt version '{version}'"
        )

    # === AGENT INTERACTION ENDPOINTS ===

    @router.post("/api/{config.api_version}/agents/{agent_name}/chat",
                 summary="Chat with agent",
                 tags=["Agent Interaction"],
                 response_model=Union[AgentResponse, APIResponse])
    @handle_api_errors
    @log_api_call
    @rate_limit(max_calls=config.rate_limit_calls, window_seconds=config.rate_limit_window)
    @validate_streaming_support
    async def chat_with_agent(
        agent_name: str,
        request: AgentRequest,
        background_tasks: BackgroundTasks
    ):
        """Chat with a specific agent with support for streaming."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )

        async with agent_context(agent_name):
            # Format the prompt
            formatted_prompt = agent.format_prompt(
                version=request.prompt_version,
                input=request.input,
                context=request.context or "",
                history=""  # Could be expanded to include conversation history
            )

            if not formatted_prompt:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Prompt version '{request.prompt_version}' not available"
                )

            # Override LLM config with request parameters
            generation_kwargs = {}
            if request.temperature is not None:
                generation_kwargs['temperature'] = request.temperature
            if request.max_tokens is not None:
                generation_kwargs['max_tokens'] = request.max_tokens

            # Generate response
            if request.streaming:
                response_generator = await agent.generate_response(
                    formatted_prompt,
                    streaming=True,
                    **generation_kwargs
                )
                return await create_streaming_response(
                    response_generator,
                    agent_name,
                    media_type="text/event-stream"
                )
            else:
                response_content = await agent.generate_response(
                    formatted_prompt,
                    streaming=False,
                    **generation_kwargs
                )

                return APIResponse(
                    data=AgentResponse(
                        output=response_content,
                        agent=agent_name,
                        prompt_version=request.prompt_version,
                        streaming=False,
                        metadata=request.metadata
                    ),
                    message="Response generated successfully"
                )

    @router.post("/api/{config.api_version}/agents/{agent_name}/run",
                 summary="Run agent with full workflow",
                 tags=["Agent Interaction"])
    @handle_api_errors
    @log_api_call
    @rate_limit(max_calls=config.rate_limit_calls, window_seconds=config.rate_limit_window)
    async def run_agent(
        agent_name: str,
        request: AgentRequest,
        background_tasks: BackgroundTasks
    ):
        """Run agent's full workflow (LangGraph execution)."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )

        async with agent_context(agent_name):
            # If queue is enabled and busy, queue the request
            if config.enable_queue:
                try:
                    request_id = await agent_queue.add_request({
                        "agent_name": agent_name,
                        "request": request.model_dump(),
                        "type": "run"
                    })

                    # Process in background
                    background_tasks.add_task(
                        _process_queued_request,
                        agent,
                        request,
                        request_id
                    )

                    return APIResponse(
                        data={"request_id": request_id},
                        message="Request queued for processing"
                    )
                except HTTPException:
                    # Queue is full, process directly
                    pass

            # Execute agent workflow directly
            result = await agent.run(request, streaming=request.streaming)

            if request.streaming:
                return await create_streaming_response(
                    result,
                    agent_name,
                    media_type="text/event-stream"
                )
            else:
                return APIResponse(
                    data=result,
                    message="Agent workflow completed successfully"
                )

    # === UTILITY FUNCTIONS ===

    async def _process_queued_request(agent, request: AgentRequest, request_id: str):
        """Process a queued agent request."""
        try:
            logger.info(f"Processing queued request {request_id}")
            result = await agent.run(request, streaming=False)
            # In a real implementation, you'd store results in a database or cache
            logger.info(f"Completed queued request {request_id}")
        except Exception as e:
            logger.error(f"Failed to process queued request {request_id}: {e}")

    # Auto-discover and include legacy agent routers for backward compatibility
    _include_legacy_agent_routers(router)

    return router


def _include_legacy_agent_routers(main_router: APIRouter) -> None:
    """Discover and include legacy agent API routers for backward compatibility."""
    try:
        # Import the agents package
        agents_package = importlib.import_module(config.agents_package)

        # Walk through all agent modules
        for _, module_name, _ in pkgutil.iter_modules(
            agents_package.__path__,
            agents_package.__name__ + "."
        ):
            try:
                # Import each agent's API module
                api_module = importlib.import_module(f"{module_name}.api")

                # Get the router from the module
                if hasattr(api_module, "router"):
                    agent_name = module_name.split(".")[-1].replace("agent_", "")
                    route_prefix = f"/api/{config.api_version}/legacy/{agent_name}"

                    main_router.include_router(
                        api_module.router,
                        prefix=route_prefix,
                        tags=[f"legacy_agent_{agent_name}"]
                    )
                    logger.info(f"Included legacy API router for agent: {agent_name}")

            except Exception as e:
                logger.debug(f"No legacy API router for {module_name}: {e}")

    except Exception as e:
        logger.error(f"Failed to discover legacy agent API routers: {e}")