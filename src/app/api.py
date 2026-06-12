"""API router with comprehensive agent endpoints and streaming support."""

import importlib
import pkgutil
from typing import Any, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, status
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from ..common.api_decorators import (
    APIResponse,
    agent_context,
    create_streaming_response,
    handle_api_errors,
    log_api_call,
    rate_limit,
    validate_streaming_support,
)
from .dependencies import agent_registry
from .settings import config


class UniversalAgentInput(BaseModel):
    """Universal input model that can handle any agent's input format."""

    payload: dict[str, Any] = Field(..., description="Agent-specific input payload")
    streaming: bool = Field(default=False, description="Enable streaming response")
    prompt_version: str = Field(default="v1", description="Prompt version to use")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="LLM temperature override"
    )
    max_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens override")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class UniversalAgentResponse(BaseModel):
    """Universal response model for any agent."""

    output: Any = Field(..., description="Agent-specific output")
    agent: str = Field(..., description="Agent name that processed the request")
    prompt_version: str = Field(..., description="Prompt version used")
    streaming: bool = Field(default=False, description="Whether response was streamed")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")
    validation_info: dict[str, Any] | None = Field(
        default=None, description="Input validation details"
    )


class AgentRequest(BaseModel):
    """Base request model for agent interactions."""

    input: str
    context: str | None = ""
    streaming: bool = False
    prompt_version: str = "v1"
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] | None = None


class AgentResponse(BaseModel):
    """Base response model for agent interactions."""

    output: str
    agent: str
    prompt_version: str
    streaming: bool = False
    metadata: dict[str, Any] | None = None


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

    def get_agent_tag(agent_name: str) -> str:
        """Get the proper tag name for an agent."""
        return agent_name.title()

    def validate_agent_input(agent_name: str, payload: dict[str, Any]) -> Any:
        """Dynamically validate input payload against agent's schema."""
        try:
            # Try to import the agent's schema module
            schema_module = importlib.import_module(f"src.agents.{agent_name}.schemas")

            # Look for the input schema class (convention: {AgentName}Input)
            input_class_name = f"{agent_name.title()}Input"
            input_class = getattr(schema_module, input_class_name, None)

            if input_class:
                # Validate the payload against the agent's schema
                validated_input = input_class(**payload)
                return validated_input
            else:
                # Fallback: return payload as-is if no schema found
                logger.warning(f"No input schema found for agent {agent_name}, using raw payload")
                return payload

        except ImportError:
            logger.warning(f"No schema module found for agent {agent_name}, using raw payload")
            return payload
        except ValidationError as e:
            # Re-raise validation errors with detailed information
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Input validation failed",
                    "agent": agent_name,
                    "validation_errors": e.errors(),
                    "expected_schema": f"{agent_name.title()}Input",
                },
            )
        except Exception as e:
            logger.error(f"Unexpected error validating input for {agent_name}: {e}")
            return payload

    def get_agent_schema_info(agent_name: str) -> dict[str, Any]:
        """Get schema information for an agent."""
        try:
            schema_module = importlib.import_module(f"src.agents.{agent_name}.schemas")
            input_class_name = f"{agent_name.title()}Input"
            input_class = getattr(schema_module, input_class_name, None)

            if input_class:
                # Extract schema information
                schema_info = {
                    "schema_class": input_class_name,
                    "fields": {},
                    "model_config": getattr(input_class, "model_config", {}),
                }

                # Get field information
                if hasattr(input_class, "model_fields"):
                    for field_name, field_info in input_class.model_fields.items():
                        schema_info["fields"][field_name] = {
                            "type": str(field_info.annotation),
                            "required": field_info.is_required(),
                            "description": getattr(field_info, "description", ""),
                        }

                return schema_info
            else:
                return {"error": f"No input schema found for agent {agent_name}"}

        except ImportError:
            return {"error": f"No schema module found for agent {agent_name}"}
        except Exception as e:
            return {"error": f"Error loading schema for {agent_name}: {str(e)}"}

    # === SYSTEM ENDPOINTS ===

    @router.get(
        f"/api/{config.api_version}/agents", summary="List all agents", tags=["Agents Management"]
    )
    @handle_api_errors
    @log_api_call
    async def list_agents():
        """List all registered agents with their information."""
        agents_info = agent_registry.list_agents()
        return APIResponse(data=agents_info, message=f"Found {len(agents_info)} registered agents")

    @router.get(
        f"/api/{config.api_version}/agents/{{agent_name}}/health",
        summary="Agent health check",
        tags=["Agents Management"],
    )
    @handle_api_errors
    @log_api_call
    async def agent_health_check(agent_name: str = Path(..., description="Agent name")):
        """Check the health of a specific agent."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_name}' not found"
            )

        health_info = await agent.health_check()
        return APIResponse(data=health_info, message="Health check completed")

    @router.get(
        f"/api/{config.api_version}/agents/{{agent_name}}/schema",
        summary="Get agent input schema",
        tags=["Agents Management"],
    )
    @handle_api_errors
    @log_api_call
    async def get_agent_schema(agent_name: str = Path(..., description="Agent name")):
        """Get the input schema information for a specific agent."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_name}' not found"
            )

        schema_info = get_agent_schema_info(agent_name)
        return APIResponse(
            data=schema_info, message=f"Schema information for agent '{agent_name}'"
        )

    @router.get(
        f"/api/{config.api_version}/agents/{{agent_name}}/prompts",
        summary="List agent prompts",
        tags=["Prompts Management"],
    )
    @handle_api_errors
    @log_api_call
    async def list_agent_prompts(agent_name: str = Path(..., description="Agent name")):
        """List all available prompt versions for an agent."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_name}' not found"
            )

        versions = agent.prompt_manager.list_versions()
        prompts_info = []

        for version in versions:
            prompt_info = agent.prompt_manager.get_prompt_info(version)
            if prompt_info:
                prompts_info.append(
                    PromptInfo(
                        version=prompt_info.version,
                        description=prompt_info.description,
                        author=prompt_info.author,
                        tags=prompt_info.tags,
                        created_at=prompt_info.created_at.isoformat(),
                    )
                )

        return APIResponse(data=prompts_info, message=f"Found {len(prompts_info)} prompt versions")

    @router.get(
        f"/api/{config.api_version}/agents/{{agent_name}}/prompts/{{version}}",
        summary="Get specific prompt",
        tags=["Prompts Management"],
    )
    @handle_api_errors
    @log_api_call
    async def get_agent_prompt(
        agent_name: str = Path(..., description="Agent name"),
        version: str = Path(..., description="Prompt version"),
    ):
        """Get a specific prompt version content."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_name}' not found"
            )

        prompt_content = agent.get_prompt(version)
        if not prompt_content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt version '{version}' not found for agent '{agent_name}'",
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
                    created_at=prompt_info.created_at.isoformat(),
                )
                if prompt_info
                else None,
            },
            message=f"Retrieved prompt version '{version}'",
        )

    # === UNIVERSAL AGENT ENDPOINTS ===

    @router.post(
        f"/api/{config.api_version}/agents/{{agent_name}}/invoke",
        summary="Universal agent invocation with dynamic validation",
        tags=["Agents"],
        response_model=UniversalAgentResponse,
    )
    @handle_api_errors
    @log_api_call
    @rate_limit(max_calls=config.rate_limit_calls, window_seconds=config.rate_limit_window)
    async def invoke_agent_universal(
        agent_name: str = Path(..., description="Agent name"), request: UniversalAgentInput = ...
    ):
        """Universal endpoint that can invoke any agent with proper input validation."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_name}' not found"
            )

        async with agent_context(agent_name):
            # Validate input against agent's specific schema
            try:
                validated_input = validate_agent_input(agent_name, request.payload)
            except HTTPException:
                raise  # Re-raise validation errors
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Input validation error for agent '{agent_name}': {str(e)}",
                )

            # Execute the agent with validated input
            try:
                result = await agent.run(validated_input, streaming=request.streaming)

                return UniversalAgentResponse(
                    output=result,
                    agent=agent_name,
                    prompt_version=request.prompt_version,
                    streaming=request.streaming,
                    metadata=request.metadata,
                    validation_info={
                        "input_schema": f"{agent_name.title()}Input",
                        "validation_passed": True,
                    },
                )
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent execution failed: {str(e)}",
                )

    # === AGENT INTERACTION ENDPOINTS ===

    @router.post(
        f"/api/{config.api_version}/agents/{{agent_name}}/chat",
        summary="Chat with agent",
        tags=["Agents"],
        response_model=Union[AgentResponse, APIResponse],
    )
    @handle_api_errors
    @log_api_call
    @rate_limit(max_calls=config.rate_limit_calls, window_seconds=config.rate_limit_window)
    @validate_streaming_support
    async def chat_with_agent(
        agent_name: str, request: AgentRequest, background_tasks: BackgroundTasks
    ):
        """Chat with a specific agent with support for streaming."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_name}' not found"
            )

        async with agent_context(agent_name):
            # Format the prompt
            formatted_prompt = agent.format_prompt(
                version=request.prompt_version,
                input=request.input,
                context=request.context or "",
                history="",  # Could be expanded to include conversation history
            )

            if not formatted_prompt:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Prompt version '{request.prompt_version}' not available",
                )

            # Override LLM config with request parameters
            generation_kwargs = {}
            if request.temperature is not None:
                generation_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                generation_kwargs["max_tokens"] = request.max_tokens

            # Generate response
            if request.streaming:
                response_generator = await agent.generate_response(
                    formatted_prompt, streaming=True, **generation_kwargs
                )
                return await create_streaming_response(
                    response_generator, agent_name, media_type="text/event-stream"
                )
            else:
                response_content = await agent.generate_response(
                    formatted_prompt, streaming=False, **generation_kwargs
                )

                return APIResponse(
                    data=AgentResponse(
                        output=response_content,
                        agent=agent_name,
                        prompt_version=request.prompt_version,
                        streaming=False,
                        metadata=request.metadata,
                    ),
                    message="Response generated successfully",
                )

    # === AGENT CAPABILITIES ENDPOINTS ===

    @router.get(
        f"/api/{config.api_version}/agents/{{agent_name}}/capabilities",
        summary="Get agent capabilities",
        tags=["Agents Management"],
    )
    @handle_api_errors
    @log_api_call
    async def get_agent_capabilities(agent_name: str = Path(..., description="Agent name")):
        """Get the capabilities and features of a specific agent."""
        agent = agent_registry.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_name}' not found"
            )

        # Get basic agent information
        capabilities = {
            "name": agent_name,
            "type": type(agent).__name__,
            "streaming_supported": hasattr(agent, "supports_streaming")
            and agent.supports_streaming,
            "prompt_versions": [],
            "input_schema": get_agent_schema_info(agent_name),
            "features": {
                "batch_processing": hasattr(agent, "process_batch"),
                "async_execution": True,
                "context_aware": True,
                "configurable": hasattr(agent, "config"),
            },
        }

        # Get available prompt versions if available
        try:
            if hasattr(agent, "prompt_manager"):
                capabilities["prompt_versions"] = agent.prompt_manager.list_versions()
        except Exception:
            capabilities["prompt_versions"] = ["v1"]  # Default

        return APIResponse(data=capabilities, message=f"Capabilities for agent '{agent_name}'")

    # === UTILITY FUNCTIONS ===

    async def _process_queued_request(agent, request: AgentRequest, request_id: str):
        """Process a queued agent request."""
        try:
            logger.info(f"Processing queued request {request_id}")
            await agent.run(request, streaming=False)
            # In a real implementation, you'd store results in a database or cache
            logger.info(f"Completed queued request {request_id}")
        except Exception as e:
            logger.error(f"Failed to process queued request {request_id}: {e}")

    # Include agent-specific routers
    _include_agent_routers(router)

    return router


def _include_agent_routers(main_router: APIRouter) -> None:
    """Automatically include agent-specific routers for all agents with proper tags."""
    try:
        # Import the agents package
        agents_package = importlib.import_module(config.agents_package)

        # Walk through all agent modules (both old and new patterns)
        for _, module_name, _ in pkgutil.iter_modules(
            agents_package.__path__, agents_package.__name__ + "."
        ):
            agent_name = module_name.split(".")[-1]

            # Handle old pattern: agent_alpha, agent_beta (convert to alpha, beta)
            if agent_name.startswith("agent_"):
                actual_agent_name = agent_name.replace("agent_", "")
            else:
                # Handle new simplified pattern: alpha, beta, gamma, etc.
                actual_agent_name = agent_name

            try:
                # Import each agent's API module
                api_module = importlib.import_module(f"{module_name}.api")

                # Get the router from the module
                if hasattr(api_module, "router"):
                    route_prefix = f"/api/{config.api_version}/{actual_agent_name}"
                    agent_tag = actual_agent_name.title()

                    main_router.include_router(
                        api_module.router, prefix=route_prefix, tags=[agent_tag]
                    )
                    logger.info(f"Included API router for agent: {actual_agent_name}")

            except Exception as e:
                logger.debug(f"No API router for {module_name}: {e}")

    except Exception as e:
        logger.error(f"Failed to discover agent API routers: {e}")
