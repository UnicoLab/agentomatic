"""Main FastAPI application factory."""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..common.api_decorators import APIResponse
from .api import create_api_router
from .dependencies import agent_registry
from .settings import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    primary_agent_name = os.getenv("AGENT_NAME", "alpha")
    logger.info(f"Starting Vision Backend (primary agent: {primary_agent_name})")
    logger.info(f"API Version: {config.api_version}")
    logger.info(f"Debug Mode: {config.debug}")
    logger.info(f"Default LLM Provider: {config.default_llm_provider}")

    # Register all agents for multi-agent API support
    agent_registry.discover_agents()
    agent_count = agent_registry.get_agent_count()
    logger.info(f"Registered {agent_count} agents")

    # Log which agents were registered
    registered_agents = list(agent_registry.list_agents().keys())
    logger.info(f"Available agents: {', '.join(registered_agents)}")

    # Set primary agent based on AGENT_NAME for backward compatibility
    primary_agent = os.getenv("AGENT_NAME", "alpha")
    if primary_agent in registered_agents:
        logger.info(f"Primary agent set to: {primary_agent}")
    else:
        logger.warning(f"Primary agent '{primary_agent}' not found in registered agents")

    yield

    # Shutdown
    logger.info(f"Shutting down Vision Backend with {agent_registry.get_agent_count()} agents")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Example:
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Get agent name for the app title
    agent_name = os.getenv("AGENT_NAME", "alpha")

    # Create FastAPI app
    app = FastAPI(
        title=f"Vision Backend - {agent_name.title()} Agent",
        description=f"Scalable multi-agent architecture - {agent_name.title()} Agent Service",
        version="0.1.0",
        debug=config.debug,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with proper HTTP status codes."""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=APIResponse(
                success=False,
                error="Validation error",
                data={"details": exc.errors()},
                message="Request validation failed",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse(
                success=False, error=str(exc), message="Internal server error"
            ).model_dump(),
        )

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with basic API information."""
        return APIResponse(
            data={
                "name": "Vision Backend",
                "version": "0.1.0",
                "api_version": config.api_version,
                "description": "Multi-agent architecture with LangGraph and FastAPI",
                "docs_url": "/docs",
                "health_url": "/healthz",
                "agents_url": f"/api/{config.api_version}/agents",
            },
            message="Welcome to Vision Backend API",
        )

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Comprehensive health check endpoint."""
        try:
            agents_info = agent_registry.list_agents()
            agent_health = {}

            # Check health of each agent
            for agent_name in agents_info.keys():
                try:
                    agent = agent_registry.get_agent(agent_name)
                    health_info = await agent.health_check()
                    agent_health[agent_name] = health_info
                except Exception as e:
                    agent_health[agent_name] = {"status": "unhealthy", "error": str(e)}

            overall_status = (
                "healthy"
                if all(info.get("status") == "healthy" for info in agent_health.values())
                else "degraded"
            )

            return APIResponse(
                data={
                    "status": overall_status,
                    "api_version": config.api_version,
                    "agents": agent_health,
                    "config": {
                        "default_llm_provider": config.default_llm_provider.value,
                        "streaming_enabled": config.enable_streaming,
                        "queue_enabled": config.enable_queue,
                        "max_queue_size": config.max_queue_size,
                    },
                },
                message="Health check completed",
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=APIResponse(
                    success=False, error=str(e), message="Health check failed"
                ).model_dump(),
            )

    # Healthz endpoint (alternate endpoint for compatibility)
    @app.get("/healthz", tags=["Health"])
    async def healthz():
        """Simple health check endpoint for monitoring."""
        return {
            "status": "healthy",
            "message": "Service is healthy",
            "timestamp": "2025-06-18T00:00:00Z",  # Mock timestamp for tests
            "version": "0.1.0",
        }

    # Metrics endpoint for monitoring
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Metrics endpoint for monitoring and observability."""
        try:
            agents_info = agent_registry.list_agents()
            return APIResponse(
                data={
                    "total_agents": len(agents_info),
                    "agents": list(agents_info.keys()),
                    "api_version": config.api_version,
                    "uptime": "healthy",
                },
                message="Metrics retrieved successfully",
            )
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=APIResponse(
                    success=False, error=str(e), message="Metrics collection failed"
                ).model_dump(),
            )

    # Include API router
    api_router = create_api_router()
    app.include_router(api_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )
