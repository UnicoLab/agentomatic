"""REST endpoint generation for pipelines.

Auto-generates FastAPI endpoints for each discovered pipeline,
mirroring how ``router_factory`` works for agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentomatic.core.registry import AgentRegistry
    from agentomatic.endpoints.registry import EndpointRegistry

    from .engine import PipelineEngine
    from .models import PipelineConfig


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class PipelineRunRequest(BaseModel):
    """Request to execute a pipeline."""

    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for the pipeline",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class PipelineRunResponse(BaseModel):
    """Response from a pipeline execution."""

    pipeline_name: str
    status: str
    output: dict[str, Any] = Field(default_factory=dict)
    steps: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None


class PipelineInfo(BaseModel):
    """Summary info about a pipeline."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: list[str] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)


class PipelineValidationResponse(BaseModel):
    """Response from pipeline validation."""

    pipeline_name: str
    valid: bool
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Router creation
# ---------------------------------------------------------------------------


def create_pipeline_router(
    pipelines: dict[str, PipelineConfig],
    registry: AgentRegistry,
    sub_pipelines: dict[str, PipelineConfig] | None = None,
    endpoints: EndpointRegistry | None = None,
) -> APIRouter:
    """Create REST endpoints for all discovered pipelines.

    Generates:
        GET  /pipelines                    — list all pipelines
        POST /pipelines/{name}/run         — execute a pipeline
        GET  /pipelines/{name}/config      — get pipeline config
        GET  /pipelines/{name}/validate    — pre-flight validation
        GET  /pipelines/{name}/visualize   — Mermaid diagram

    Args:
        pipelines: Dict of pipeline name → config.
        registry: Agent registry for resolving agents.
        sub_pipelines: Optional dict of sub-pipelines.

    Returns:
        A FastAPI ``APIRouter`` with pipeline endpoints.
    """
    router = APIRouter()
    all_pipelines = dict(pipelines)
    all_sub = dict(sub_pipelines or {})

    def _get_engine(name: str) -> PipelineEngine:
        """Resolve a pipeline engine by name."""
        from .engine import PipelineEngine

        config = all_pipelines.get(name)
        if config is None:
            raise HTTPException(404, f"Pipeline '{name}' not found")
        return PipelineEngine(config, registry, all_sub, endpoints=endpoints)

    @router.get(
        "/pipelines",
        response_model=list[PipelineInfo],
        summary="List all pipelines",
    )
    async def list_pipelines() -> list[PipelineInfo]:
        """List all discovered pipelines."""
        infos = []
        for name, config in sorted(all_pipelines.items()):
            infos.append(
                PipelineInfo(
                    name=config.name,
                    description=config.description,
                    version=config.version,
                    steps=config.step_names,
                    agents_used=sorted(config.get_agent_names()),
                )
            )
        return infos

    @router.post(
        "/pipelines/{name}/run",
        response_model=PipelineRunResponse,
        summary="Execute a pipeline",
    )
    async def run_pipeline(name: str, request: PipelineRunRequest) -> PipelineRunResponse:
        """Execute a pipeline with the given input."""
        engine = _get_engine(name)

        # Validate first
        errors = engine.validate()
        if errors:
            raise HTTPException(
                422,
                detail={
                    "message": "Pipeline validation failed",
                    "errors": errors,
                },
            )

        logger.info(f"🔁 Pipeline API: running '{name}'")
        result = await engine.run(request.input)

        return PipelineRunResponse(
            pipeline_name=result.pipeline_name,
            status=result.status.value,
            output=result.output,
            steps={k: v.model_dump() for k, v in result.steps.items()},
            duration_ms=result.duration_ms,
            error=result.error,
        )

    @router.get(
        "/pipelines/{name}/config",
        summary="Get pipeline configuration",
    )
    async def get_pipeline_config(name: str) -> dict[str, Any]:
        """Get the configuration of a pipeline."""
        config = all_pipelines.get(name)
        if config is None:
            raise HTTPException(404, f"Pipeline '{name}' not found")
        return config.model_dump()

    @router.get(
        "/pipelines/{name}/validate",
        response_model=PipelineValidationResponse,
        summary="Validate a pipeline",
    )
    async def validate_pipeline(
        name: str,
    ) -> PipelineValidationResponse:
        """Pre-flight validation of a pipeline."""
        engine = _get_engine(name)
        errors = engine.validate()
        return PipelineValidationResponse(
            pipeline_name=name,
            valid=len(errors) == 0,
            errors=errors,
        )

    @router.get(
        "/pipelines/{name}/visualize",
        summary="Get pipeline Mermaid diagram",
    )
    async def visualize_pipeline(name: str) -> dict[str, str]:
        """Get a Mermaid diagram of the pipeline."""
        engine = _get_engine(name)
        return {"mermaid": engine.visualize()}

    return router
