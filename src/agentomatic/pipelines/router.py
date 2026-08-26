"""REST endpoint generation for pipelines.

Auto-generates FastAPI endpoints for each discovered pipeline,
mirroring how ``router_factory`` works for agents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from agentomatic.tasks.sugar import BatchSubmitRequest

from .loader import PipelineLoader, pipeline_to_yaml
from .validation import validate_pipeline_draft

if TYPE_CHECKING:
    from agentomatic.core.registry import AgentRegistry
    from agentomatic.endpoints.registry import EndpointRegistry
    from agentomatic.ingestion.registry import IngestionRegistry
    from agentomatic.logs.recorder import InvocationLogRecorder
    from agentomatic.plugins.registry import PluginRegistry

    from .engine import PipelineEngine
    from .models import PipelineConfig


# Pipeline names must be safe to use as file names.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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


class PipelineDraftRequest(BaseModel):
    """Payload for the stateless builder endpoints.

    Exactly one of ``pipeline`` / ``yaml`` must be provided.
    """

    pipeline: dict[str, Any] | None = Field(
        default=None,
        description="Pipeline config as a JSON object (same shape as the YAML).",
    )
    yaml: str | None = Field(
        default=None,
        description="Pipeline config as a YAML string.",
    )


class PipelineDraftValidationResponse(BaseModel):
    """Result of validating a pipeline draft without saving it."""

    name: str = ""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    step_count: int = 0


class PipelineSaveResponse(BaseModel):
    """Result of creating/updating a pipeline via the builder API."""

    name: str
    path: str = ""
    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    step_count: int = 0


# ---------------------------------------------------------------------------
# Router creation
# ---------------------------------------------------------------------------


def create_pipeline_router(
    pipelines: dict[str, PipelineConfig],
    registry: AgentRegistry,
    sub_pipelines: dict[str, PipelineConfig] | None = None,
    endpoints: EndpointRegistry | None = None,
    ingestors: IngestionRegistry | None = None,
    plugins: PluginRegistry | None = None,
    task_manager: Any | None = None,
    api_prefix: str = "/api/v1",
    log_recorder: InvocationLogRecorder | None = None,
    pipelines_dir: Path | None = None,
    pipeline_paths: dict[str, Path] | None = None,
) -> APIRouter:
    """Create REST endpoints for all discovered pipelines.

    Generates:
        GET    /pipelines                  — list all pipelines
        POST   /pipelines/validate-draft   — stateless validation of a draft
        POST   /pipelines/{name}           — create/update a pipeline (YAML)
        DELETE /pipelines/{name}           — delete a pipeline
        POST   /pipelines/{name}/run       — execute a pipeline
        GET    /pipelines/{name}/config    — get pipeline config
        GET    /pipelines/{name}/validate  — pre-flight validation
        GET    /pipelines/{name}/visualize — Mermaid diagram

    Args:
        pipelines: Dict of pipeline name → config.  Kept as a live
            reference so builder writes are visible to the task dispatcher
            and status API immediately.
        registry: Agent registry for resolving agents.
        sub_pipelines: Optional dict of sub-pipelines.
        pipelines_dir: Directory where new pipelines are persisted as
            ``<name>.yaml``.  When ``None``, save/delete return 400.
        pipeline_paths: Optional mapping of pipeline name → source file,
            used to update/delete the exact file a pipeline was discovered
            from (e.g. ``agents/*/pipeline.yaml``).

    Returns:
        A FastAPI ``APIRouter`` with pipeline endpoints.
    """
    router = APIRouter()
    # Live reference (not a copy): save/delete mutate this dict so the task
    # dispatcher and status/health APIs see builder changes immediately.
    all_pipelines = pipelines
    all_sub = dict(sub_pipelines or {})
    _pipeline_paths: dict[str, Path] = dict(pipeline_paths or {})
    _pipelines_dir = Path(pipelines_dir) if pipelines_dir is not None else None

    def _engine_for(config: PipelineConfig) -> PipelineEngine:
        """Build an engine for an arbitrary (possibly unsaved) config."""
        from .engine import PipelineEngine

        # Every served pipeline doubles as a sub-pipeline, so `sub_pipeline`
        # steps can compose what the platform already discovered. Read
        # `all_pipelines` here rather than at router-build time so a pipeline
        # saved through the builder is immediately referencable. Explicit
        # `sub_pipelines` win on a name clash.
        return PipelineEngine(
            config,
            registry,
            {**all_pipelines, **all_sub},
            endpoints=endpoints,
            ingestors=ingestors,
            plugins=plugins,
        )

    def _get_engine(name: str) -> PipelineEngine:
        """Resolve a pipeline engine by name."""
        config = all_pipelines.get(name)
        if config is None:
            raise HTTPException(404, f"Pipeline '{name}' not found")
        return _engine_for(config)

    def _parse_draft(request: PipelineDraftRequest) -> tuple[PipelineConfig | None, list[str]]:
        """Parse a draft payload into a config, collecting parse errors."""
        try:
            if request.pipeline is not None:
                return PipelineLoader.from_dict(request.pipeline), []
            if request.yaml is not None:
                return PipelineLoader.from_yaml_string(request.yaml), []
            return None, ["Provide exactly one of 'pipeline' (object) or 'yaml' (string)"]
        except Exception as exc:  # noqa: BLE001
            return None, [str(exc)]

    def _validate_config(config: PipelineConfig) -> tuple[list[str], list[str]]:
        """Run engine + draft validation, returning ``(errors, warnings)``."""
        errors = _engine_for(config).validate()
        draft_errors, draft_warnings = validate_pipeline_draft(config)
        return errors + draft_errors, draft_warnings

    @router.post(
        "/pipelines/validate-draft",
        response_model=PipelineDraftValidationResponse,
        summary="Validate a pipeline draft without saving",
    )
    async def validate_draft(request: PipelineDraftRequest) -> PipelineDraftValidationResponse:
        """Stateless validation of an arbitrary pipeline config.

        Accepts either a JSON object (``pipeline``) or a raw YAML string
        (``yaml``).  Returns registry + structural validation errors and
        advisory warnings so the builder can give instant feedback without
        persisting anything.
        """
        config, errors = _parse_draft(request)
        if config is None:
            return PipelineDraftValidationResponse(valid=False, errors=errors)

        warnings: list[str] = []
        if config.name in all_pipelines:
            warnings.append(f"Pipeline '{config.name}' already exists — saving will overwrite it")
        engine_errors, draft_warnings = _validate_config(config)
        errors += engine_errors
        warnings += draft_warnings
        return PipelineDraftValidationResponse(
            name=config.name,
            valid=not errors,
            errors=errors,
            warnings=warnings,
            step_count=len(config.steps),
        )

    @router.post(
        "/pipelines/{name}",
        response_model=PipelineSaveResponse,
        summary="Create or update a pipeline",
    )
    async def save_pipeline(name: str, request: PipelineDraftRequest) -> PipelineSaveResponse:
        """Create or update a pipeline from a draft payload.

        Validates the draft first; on success writes loader-compatible YAML
        to the platform's ``pipelines/`` directory (or updates the pipeline's
        existing source file in place) and makes it immediately available.
        """
        config, errors = _parse_draft(request)
        if config is None:
            raise HTTPException(422, detail={"message": "Invalid pipeline", "errors": errors})
        if config.name != name:
            raise HTTPException(
                422,
                detail={
                    "message": f"Path name '{name}' does not match pipeline name '{config.name}'",
                    "errors": [],
                },
            )
        if _pipelines_dir is None:
            raise HTTPException(
                400,
                detail={"message": "Pipeline persistence is not configured on this platform"},
            )
        if not _NAME_RE.fullmatch(name):
            raise HTTPException(
                422,
                detail={
                    "message": (
                        f"Invalid pipeline name '{name}' — use letters, digits, '_', '-', '.'"
                    ),
                    "errors": [],
                },
            )

        engine_errors, warnings = _validate_config(config)
        if engine_errors:
            raise HTTPException(
                422,
                detail={
                    "message": "Pipeline validation failed",
                    "errors": engine_errors,
                    "warnings": warnings,
                },
            )

        # Update the existing source file in place when known, otherwise
        # write to the canonical pipelines/ directory.
        target = _pipeline_paths.get(name) or (_pipelines_dir / f"{name}.yaml")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(pipeline_to_yaml(config), encoding="utf-8")
        except OSError as exc:
            # A production deployment often mounts source folders read-only.
            # Return an actionable client error instead of an uncaught 500;
            # the Compose deployment deliberately makes /app/pipelines
            # writable for Studio-authored definitions.
            raise HTTPException(
                409,
                detail={
                    "message": (
                        f"Pipeline storage is not writable at '{target}'. "
                        "Mount the pipeline directory read-write to save from Studio."
                    ),
                    "errors": [str(exc)],
                },
            ) from exc

        all_pipelines[name] = config
        _pipeline_paths[name] = target.resolve()
        logger.info(f"💾 Pipeline API: saved '{name}' -> {target}")
        return PipelineSaveResponse(
            name=name,
            path=str(target),
            warnings=warnings,
            step_count=len(config.steps),
        )

    @router.delete(
        "/pipelines/{name}",
        summary="Delete a pipeline",
    )
    async def delete_pipeline(name: str) -> dict[str, Any]:
        """Delete a pipeline and its source YAML file."""
        if _pipelines_dir is None:
            raise HTTPException(
                400,
                detail={"message": "Pipeline persistence is not configured on this platform"},
            )
        target = _pipeline_paths.get(name)
        if target is None:
            for candidate in (_pipelines_dir / f"{name}.yaml", _pipelines_dir / f"{name}.yml"):
                if candidate.exists():
                    target = candidate
                    break
        if target is None or not target.exists():
            raise HTTPException(404, f"Pipeline '{name}' not found")
        try:
            target.unlink()
        except OSError as exc:
            raise HTTPException(
                409,
                detail={
                    "message": (
                        f"Pipeline storage is not writable at '{target}'. "
                        "Mount the pipeline directory read-write to delete from Studio."
                    ),
                    "errors": [str(exc)],
                },
            ) from exc
        all_pipelines.pop(name, None)
        _pipeline_paths.pop(name, None)
        logger.info(f"🗑  Pipeline API: deleted '{name}' ({target})")
        return {"deleted": True, "name": name, "path": str(target)}

    @router.get(
        "/pipelines",
        response_model=list[PipelineInfo],
        summary="List all pipelines",
    )
    async def list_pipelines() -> list[PipelineInfo]:
        """List all discovered pipelines."""
        infos = []
        for _, config in sorted(all_pipelines.items()):
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

        if log_recorder is not None:
            from agentomatic.logs.helpers import record_invocation

            status_value = str(result.status.value).lower()
            status = (
                "error"
                if result.error or status_value in {"failed", "error", "cancelled"}
                else "ok"
            )
            await record_invocation(
                resource_type="pipeline",
                resource_name=result.pipeline_name,
                endpoint="run",
                input_data=request.input,
                output_data={
                    "output": result.output,
                    "steps": {k: v.model_dump() for k, v in result.steps.items()},
                    "status": result.status.value,
                },
                metadata=request.metadata or {},
                error=result.error,
                duration_ms=round(result.duration_ms, 2),
                status=status,
                recorder=log_recorder,
            )

        return PipelineRunResponse(
            pipeline_name=result.pipeline_name,
            status=result.status.value,
            output=result.output,
            steps={k: v.model_dump() for k, v in result.steps.items()},
            duration_ms=result.duration_ms,
            error=result.error,
        )

    if task_manager is not None:
        from agentomatic.tasks.models import TargetType
        from agentomatic.tasks.sugar import task_links

        @router.post(
            "/pipelines/{name}/run/async",
            status_code=202,
            summary="Execute a pipeline as a background task",
        )
        async def run_pipeline_async(name: str, request: PipelineRunRequest) -> dict[str, Any]:
            """Submit a pipeline run as a background task and return a task id."""
            if name not in all_pipelines:
                raise HTTPException(404, f"Pipeline '{name}' not found")
            record = await task_manager.submit(
                TargetType.PIPELINE,
                name,
                input=request.input,
                mode="async",
                metadata=request.metadata,
            )
            data = record.public_dict()
            data["links"] = task_links(record.id, api_prefix)
            return data

        @router.post(
            "/pipelines/{name}/run/batch",
            status_code=202,
            summary="Execute a pipeline over a batch of inputs",
        )
        async def run_pipeline_batch(name: str, request: BatchSubmitRequest) -> dict[str, Any]:
            """Submit many pipeline runs as one batch task."""
            if name not in all_pipelines:
                raise HTTPException(404, f"Pipeline '{name}' not found")
            record = await task_manager.submit(
                TargetType.PIPELINE,
                name,
                batch=request.inputs,
                mode="batch",
                metadata=request.metadata,
                callback_url=request.callback_url,
                batch_concurrency=request.batch_concurrency,
            )
            data = record.public_dict()
            data["links"] = task_links(record.id, api_prefix)
            return data

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
