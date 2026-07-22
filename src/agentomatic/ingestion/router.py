"""Auto-generated REST endpoints for registered ingestors.

Each ingestor is exposed with a strictly-typed synchronous ``/run`` endpoint and
an asynchronous ``/run/async`` endpoint that submits the work to the task
manager (returning a pollable task id with live progress).
"""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from .context import NullIngestionContext
from .registry import IngestionRegistry

if TYPE_CHECKING:
    from agentomatic.logs.recorder import InvocationLogRecorder
    from agentomatic.tasks.manager import TaskManager

_TAG = "Ingestion"


def create_ingestion_router(
    registry: IngestionRegistry,
    *,
    task_manager: TaskManager | None = None,
    api_prefix: str = "/api/v1",
    log_recorder: InvocationLogRecorder | None = None,
) -> APIRouter:
    """Build the ingestion router bound to ``registry``."""
    router = APIRouter(tags=[_TAG])

    @router.get("", summary="List ingestors")
    async def list_ingestors() -> list[dict[str, Any]]:
        """List all registered ingestors."""
        return [ing.info() for ing in registry.list_ingestors().values()]

    for name, ingestor in registry.list_ingestors().items():
        _mount_ingestor(router, name, ingestor, task_manager, api_prefix, log_recorder)

    return router


def _mount_ingestor(
    router: APIRouter,
    name: str,
    ingestor: Any,
    task_manager: TaskManager | None,
    api_prefix: str,
    log_recorder: InvocationLogRecorder | None,
) -> None:
    """Mount the per-ingestor routes (info, health, run, run/async)."""
    input_schema = ingestor.get_input_schema()

    @router.get(f"/{name}/info", summary=f"Info for {name}")
    async def info(_ingestor: Any = ingestor) -> dict[str, Any]:
        """Return ingestor metadata."""
        return _ingestor.info()

    @router.get(f"/{name}/health", summary=f"Health for {name}")
    async def health(_ingestor: Any = ingestor) -> dict[str, Any]:
        """Return ingestor health."""
        return await _ingestor.health_check()

    async def run_endpoint(request: Any, _ingestor: Any = ingestor) -> Any:
        """Run the ingestor synchronously and return the result."""
        t0 = time.perf_counter()
        try:
            result = await _ingestor.run(request, NullIngestionContext())
        except Exception as exc:  # noqa: BLE001
            duration = (time.perf_counter() - t0) * 1000
            logger.error(f"Ingestor '{_ingestor.ingestor_name}' failed: {exc}")
            if log_recorder is not None:
                from agentomatic.logs.helpers import record_invocation

                await record_invocation(
                    resource_type="ingestion",
                    resource_name=_ingestor.ingestor_name,
                    endpoint="run",
                    input_data=request,
                    error=str(exc),
                    duration_ms=round(duration, 2),
                    status="error",
                    recorder=log_recorder,
                )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        duration = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Ingestor '{_ingestor.ingestor_name}' ran in {duration:.1f}ms"
        )
        if log_recorder is not None:
            from agentomatic.logs.helpers import record_invocation

            await record_invocation(
                resource_type="ingestion",
                resource_name=_ingestor.ingestor_name,
                endpoint="run",
                input_data=request,
                output_data=result,
                duration_ms=round(duration, 2),
                status="ok",
                recorder=log_recorder,
            )
        return result

    sig = inspect.signature(run_endpoint)
    params = list(sig.parameters.values())
    params[0] = params[0].replace(annotation=input_schema)
    setattr(run_endpoint, "__signature__", sig.replace(parameters=params))
    router.add_api_route(
        f"/{name}/run",
        run_endpoint,
        methods=["POST"],
        summary=f"Run ingestor '{name}' synchronously",
        description=ingestor.ingestor_description,
    )

    if task_manager is not None:
        from agentomatic.tasks.models import TargetType
        from agentomatic.tasks.sugar import attach_execution_modes

        attach_execution_modes(
            router,
            task_manager=task_manager,
            target_type=TargetType.INGESTION,
            target=name,
            base_path=f"/{name}/run",
            input_schema=input_schema,
            api_prefix=api_prefix,
            summary_label=f"Run ingestor '{name}'",
        )
