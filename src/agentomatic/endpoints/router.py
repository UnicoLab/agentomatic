"""Auto-generate REST endpoints for registered custom endpoints."""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from agentomatic.endpoints.base import BaseEndpoint

if TYPE_CHECKING:
    from agentomatic.logs.recorder import InvocationLogRecorder


def _observe_endpoint(name: str, status: str, elapsed: float) -> None:
    """Emit best-effort Prometheus metrics for a custom endpoint call."""
    try:
        from agentomatic.observability.metrics import (
            ENDPOINT_CALL_COUNT,
            ENDPOINT_DURATION,
        )

        ENDPOINT_CALL_COUNT.labels(endpoint=name, status=status).inc()
        ENDPOINT_DURATION.labels(endpoint=name).observe(elapsed)
    except Exception:  # noqa: BLE001 - metrics are optional
        pass


def create_endpoint_router(
    endpoint: BaseEndpoint,
    *,
    task_manager: Any | None = None,
    api_prefix: str = "/api/v1",
    log_recorder: InvocationLogRecorder | None = None,
) -> APIRouter:
    """Create a FastAPI router for a specific custom endpoint.

    Generates:
        GET  {mount}/health   — readiness of the endpoint and upstreams
        GET  {mount}/info     — endpoint metadata
        *    {mount}{path}    — the main handler (typed by the endpoint schemas)
        POST {mount}{path}/async — submit as a background task (if tasks enabled)
        POST {mount}{path}/batch — batch submission (if tasks enabled)

    Args:
        endpoint: The endpoint instance to expose.
        task_manager: Optional task manager enabling async/batch modes.
        api_prefix: API prefix used to build task links.
        log_recorder: Optional invocation log recorder when logs_history is on.

    Returns:
        A configured :class:`~fastapi.APIRouter`.
    """
    router = APIRouter(tags=[f"Endpoint: {endpoint.endpoint_name}"])

    input_schema = endpoint.get_input_schema()
    output_schema = endpoint.get_output_schema()

    @router.get("/health", response_model=dict[str, Any])
    async def health_check() -> dict[str, Any]:
        """Report endpoint readiness."""
        return {
            "status": "ok" if endpoint.is_ready else "unready",
            "endpoint": endpoint.endpoint_name,
            "version": endpoint.endpoint_version,
        }

    @router.get("/info", response_model=dict[str, Any])
    async def endpoint_info() -> dict[str, Any]:
        """Return endpoint metadata."""
        return endpoint.info()

    async def call_endpoint(request: Any) -> Any:
        """Invoke the endpoint's handler."""
        t0 = time.perf_counter()
        status = "ok"
        try:
            result = await endpoint.handle(request)
            duration = (time.perf_counter() - t0) * 1000
            logger.debug(
                f"Endpoint '{endpoint.endpoint_name}' handled request in {duration:.2f}ms"
            )
            if log_recorder is not None:
                from agentomatic.logs.helpers import record_invocation

                await record_invocation(
                    resource_type="endpoint",
                    resource_name=endpoint.endpoint_name,
                    endpoint="handle",
                    input_data=request,
                    output_data=result,
                    metadata={"path": endpoint.path, "methods": list(endpoint.methods)},
                    duration_ms=round(duration, 2),
                    status="ok",
                    recorder=log_recorder,
                )
            return result
        except HTTPException as exc:
            status = "error"
            if log_recorder is not None:
                from agentomatic.logs.helpers import record_invocation

                await record_invocation(
                    resource_type="endpoint",
                    resource_name=endpoint.endpoint_name,
                    endpoint="handle",
                    input_data=request,
                    error=str(exc.detail),
                    metadata={"path": endpoint.path, "methods": list(endpoint.methods)},
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    status="error",
                    recorder=log_recorder,
                )
            raise
        except Exception as exc:  # noqa: BLE001
            status = "error"
            logger.error(f"Endpoint '{endpoint.endpoint_name}' failed: {exc}")
            if log_recorder is not None:
                from agentomatic.logs.helpers import record_invocation

                await record_invocation(
                    resource_type="endpoint",
                    resource_name=endpoint.endpoint_name,
                    endpoint="handle",
                    input_data=request,
                    error=str(exc),
                    metadata={"path": endpoint.path, "methods": list(endpoint.methods)},
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    status="error",
                    recorder=log_recorder,
                )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _observe_endpoint(endpoint.endpoint_name, status, time.perf_counter() - t0)

    # Rewrite the signature so FastAPI extracts the correct Pydantic schemas.
    sig = inspect.signature(call_endpoint)
    params = list(sig.parameters.values())
    params[0] = params[0].replace(annotation=input_schema)
    setattr(
        call_endpoint,
        "__signature__",
        sig.replace(parameters=params, return_annotation=output_schema),
    )

    router.add_api_route(
        endpoint.path,
        call_endpoint,
        methods=endpoint.methods,
        response_model=output_schema,
        summary=f"Invoke the '{endpoint.endpoint_name}' endpoint",
        description=endpoint.endpoint_description,
    )

    # Async + batch execution modes via the task system.
    if task_manager is not None:
        from agentomatic.tasks.models import TargetType
        from agentomatic.tasks.sugar import attach_execution_modes

        attach_execution_modes(
            router,
            task_manager=task_manager,
            target_type=TargetType.ENDPOINT,
            target=endpoint.endpoint_name,
            base_path=endpoint.path.rstrip("/"),
            input_schema=input_schema,
            api_prefix=api_prefix,
            summary_label=f"Invoke the '{endpoint.endpoint_name}' endpoint",
        )

    return router
