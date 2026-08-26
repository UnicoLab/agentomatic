"""Auto-generate REST endpoints for registered custom endpoints."""

from __future__ import annotations

import inspect
import json
import time
from typing import TYPE_CHECKING, Any, get_args, get_origin

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import ValidationError

from agentomatic.core.errors import client_safe_detail
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


def _is_collection(annotation: Any) -> bool:
    """Whether a query-model field needs all repeated values, not just one."""
    origin = get_origin(annotation)
    if origin in {list, set, tuple, frozenset}:
        return True
    return any(_is_collection(item) for item in get_args(annotation))


def _decode_json_query_value(value: Any) -> Any:
    """Decode the object/array representation used by URL query strings."""
    if isinstance(value, str) and value[:1] in {"{", "["}:
        try:
            return json.loads(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_decode_json_query_value(item) for item in value]
    return value


def _query_openapi_parameters(input_schema: type[Any]) -> list[dict[str, Any]]:
    """Build query parameter docs from a Pydantic model for runtime routes."""
    schema = input_schema.model_json_schema()
    definitions = schema.get("$defs", {})

    def inline_local_refs(value: Any) -> Any:
        if isinstance(value, list):
            return [inline_local_refs(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("$ref", "").startswith("#/$defs/"):
            name = value["$ref"].rsplit("/", 1)[-1]
            resolved = definitions.get(name)
            if isinstance(resolved, dict):
                return inline_local_refs(
                    {**resolved, **{k: v for k, v in value.items() if k != "$ref"}}
                )
        return {key: inline_local_refs(item) for key, item in value.items()}

    required = set(schema.get("required", []))
    return [
        {
            "name": name,
            "in": "query",
            "required": name in required,
            "schema": inline_local_refs(field_schema),
        }
        for name, field_schema in schema.get("properties", {}).items()
    ]


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
            raise HTTPException(
                status_code=500, detail=client_safe_detail(exc, context="Endpoint call failed")
            ) from exc
        finally:
            _observe_endpoint(endpoint.endpoint_name, status, time.perf_counter() - t0)

    def _apply_typed_signature(handler: Any, *, query: bool = False) -> None:
        """Expose the endpoint's Pydantic contract to FastAPI/OpenAPI.

        A JSON request body is correct for write operations. Browsers cannot
        send a body with ``GET`` or ``HEAD`` however, so read-only endpoint
        methods must be modelled as query parameters. This also lets Studio
        generate a usable form from the served OpenAPI document.
        """
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        params[0] = params[0].replace(
            annotation=input_schema,
            default=Query(...) if query else inspect.Parameter.empty,
        )
        setattr(
            handler,
            "__signature__",
            sig.replace(parameters=params, return_annotation=output_schema),
        )

    methods = {str(method).upper() for method in endpoint.methods}
    query_methods = sorted(method for method in methods if method in {"GET", "HEAD"})
    body_methods = sorted(method for method in methods if method not in {"GET", "HEAD"})
    if body_methods:
        _apply_typed_signature(call_endpoint)
        router.add_api_route(
            endpoint.path,
            call_endpoint,
            methods=body_methods,
            response_model=output_schema,
            summary=f"Invoke the '{endpoint.endpoint_name}' endpoint",
            description=endpoint.endpoint_description,
        )

    if query_methods:

        async def call_endpoint_from_query(request: Request) -> Any:
            """Validate browser-safe query parameters with the endpoint model."""
            raw_input: dict[str, Any] = {}
            for name, field in input_schema.model_fields.items():
                values = request.query_params.getlist(name)
                if not values:
                    continue
                raw_input[name] = _decode_json_query_value(
                    values if _is_collection(field.annotation) else values[-1]
                )
            try:
                endpoint_request = input_schema.model_validate(raw_input)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc
            return await call_endpoint(endpoint_request)

        router.add_api_route(
            endpoint.path,
            call_endpoint_from_query,
            methods=query_methods,
            response_model=output_schema,
            summary=f"Invoke the '{endpoint.endpoint_name}' endpoint",
            description=endpoint.endpoint_description,
            openapi_extra={"parameters": _query_openapi_parameters(input_schema)},
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
