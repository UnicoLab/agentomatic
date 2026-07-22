"""Auto-generate REST endpoints for registered ML plugins."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from .ml import BaseMLPlugin

if TYPE_CHECKING:
    from agentomatic.logs.recorder import InvocationLogRecorder


def create_plugin_router(
    plugin: BaseMLPlugin,
    *,
    task_manager: Any | None = None,
    api_prefix: str = "/api/v1",
    log_recorder: InvocationLogRecorder | None = None,
) -> APIRouter:
    """Create a FastAPI router for a specific ML plugin."""
    router = APIRouter(tags=[f"Plugin: {plugin.plugin_name}"])

    input_schema = plugin.get_input_schema()
    output_schema = plugin.get_output_schema()

    @router.get("/health", response_model=dict[str, Any])
    async def health_check() -> dict[str, Any]:
        """Check if the plugin is loaded and healthy."""
        return {
            "status": "ok" if plugin.is_loaded else "unloaded",
            "plugin_name": plugin.plugin_name,
            "version": plugin.plugin_version,
            "loaded_at": plugin.loaded_at,
        }

    @router.get("/model_card", response_model=dict[str, Any])
    async def get_model_card() -> dict[str, Any]:
        """Retrieve the model card / metadata."""
        return plugin.model_card()

    @router.post("/reload", response_model=dict[str, Any])
    async def reload_plugin() -> dict[str, Any]:
        """Reload model weights from the current artifact pointer.

        Re-calls ``load_model()`` on the live registry instance so subsequent
        ``predict`` and pipeline ``plugin:`` steps use the fresh weights.
        """
        try:
            return await plugin.reload_model()
        except Exception as exc:
            logger.error("Reload failed for plugin '{}': {}", plugin.plugin_name, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Create the dynamic predict endpoint
    async def predict_endpoint(request: Any) -> Any:
        """Execute the plugin's prediction logic."""
        if not plugin.is_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Plugin '{plugin.plugin_name}' is not loaded yet.",
            )

        start_time = time.perf_counter()
        try:
            result = await plugin.predict(request)

            duration = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Plugin '{plugin.plugin_name}' inference completed in {duration:.2f}ms")
            if log_recorder is not None:
                from agentomatic.logs.helpers import record_invocation

                await record_invocation(
                    resource_type="plugin",
                    resource_name=plugin.plugin_name,
                    endpoint="predict",
                    input_data=request,
                    output_data=result,
                    duration_ms=round(duration, 2),
                    status="ok",
                    recorder=log_recorder,
                )
            return result

        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error during inference for plugin '{plugin.plugin_name}': {e}")
            if log_recorder is not None:
                from agentomatic.logs.helpers import record_invocation

                await record_invocation(
                    resource_type="plugin",
                    resource_name=plugin.plugin_name,
                    endpoint="predict",
                    input_data=request,
                    error=str(e),
                    duration_ms=round(duration, 2),
                    status="error",
                    recorder=log_recorder,
                )
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Dynamically adjust the signature so FastAPI extracts the correct Pydantic schemas
    import inspect

    sig = inspect.signature(predict_endpoint)
    params = list(sig.parameters.values())
    params[0] = params[0].replace(annotation=input_schema)
    setattr(
        predict_endpoint,
        "__signature__",
        sig.replace(parameters=params, return_annotation=output_schema),
    )

    router.add_api_route(
        "/predict",
        predict_endpoint,
        methods=["POST"],
        response_model=output_schema,
        summary=f"Run inference using {plugin.plugin_name}",
        description=plugin.plugin_description,
    )

    # Async + batch execution modes via the task system.
    if task_manager is not None:
        from agentomatic.tasks.models import TargetType
        from agentomatic.tasks.sugar import attach_execution_modes

        attach_execution_modes(
            router,
            task_manager=task_manager,
            target_type=TargetType.PLUGIN,
            target=plugin.plugin_name,
            base_path="/predict",
            input_schema=input_schema,
            api_prefix=api_prefix,
            summary_label=f"Run inference using {plugin.plugin_name}",
        )

    return router
