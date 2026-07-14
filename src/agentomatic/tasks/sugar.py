"""Per-resource execution-mode sugar.

Every resource (agent, plugin, pipeline, endpoint, ingestor) exposes a
synchronous route. This module adds the *same* ergonomic asynchronous and batch
convenience routes on top of the unified task system, so callers get a
consistent ``<sync>``, ``<sync>/async``, and ``<sync>/batch`` surface everywhere
without each router re-implementing task submission.

- ``POST <base>/async`` — submit a single input, return ``202`` + a task id and
  links to poll/stream/cancel.
- ``POST <base>/batch`` — submit many inputs as one batch task with per-item
  progress.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import APIRouter

    from .manager import TaskManager
    from .models import TargetType


class BatchSubmitRequest(BaseModel):
    """Body for a batch execution request."""

    inputs: list[Any] = Field(default_factory=list, description="Input payloads to fan out.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Task metadata.")
    callback_url: str | None = Field(
        default=None, description="Optional webhook POSTed on completion."
    )
    batch_concurrency: int | None = Field(
        default=None, description="Override max concurrent items for this batch."
    )


def task_links(task_id: str, api_prefix: str) -> dict[str, str]:
    """Return the canonical hypermedia links for a submitted task."""
    base = f"{api_prefix}/tasks/{task_id}"
    return {
        "status": base,
        "events": f"{base}/events",
        "result": f"{base}/result",
        "cancel": f"{base}/cancel",
    }


def attach_execution_modes(
    router: APIRouter,
    *,
    task_manager: TaskManager | None,
    target_type: TargetType,
    target: str,
    base_path: str,
    input_schema: type[BaseModel] | None,
    api_prefix: str,
    summary_label: str,
    transform: Callable[[Any], Any] | None = None,
) -> None:
    """Register ``<base_path>/async`` and ``<base_path>/batch`` on ``router``.

    Args:
        router: The router to mount onto.
        task_manager: The platform task manager (no-op if ``None``).
        target_type: Which dispatcher to route to.
        target: Resource name passed to the dispatcher.
        base_path: The resource's sync route (e.g. ``/invoke``, ``/predict``).
        input_schema: Pydantic model for the single-input body (typed OpenAPI).
        api_prefix: API prefix, used to build task links.
        summary_label: Human label used in OpenAPI summaries.
        transform: Optional payload transform applied before submission (used by
            pipelines to unwrap the ``input`` field).
    """
    if task_manager is None:
        return

    _transform = transform or (lambda p: p)

    async def async_endpoint(request: Any, _target: str = target) -> dict[str, Any]:
        payload = request.model_dump() if hasattr(request, "model_dump") else request
        record = await task_manager.submit(
            target_type, _target, input=_transform(payload), mode="async"
        )
        data = record.public_dict()
        data["links"] = task_links(record.id, api_prefix)
        return data

    if input_schema is not None:
        sig = inspect.signature(async_endpoint)
        params = list(sig.parameters.values())
        params[0] = params[0].replace(annotation=input_schema)
        setattr(async_endpoint, "__signature__", sig.replace(parameters=params))

    router.add_api_route(
        f"{base_path}/async",
        async_endpoint,
        methods=["POST"],
        status_code=202,
        summary=f"{summary_label} (async task)",
    )

    async def batch_endpoint(
        request: BatchSubmitRequest, _target: str = target
    ) -> dict[str, Any]:
        record = await task_manager.submit(
            target_type,
            _target,
            batch=[_transform(item) for item in request.inputs],
            mode="batch",
            metadata=request.metadata,
            callback_url=request.callback_url,
            batch_concurrency=request.batch_concurrency,
        )
        data = record.public_dict()
        data["links"] = task_links(record.id, api_prefix)
        return data

    router.add_api_route(
        f"{base_path}/batch",
        batch_endpoint,
        methods=["POST"],
        status_code=202,
        summary=f"{summary_label} (batch task)",
    )
