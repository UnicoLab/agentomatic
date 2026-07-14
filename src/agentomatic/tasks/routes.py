"""HTTP surface for the unified task subsystem.

Mounts a single, uniform task board under ``{api_prefix}/tasks`` that works for
every resource type::

    POST   {api_prefix}/tasks                 submit work (202 + task_id)
    GET    {api_prefix}/tasks                 list/filter tasks
    GET    {api_prefix}/tasks/{id}            poll status + progress
    GET    {api_prefix}/tasks/{id}/result     fetch the result (409 if pending)
    GET    {api_prefix}/tasks/{id}/events     live SSE progress stream
    POST   {api_prefix}/tasks/{id}/cancel     request cancellation
    DELETE {api_prefix}/tasks/{id}            delete a terminal record
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .models import TargetType, TaskStatus

if TYPE_CHECKING:
    from .manager import TaskManager

_TAG = "Tasks"


class TaskSubmitRequest(BaseModel):
    """Request body for submitting a task."""

    target_type: TargetType = Field(description="agent | plugin | pipeline | endpoint")
    target: str = Field(description="Name of the resource to run.")
    input: Any = Field(default=None, description="Single input payload.")
    batch: list[Any] | None = Field(default=None, description="Batch of input payloads.")
    mode: str = Field(default="async", description="async | sync | batch | stream")
    metadata: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = Field(default=None, description="Webhook for completion.")
    wait: bool = Field(default=False, description="Block until the task is terminal.")
    timeout: float | None = Field(default=None, description="Max seconds to wait when wait=True.")


def create_task_router(manager: TaskManager) -> APIRouter:
    """Build the task board router bound to ``manager``."""
    router = APIRouter(tags=[_TAG])

    @router.post("", status_code=202, summary="Submit a task")
    async def submit_task(request: TaskSubmitRequest, response: Response) -> dict[str, Any]:
        """Submit any resource for sync/async/batch execution."""
        try:
            if request.wait or request.mode == "sync":
                record = await manager.submit_and_wait(
                    request.target_type,
                    request.target,
                    input=request.input,
                    batch=request.batch,
                    metadata=request.metadata,
                    callback_url=request.callback_url,
                    timeout=request.timeout,
                )
                response.status_code = 200
            else:
                record = await manager.submit(
                    request.target_type,
                    request.target,
                    input=request.input,
                    batch=request.batch,
                    mode=request.mode,
                    metadata=request.metadata,
                    callback_url=request.callback_url,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = record.public_dict()
        payload["links"] = _links(record.id)
        return payload

    @router.get("", summary="List tasks")
    async def list_tasks(
        status: TaskStatus | None = None,
        target_type: TargetType | None = None,
        target: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """List tasks, most recent first, with optional filters."""
        records = await manager.list(
            status=status,
            target_type=target_type,
            target=target,
            limit=limit,
            offset=offset,
        )
        return {
            "tasks": [r.public_dict() for r in records],
            "count": len(records),
            "total": await manager.count(),
        }

    @router.get("/{task_id}", summary="Get task status")
    async def get_task(task_id: str) -> dict[str, Any]:
        """Return the current status and progress of a task."""
        record = await manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        payload = record.public_dict()
        payload["links"] = _links(task_id)
        return payload

    @router.get("/{task_id}/result", summary="Get task result")
    async def get_result(task_id: str) -> Any:
        """Return the result of a succeeded task.

        Responds ``409`` while the task is still pending, and ``422`` if the
        task failed or was cancelled.
        """
        record = await manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        if not record.status.is_terminal:
            raise HTTPException(status_code=409, detail=f"Task is {record.status.value}")
        if record.status != TaskStatus.SUCCEEDED:
            raise HTTPException(
                status_code=422,
                detail=record.error or f"Task {record.status.value}",
            )
        return {"task_id": task_id, "result": record.result}

    @router.post("/{task_id}/cancel", summary="Cancel a task")
    async def cancel_task(task_id: str) -> dict[str, Any]:
        """Request cancellation of a queued or running task."""
        cancelled = await manager.cancel(task_id)
        if not cancelled:
            raise HTTPException(
                status_code=409,
                detail=f"Task '{task_id}' not found or already terminal",
            )
        return {"task_id": task_id, "status": "cancelling"}

    @router.delete("/{task_id}", summary="Delete a task record")
    async def delete_task(task_id: str) -> dict[str, Any]:
        """Delete a terminal task record."""
        record = await manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        if not record.status.is_terminal:
            raise HTTPException(status_code=409, detail="Cannot delete a running task")
        await manager.store.delete(task_id)
        return {"task_id": task_id, "deleted": True}

    @router.get("/{task_id}/events", summary="Stream task progress (SSE)")
    async def stream_events(task_id: str) -> StreamingResponse:
        """Stream live progress events for a task as Server-Sent Events."""
        record = await manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

        async def event_stream() -> Any:
            queue = await manager.subscribe(task_id)
            try:
                # Emit the current snapshot immediately.
                snapshot = await manager.get(task_id)
                if snapshot is not None:
                    yield _sse(snapshot.public_dict())
                    if snapshot.status.is_terminal:
                        yield "data: [DONE]\n\n"
                        return
                while True:
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield _sse(evt.model_dump())
                    if evt.status.is_terminal:
                        yield "data: [DONE]\n\n"
                        return
            finally:
                manager.unsubscribe(task_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _links(task_id: str) -> dict[str, str]:
    """Return HATEOAS-style relative links for a task."""
    return {
        "self": f"tasks/{task_id}",
        "result": f"tasks/{task_id}/result",
        "events": f"tasks/{task_id}/events",
        "cancel": f"tasks/{task_id}/cancel",
    }


def _sse(payload: dict[str, Any]) -> str:
    """Format a payload as an SSE ``data:`` frame."""
    return f"data: {json.dumps(payload, default=str)}\n\n"
