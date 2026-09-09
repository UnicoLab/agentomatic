"""HTTP surface for the unified task subsystem.

Mounts a single, uniform task board under ``{api_prefix}/tasks`` that works for
every resource type::

    POST   {api_prefix}/tasks                 submit work (202 + task_id)
    GET    {api_prefix}/tasks                 list/filter tasks
    GET    {api_prefix}/tasks/{id}            poll status + progress
    GET    {api_prefix}/tasks/{id}/result     fetch the result (409 if pending)
    GET    {api_prefix}/tasks/{id}/events     resumable SSE progress stream
    POST   {api_prefix}/tasks/{id}/cancel     request cancellation
    DELETE {api_prefix}/tasks/{id}            delete a terminal record
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .event_log import parse_last_event_id
from .manager import TaskInputValidationError, TaskManager
from .models import TargetType, TaskStatus

_TAG = "Tasks"


class TaskSubmitRequest(BaseModel):
    """Request body for submitting a task."""

    target_type: TargetType = Field(description="agent | plugin | pipeline | endpoint | ingestion")
    target: str = Field(description="Name of the resource to run.")
    input: Any = Field(default=None, description="Single input payload.")
    batch: list[Any] | None = Field(default=None, description="Batch of input payloads.")
    mode: Literal["async", "sync", "batch"] = Field(
        default="async",
        description=(
            "async | sync | batch. Task progress is streamed separately via "
            "GET /tasks/{task_id}/events."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = Field(default=None, description="Webhook for completion.")
    wait: bool = Field(default=False, description="Block until the task is terminal.")
    timeout: float | None = Field(default=None, description="Max seconds to wait when wait=True.")
    retry: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional task-level retry config, e.g. "
            "``{'max_attempts': 3, 'backoff': 'exponential', 'base_delay': 1.0}``."
        ),
    )


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
                    retry=request.retry,
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
                    retry=request.retry,
                )
        except TaskInputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
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
        # The record is gone; its retained events would otherwise linger until
        # evicted, and could be replayed for a task that no longer exists.
        await manager.forget_events(task_id)
        return {"task_id": task_id, "deleted": True}

    @router.get("/{task_id}/events", summary="Stream task progress (SSE, resumable)")
    async def stream_events(
        task_id: str,
        request: Request,
        since: int = Query(
            default=0,
            ge=0,
            description=(
                "Resume after this event sequence. Usually unnecessary — a browser "
                "EventSource replays its Last-Event-ID header automatically."
            ),
        ),
    ) -> StreamingResponse:
        """Stream a task's progress as Server-Sent Events, resumably.

        Every frame carries an ``id:`` holding the event's monotonic sequence.
        A client that loses the connection reconnects and is sent the events it
        missed before the stream goes live again — from the ``Last-Event-ID``
        header a browser sends on its own, or from an explicit ``?since=``.

        Replay depth is bounded by the configured event log. When a resume
        point has already been evicted the stream says so in a ``truncated``
        frame and follows it with the current snapshot, so a client is never
        left believing it has a complete history.
        """
        record = await manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

        resume_from = since or parse_last_event_id(request.headers.get("last-event-id"))

        async def event_stream() -> AsyncIterator[str]:
            # Subscribe *before* replaying, so events emitted during replay are
            # queued rather than lost in the gap between the two.
            queue = await manager.subscribe(task_id)
            try:
                resumption = await manager.resume(task_id, after=resume_from)
                cursor = resumption.cursor
                if resume_from <= 0:
                    # Read the high-water mark *before* the snapshot. An event
                    # landing between the two is then re-delivered from the
                    # live queue rather than skipped: duplicated state is
                    # harmless, a silently dropped event is not.
                    cursor = await manager.event_log.latest_sequence(task_id)

                if resumption.truncated:
                    yield _sse(
                        {
                            "event": "truncated",
                            "task_id": task_id,
                            "resumed_after": resume_from,
                            "detail": (
                                "Events before this point are no longer retained; "
                                "treat the following snapshot as authoritative."
                            ),
                        }
                    )
                for evt in resumption.replay:
                    yield _sse(evt.model_dump(), event_id=evt.sequence)

                # A fresh subscriber (or a truncated one) needs the current
                # state; a caught-up one already has it from the replay.
                snapshot = await manager.get(task_id)
                if snapshot is not None and (resume_from <= 0 or resumption.truncated):
                    # The snapshot carries the cursor it reflects, so a client
                    # that drops immediately after it can still resume exactly.
                    yield _sse(snapshot.public_dict(), event_id=cursor)
                if snapshot is not None and snapshot.status.is_terminal:
                    yield "data: [DONE]\n\n"
                    return

                while True:
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    # Replay and the live queue overlap by design; skip what
                    # the client has already been sent.
                    if evt.sequence and evt.sequence <= cursor:
                        continue
                    cursor = evt.sequence or cursor
                    yield _sse(evt.model_dump(), event_id=evt.sequence)
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


def _sse(payload: dict[str, Any], *, event_id: int | None = None) -> str:
    """Format a payload as an SSE frame.

    Args:
        payload: JSON-serialisable frame body.
        event_id: Sequence to publish as the SSE ``id:`` field. Browsers echo
            the last one they saw back in ``Last-Event-ID`` when they
            reconnect, which is what makes the stream resumable for free.

    Returns:
        The encoded frame.
    """
    body = f"data: {json.dumps(payload, default=str)}\n\n"
    if event_id:
        return f"id: {event_id}\n{body}"
    return body
