"""Data models for the unified task/execution subsystem.

A *task* is a uniform, trackable unit of work that wraps the execution of any
platform resource — an agent, an ML plugin, a pipeline, or a custom endpoint —
in synchronous, asynchronous (background), batch, or streaming modes.

The same :class:`TaskRecord` shape is returned for every resource type, so the
frontend can submit work, poll status, subscribe to progress, fetch the result,
or cancel — regardless of what is actually running underneath.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Lifecycle status of a task.

    The transitions are::

        QUEUED -> RUNNING -> SUCCEEDED
                          \\-> FAILED
        QUEUED/RUNNING    -> CANCELLED
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if no further transitions are possible."""
        return self in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED)


class TargetType(StrEnum):
    """The kind of platform resource a task executes."""

    AGENT = "agent"
    PLUGIN = "plugin"
    PIPELINE = "pipeline"
    ENDPOINT = "endpoint"
    INGESTION = "ingestion"


class TaskRetryConfig(BaseModel):
    """Optional retry configuration for a submitted task.

    Retries wrap the *entire* task execution: when the dispatcher raises
    an exception, the manager waits ``base_delay`` seconds (using the
    chosen backoff strategy) and re-invokes the dispatcher, up to
    ``max_attempts`` attempts total. Partial map-step checkpoints (see
    :attr:`TaskRecord.checkpoints`) are re-injected into the pipeline
    input on retry so already-completed items are not re-run.
    """

    max_attempts: int = Field(1, ge=1, le=10, description="Total attempts, including the first.")
    backoff: str = Field(
        default="exponential",
        description="Backoff strategy: fixed | linear | exponential.",
    )
    base_delay: float = Field(default=1.0, ge=0.0, le=60.0)


class TaskProgress(BaseModel):
    """Progress information for a running task.

    ``percent`` is ``None`` for indeterminate work (e.g. a single agent call);
    it is populated for determinate work such as batches (``done / total``) or
    pipelines (``completed_steps / total_steps``).
    """

    percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Completion percentage (None = indeterminate).",
    )
    message: str = Field(default="", description="Human-readable status message.")
    current: int = Field(default=0, description="Units of work completed.")
    total: int | None = Field(default=None, description="Total units of work, if known.")
    stage: str = Field(default="", description="Current stage/step identifier.")


class TaskEvent(BaseModel):
    """A single progress/status event emitted during task execution."""

    task_id: str
    event: str = Field(
        description="Event name: queued|started|progress|succeeded|failed|cancelled."
    )
    status: TaskStatus
    progress: TaskProgress | None = None
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    """A durable, uniform record describing one unit of work.

    This is the single response shape used across every execution mode and
    resource type. Persist it via a :class:`~agentomatic.tasks.store.TaskStore`
    so status survives process restarts and can be queried by a frontend.
    """

    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:16]}")
    target_type: TargetType
    target: str = Field(description="Name of the agent/plugin/pipeline/endpoint.")
    mode: str = Field(default="async", description="sync|async|batch|stream.")
    status: TaskStatus = TaskStatus.QUEUED
    progress: TaskProgress = Field(default_factory=TaskProgress)

    input: Any = Field(default=None, description="Single input payload.")
    batch: list[Any] | None = Field(default=None, description="Batch of input payloads.")
    result: Any = Field(default=None, description="Result once succeeded.")
    error: str | None = Field(default=None, description="Error message once failed.")

    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = Field(
        default=None,
        description="Optional webhook POSTed with the final record on completion.",
    )
    parent_id: str | None = Field(default=None, description="Parent task (for A2A / sub-tasks).")
    retry: TaskRetryConfig | None = Field(
        default=None,
        description="Optional task-level retry configuration.",
    )
    attempts: int = Field(
        default=0,
        description="Number of dispatcher invocations attempted so far.",
    )
    checkpoints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Lightweight per-step checkpoints. For map/fan-out steps this "
            "holds ``{step_name: {index: sub_result_dict}}`` so a restart can "
            "skip already-completed items."
        ),
    )

    @property
    def duration_ms(self) -> float | None:
        """Return wall-clock duration in milliseconds, if the task ran."""
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return (end - self.started_at) * 1000.0

    def public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict including derived fields for API responses."""
        data = self.model_dump()
        data["duration_ms"] = self.duration_ms
        return data
