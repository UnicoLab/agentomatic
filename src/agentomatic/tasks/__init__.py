"""Unified task/execution subsystem.

Run any platform resource — agent, ML plugin, pipeline, or custom endpoint — in
synchronous, asynchronous (background), batch, or streaming modes with a single,
uniform, trackable :class:`TaskRecord`. Supports status polling, live SSE
progress, cancellation, batch fan-out, and completion webhooks.

Example::

    from agentomatic.tasks import TaskManager, TargetType
    from agentomatic.tasks.dispatchers import make_agent_dispatcher

    manager = TaskManager()
    manager.register_dispatcher(TargetType.AGENT, make_agent_dispatcher(registry))

    record = await manager.submit(TargetType.AGENT, "researcher", input={"query": "hi"})
    status = await manager.get(record.id)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .context import TaskContext
from .dispatchers import (
    TargetNotFoundError,
    make_agent_dispatcher,
    make_endpoint_dispatcher,
    make_ingestion_dispatcher,
    make_pipeline_dispatcher,
    make_plugin_dispatcher,
)
from .manager import TaskManager
from .models import (
    TargetType,
    TaskEvent,
    TaskProgress,
    TaskRecord,
    TaskRetryConfig,
    TaskStatus,
)
from .routes import TaskSubmitRequest, create_task_router
from .store import InMemoryTaskStore, TaskStore
from .sugar import BatchSubmitRequest, attach_execution_modes, task_links

if TYPE_CHECKING:
    from .sql_store import SQLAlchemyTaskStore

__all__ = [
    "BatchSubmitRequest",
    "InMemoryTaskStore",
    "SQLAlchemyTaskStore",
    "TargetNotFoundError",
    "TargetType",
    "TaskContext",
    "TaskEvent",
    "TaskManager",
    "TaskProgress",
    "TaskRecord",
    "TaskRetryConfig",
    "TaskStatus",
    "TaskStore",
    "TaskSubmitRequest",
    "attach_execution_modes",
    "create_task_router",
    "make_agent_dispatcher",
    "make_endpoint_dispatcher",
    "make_ingestion_dispatcher",
    "make_pipeline_dispatcher",
    "make_plugin_dispatcher",
    "task_links",
]


def __getattr__(name: str) -> object:
    """Lazily expose the optional SQLAlchemy store without importing it eagerly."""
    if name == "SQLAlchemyTaskStore":
        from .sql_store import SQLAlchemyTaskStore

        return SQLAlchemyTaskStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
