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
    make_agent_input_validator,
    make_endpoint_dispatcher,
    make_endpoint_input_validator,
    make_ingestion_dispatcher,
    make_ingestion_input_validator,
    make_pipeline_dispatcher,
    make_pipeline_input_validator,
    make_plugin_dispatcher,
    make_plugin_input_validator,
)
from .manager import TaskInputValidationError, TaskManager
from .models import (
    TargetType,
    TaskEvent,
    TaskProgress,
    TaskRecord,
    TaskRetryConfig,
    TaskStatus,
)
from .progress import (
    bind_task_context,
    get_task_context,
    install_task_progress_bridge,
    report_stage,
    report_stage_sync,
    reset_task_context,
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
    "TaskInputValidationError",
    "TaskProgress",
    "TaskRecord",
    "TaskRetryConfig",
    "TaskStatus",
    "TaskStore",
    "TaskSubmitRequest",
    "attach_execution_modes",
    "bind_task_context",
    "create_task_router",
    "get_task_context",
    "install_task_progress_bridge",
    "make_agent_dispatcher",
    "make_agent_input_validator",
    "make_endpoint_dispatcher",
    "make_endpoint_input_validator",
    "make_ingestion_dispatcher",
    "make_ingestion_input_validator",
    "make_pipeline_dispatcher",
    "make_pipeline_input_validator",
    "make_plugin_dispatcher",
    "make_plugin_input_validator",
    "report_stage",
    "report_stage_sync",
    "reset_task_context",
    "task_links",
]


def __getattr__(name: str) -> object:
    """Lazily expose the optional SQLAlchemy store without importing it eagerly."""
    if name == "SQLAlchemyTaskStore":
        from .sql_store import SQLAlchemyTaskStore

        return SQLAlchemyTaskStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
