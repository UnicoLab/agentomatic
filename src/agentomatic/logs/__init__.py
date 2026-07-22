"""Invocation log history and optional LLM log analysis.

Enable via platform flags / env::

    AgentPlatform.from_folder(
        "agents/",
        store=SQLAlchemyStore("postgresql+asyncpg://…"),  # or sqlite+aiosqlite
        logs_history=True,
        allow_logsllm_analysis=True,
    )

Or::

    AGENTOMATIC_LOGS_HISTORY=1
    AGENTOMATIC_ALLOW_LOGSLLM_ANALYSIS=1
    DATABASE_URL=sqlite+aiosqlite:///./agentomatic.db   # any SQLAlchemy async URL

When a DB URL / MEMORY connection is available the platform auto-derives
``SQLAlchemyStore`` (Postgres, SQLite, …). MemoryStore is only a last-resort
fallback when no DB is configured.

Logs cover agents, plugins, pipelines, ingestion, and custom endpoints.
Cross-resource REST: ``GET /api/v1/logs?resource=plugin&name=…``.
Per-agent routes ``/{agent}/logs`` remain for backward compatibility.
"""

from __future__ import annotations

from agentomatic.logs.analyser import LogAnalyser, LogAnalysisResult
from agentomatic.logs.optimization_store import OptimizationRunStore
from agentomatic.logs.recorder import InvocationLogRecorder, truncate_for_storage
from agentomatic.logs.runtime import (
    RESOURCE_TYPES,
    get_invocation_log_recorder,
    set_invocation_log_recorder,
)

__all__ = [
    "RESOURCE_TYPES",
    "InvocationLogRecorder",
    "LogAnalyser",
    "LogAnalysisResult",
    "OptimizationRunStore",
    "get_invocation_log_recorder",
    "set_invocation_log_recorder",
    "truncate_for_storage",
]
