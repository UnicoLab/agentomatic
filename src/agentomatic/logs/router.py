"""Cross-resource invocation log REST API.

Mounted at ``/api/v1/logs`` when ``logs_history`` is enabled. Per-agent
routes under ``/{agent}/logs`` remain for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from agentomatic.logs.runtime import RESOURCE_TYPES, normalize_resource_type

if TYPE_CHECKING:
    from agentomatic.storage.base import BaseStore


class AnalyzeLogsRequest(BaseModel):
    """Body for ``POST /logs/analyze``."""

    resource: str = Field(
        default="agent",
        description="Resource type: agent|plugin|pipeline|ingestion|endpoint",
    )
    name: str = Field(..., description="Resource name to analyse")
    persist: bool = True
    sample_limit: int = Field(default=20, ge=1, le=100)


def create_logs_router(
    *,
    store: BaseStore | None,
    logs_history: bool = False,
    allow_logsllm_analysis: bool = False,
) -> APIRouter:
    """Build the global logs list/get/analyze router."""
    router = APIRouter(tags=["Invocation Logs"])

    def _require_history() -> None:
        if not logs_history:
            raise HTTPException(
                400,
                detail={
                    "error": "Invocation log history is disabled. "
                    "Set logs_history=True / AGENTOMATIC_LOGS_HISTORY=1."
                },
            )
        if store is None:
            raise HTTPException(400, detail={"error": "Storage backend is not configured"})

    def _require_analysis() -> None:
        if not allow_logsllm_analysis:
            raise HTTPException(
                400,
                detail={
                    "error": "Log LLM analysis is disabled. "
                    "Set allow_logsllm_analysis=True / "
                    "AGENTOMATIC_ALLOW_LOGSLLM_ANALYSIS=1."
                },
            )

    def _parse_resource(resource: str | None) -> str | None:
        if resource is None or resource == "":
            return None
        try:
            return normalize_resource_type(resource)
        except ValueError as exc:
            raise HTTPException(
                400,
                detail={
                    "error": str(exc),
                    "allowed": sorted(RESOURCE_TYPES),
                },
            ) from exc

    @router.get("")
    async def list_logs(
        resource: str | None = None,
        name: str | None = None,
        thread_id: str | None = None,
        status: str | None = None,
        endpoint: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List invocation logs across resource types.

        Query params:
            resource: Filter by type (agent|plugin|pipeline|ingestion|endpoint).
            name: Filter by resource name.
        """
        _require_history()
        assert store is not None
        rtype = _parse_resource(resource)
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        logs = await store.list_invocation_logs(
            agent_name=name,
            resource_type=rtype,
            thread_id=thread_id,
            status=status,
            endpoint=endpoint,
            limit=limit,
            offset=offset,
        )
        total = await store.count_invocation_logs(
            agent_name=name,
            resource_type=rtype,
            thread_id=thread_id,
            status=status,
            endpoint=endpoint,
        )
        return {
            "resource": rtype,
            "name": name,
            "logs": logs,
            "count": len(logs),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @router.get("/analysis")
    async def get_latest_analysis(
        resource: str = "agent",
        name: str | None = None,
    ) -> dict[str, Any]:
        """Return the most recent LLM log analysis for a resource."""
        _require_analysis()
        _require_history()
        assert store is not None
        if not name:
            raise HTTPException(400, detail={"error": "Query param 'name' is required"})
        rtype = _parse_resource(resource) or "agent"
        analysis = await store.get_latest_log_analysis(name, resource_type=rtype)
        if not analysis:
            raise HTTPException(
                404,
                detail={
                    "error": f"No log analysis found for {rtype} '{name}'",
                },
            )
        return {"resource": rtype, "name": name, "analysis": analysis}

    @router.post("/analyze")
    async def analyze_logs(request: AnalyzeLogsRequest) -> dict[str, Any]:
        """Run LLM (or heuristic) analysis over recent invocation logs."""
        _require_analysis()
        _require_history()
        assert store is not None
        rtype = _parse_resource(request.resource) or "agent"
        from agentomatic.logs.analyser import LogAnalyser

        analyser = LogAnalyser(store, sample_limit=request.sample_limit)
        try:
            result = await analyser.analyse(
                resource_type=rtype,
                resource_name=request.name,
                persist=request.persist,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Log analysis failed for {}:{}: {}", rtype, request.name, exc)
            raise HTTPException(500, detail={"error": f"Log analysis failed: {exc}"}) from exc
        return {"resource": rtype, "name": request.name, "analysis": result.to_dict()}

    @router.get("/{log_id}")
    async def get_log(log_id: str) -> dict[str, Any]:
        """Fetch a single invocation log by id."""
        _require_history()
        assert store is not None
        entry = await store.get_invocation_log(log_id)
        if not entry:
            raise HTTPException(404, detail={"error": f"Log '{log_id}' not found"})
        return entry

    return router
