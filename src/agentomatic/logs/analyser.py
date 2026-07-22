"""Optional LLM-based analyser for per-agent invocation logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.logs.recorder import truncate_for_storage

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec
    from agentomatic.storage.base import BaseStore

_DEFAULT_SAMPLE_LIMIT = 20
_DEFAULT_PAYLOAD_CHARS = 1_200
_ANALYSER_SYSTEM = (
    "You are an expert AI operations analyst reviewing agent invocation logs. "
    "Return ONLY valid JSON with keys: score (0-1 float), summary (string), "
    "status (one of: healthy, degraded, failing, unknown), "
    "recommendations (array of short actionable strings)."
)


@dataclass(slots=True)
class LogAnalysisResult:
    """Structured result of an LLM log analysis pass.

    Attributes:
        score: Quality/health score in ``[0, 1]`` when available.
        summary: Short narrative summary of recent behaviour.
        status: Coarse health status label.
        recommendations: Actionable improvement suggestions.
        metadata: Extra fields (sample size, model, etc.).
        analysis_id: Persistence id when saved.
        agent_name: Agent that was analysed.
    """

    score: float | None = None
    summary: str = ""
    status: str = "unknown"
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    analysis_id: str | None = None
    agent_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "id": self.analysis_id,
            "agent_name": self.agent_name,
            "score": self.score,
            "summary": self.summary,
            "status": self.status,
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }


class LogAnalyser:
    """Sample recent logs and ask an LLM for scoring + recommendations.

    Budget-aware: only the newest ``sample_limit`` logs are used, and each
    input/output payload is truncated before being sent to the model.
    """

    def __init__(
        self,
        store: BaseStore | None,
        *,
        llm: LLMSpec | None = None,
        sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
        payload_chars: int = _DEFAULT_PAYLOAD_CHARS,
    ) -> None:
        """Create an analyser bound to a store.

        Args:
            store: Persistence backend providing invocation logs.
            llm: Optional LLM spec/callable for :class:`LLMCaller`.
            sample_limit: Max recent logs to include in the prompt.
            payload_chars: Per-field truncation for analyser prompts.
        """
        self._store = store
        self._llm = llm
        self._sample_limit = max(1, sample_limit)
        self._payload_chars = max(200, payload_chars)

    async def analyse(
        self,
        agent_name: str,
        *,
        persist: bool = True,
    ) -> LogAnalysisResult:
        """Analyse recent logs for ``agent_name``.

        Args:
            agent_name: Agent whose logs should be reviewed.
            persist: When ``True``, save the result via the store.

        Returns:
            A :class:`LogAnalysisResult`.

        Raises:
            RuntimeError: If no store is configured.
        """
        if self._store is None:
            raise RuntimeError("Log analysis requires a configured store")

        logs = await self._store.list_invocation_logs(
            agent_name=agent_name,
            limit=self._sample_limit,
            offset=0,
        )
        if not logs:
            result = LogAnalysisResult(
                agent_name=agent_name,
                score=None,
                summary="No invocation logs available for analysis.",
                status="unknown",
                recommendations=["Enable logs_history and generate traffic first."],
                metadata={"sample_size": 0},
            )
            if persist:
                saved = await self._store.save_log_analysis(
                    agent_name=agent_name,
                    score=result.score,
                    summary=result.summary,
                    status=result.status,
                    recommendations=result.recommendations,
                    metadata=result.metadata,
                )
                result.analysis_id = saved.get("id")
            return result

        prompt_logs = [self._compact_log(entry) for entry in logs]
        llm_result = await self._call_llm(agent_name, prompt_logs)
        result = self._normalize_result(agent_name, llm_result, sample_size=len(logs))

        if persist:
            saved = await self._store.save_log_analysis(
                agent_name=agent_name,
                score=result.score,
                summary=result.summary,
                status=result.status,
                recommendations=result.recommendations,
                metadata=result.metadata,
            )
            result.analysis_id = saved.get("id")
        return result

    def _compact_log(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Shrink a log entry for the analyser prompt."""
        return {
            "id": entry.get("id"),
            "timestamp": entry.get("timestamp"),
            "endpoint": entry.get("endpoint"),
            "status": entry.get("status"),
            "duration_ms": entry.get("duration_ms"),
            "error": entry.get("error"),
            "thread_id": entry.get("thread_id"),
            "input": truncate_for_storage(
                entry.get("input") or {},
                max_chars=self._payload_chars,
                max_string=self._payload_chars,
            ),
            "output": truncate_for_storage(
                entry.get("output") or {},
                max_chars=self._payload_chars,
                max_string=self._payload_chars,
            ),
            "metadata": truncate_for_storage(
                entry.get("metadata") or {},
                max_chars=min(400, self._payload_chars),
                max_string=400,
            ),
        }

    async def _call_llm(
        self,
        agent_name: str,
        compact_logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Invoke the LLM and return a parsed JSON dict (or heuristic fallback)."""
        user_prompt = (
            f"Analyse the following {len(compact_logs)} recent invocation logs "
            f"for agent '{agent_name}'. Focus on errors, latency, input/output "
            f"quality, and recurring failure patterns.\n\n"
            f"LOGS_JSON:\n{json.dumps(compact_logs, default=str)}"
        )

        if self._llm is None:
            return self._heuristic_analysis(compact_logs)

        try:
            from agentomatic.optimize.llm_caller import LLMCaller

            data = await LLMCaller.call_with_json(
                self._llm,
                user_prompt,
                system_prompt=_ANALYSER_SYSTEM,
            )
            if isinstance(data, dict) and data:
                return data
            logger.warning("Log analyser LLM returned empty/invalid JSON; using heuristic")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Log analyser LLM call failed: {}; using heuristic", exc)

        return self._heuristic_analysis(compact_logs)

    @staticmethod
    def _heuristic_analysis(logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Cheap offline fallback when no LLM is available or it fails."""
        total = len(logs)
        errors = sum(1 for log in logs if log.get("status") == "error" or log.get("error"))
        suspended = sum(1 for log in logs if log.get("status") == "suspended")
        durations = [
            float(log["duration_ms"])
            for log in logs
            if isinstance(log.get("duration_ms"), (int, float))
        ]
        avg_ms = sum(durations) / len(durations) if durations else None
        error_rate = errors / total if total else 0.0
        score = max(0.0, min(1.0, 1.0 - error_rate - (0.1 if suspended else 0.0)))

        if error_rate >= 0.5:
            status = "failing"
        elif error_rate >= 0.15 or suspended:
            status = "degraded"
        else:
            status = "healthy"

        recommendations: list[str] = []
        if errors:
            recommendations.append(
                f"Investigate {errors}/{total} failed invocations and fix recurring errors."
            )
        if avg_ms is not None and avg_ms > 5_000:
            recommendations.append(
                f"Average latency is {avg_ms:.0f}ms — profile slow nodes/tools."
            )
        if not recommendations:
            recommendations.append("No critical issues detected in the sampled window.")

        summary = (
            f"Sampled {total} logs: {errors} errors, {suspended} suspended"
            + (f", avg {avg_ms:.0f}ms" if avg_ms is not None else "")
            + "."
        )
        return {
            "score": round(score, 3),
            "summary": summary,
            "status": status,
            "recommendations": recommendations,
            "heuristic": True,
        }

    @staticmethod
    def _normalize_result(
        agent_name: str,
        raw: dict[str, Any],
        *,
        sample_size: int,
    ) -> LogAnalysisResult:
        """Coerce LLM/heuristic output into :class:`LogAnalysisResult`."""
        score_raw = raw.get("score")
        score: float | None
        try:
            score = float(score_raw) if score_raw is not None else None
            if score is not None:
                score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = None

        recommendations = raw.get("recommendations") or []
        if isinstance(recommendations, str):
            recommendations = [recommendations]
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
        recommendations = [str(item) for item in recommendations]

        status = str(raw.get("status") or "unknown").lower()
        if status not in {"healthy", "degraded", "failing", "unknown"}:
            status = "unknown"

        metadata = {
            "sample_size": sample_size,
            "heuristic": bool(raw.get("heuristic", False)),
        }
        for key in ("model", "notes", "error_patterns"):
            if key in raw:
                metadata[key] = raw[key]

        return LogAnalysisResult(
            agent_name=agent_name,
            score=score,
            summary=str(raw.get("summary") or ""),
            status=status,
            recommendations=recommendations,
            metadata=metadata,
        )
