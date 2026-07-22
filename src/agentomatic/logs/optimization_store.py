"""Auditable persistence for prompt-fit / retrain artefacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.logs.recorder import truncate_for_storage

if TYPE_CHECKING:
    from agentomatic.optimize.config import PromptFitResult
    from agentomatic.storage.base import BaseStore


class OptimizationRunStore:
    """Thin facade over :class:`~agentomatic.storage.base.BaseStore` for fit runs.

    Keeps retrain history (scores, prompt versions, learnings) auditable in
    the same SQL/memory store used for threads and invocation logs.
    """

    def __init__(self, store: BaseStore | None) -> None:
        """Bind to an optional store.

        Args:
            store: Persistence backend. When ``None``, writes are no-ops.
        """
        self._store = store

    @property
    def enabled(self) -> bool:
        """Return ``True`` when a store is available."""
        return self._store is not None

    async def save_run(
        self,
        *,
        experiment_id: str,
        agent_name: str,
        baseline_score: float | None = None,
        best_score: float | None = None,
        prompt_versions: dict[str, Any] | None = None,
        score_history: list[Any] | None = None,
        learnings: list[Any] | None = None,
        artefacts: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Persist one optimization/retrain run.

        Returns:
            Stored run dict, or ``None`` when the store is unavailable.
        """
        if self._store is None:
            return None
        try:
            return await self._store.create_optimization_run(
                experiment_id=experiment_id,
                agent_name=agent_name,
                baseline_score=baseline_score,
                best_score=best_score,
                prompt_versions=truncate_for_storage(prompt_versions or {}),
                score_history=truncate_for_storage(score_history or []),
                learnings=truncate_for_storage(learnings or []),
                artefacts=truncate_for_storage(artefacts or {}),
                metadata=truncate_for_storage(metadata or {}),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to persist optimization run '{}' for '{}': {}",
                experiment_id,
                agent_name,
                exc,
            )
            return None

    async def save_fit_result(self, result: PromptFitResult) -> dict[str, Any] | None:
        """Persist a :class:`~agentomatic.optimize.config.PromptFitResult`.

        Args:
            result: Completed fit result from :class:`PromptFitter`.

        Returns:
            Stored run dict, or ``None`` when unavailable.
        """
        prompt_versions: dict[str, Any] = {
            "baseline": getattr(result.baseline_config, "system_prompt", None),
            "best": getattr(result.best_config, "system_prompt", None),
        }
        if result.deployment_recommendation is not None:
            rec = result.deployment_recommendation
            prompt_versions["recommended_version"] = getattr(rec, "prompt_version", None)

        learnings: list[Any] = list(getattr(result, "prompt_history", None) or [])
        if not learnings:
            learnings = list(result.suggestions or [])
            for cluster in result.failure_clusters or []:
                if isinstance(cluster, dict):
                    label = cluster.get("label") or cluster.get("pattern")
                    fix = cluster.get("fix") or cluster.get("recommendation")
                    if label or fix:
                        learnings.append(f"{label}: {fix}" if fix else str(label))

        artefacts: dict[str, Any] = {
            "metric_deltas": result.metric_deltas,
            "absolute_improvement": result.absolute_improvement,
            "duration_seconds": result.duration_seconds,
            "n_trials": len(result.trials or []),
            "suggestions": list(result.suggestions or []),
        }
        for attr in ("holdout_score", "baseline_holdout_score", "generalization_gap"):
            if hasattr(result, attr):
                artefacts[attr] = getattr(result, attr)
        if result.deployment_recommendation is not None and hasattr(
            result.deployment_recommendation, "to_dict"
        ):
            artefacts["deployment_recommendation"] = result.deployment_recommendation.to_dict()

        return await self.save_run(
            experiment_id=result.experiment_id,
            agent_name=result.agent or "",
            baseline_score=result.baseline_score,
            best_score=result.best_score,
            prompt_versions=prompt_versions,
            score_history=list(result.score_history or result.history),
            learnings=learnings,
            artefacts=artefacts,
            metadata={"source": "prompt_fitter"},
        )

    async def get(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by id."""
        if self._store is None:
            return None
        return await self._store.get_optimization_run(run_id)

    async def list(
        self,
        *,
        agent_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List optimization runs with optional filters."""
        if self._store is None:
            return []
        return await self._store.list_optimization_runs(
            agent_name=agent_name,
            experiment_id=experiment_id,
            limit=limit,
            offset=offset,
        )
