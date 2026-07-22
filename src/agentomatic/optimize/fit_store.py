"""Optional durable persistence for prompt-fit artefacts.

The fitter always writes JSON under ``experiment_dir``. When a platform
store is registered (via :func:`set_fit_store`), the same artefacts are
also written through :class:`~agentomatic.logs.OptimizationRunStore` so
retrain history survives restarts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.optimize.config import PromptFitResult
    from agentomatic.storage.base import BaseStore

_FIT_STORE: BaseStore | None = None


def set_fit_store(store: BaseStore | None) -> None:
    """Register the platform store used for optimization-run persistence.

    Args:
        store: Shared :class:`~agentomatic.storage.base.BaseStore`, or
            ``None`` to clear the registration.
    """
    global _FIT_STORE
    _FIT_STORE = store


def get_fit_store() -> BaseStore | None:
    """Return the currently registered fit store, if any."""
    return _FIT_STORE


async def persist_fit_result(
    result: PromptFitResult,
    *,
    experiment_dir: str | Path | None = None,
    store: BaseStore | None = None,
) -> dict[str, Any] | None:
    """Persist a fit result into the registered (or explicit) store.

    Args:
        result: Completed :class:`~agentomatic.optimize.config.PromptFitResult`.
        experiment_dir: Unused (kept for call-site compatibility); JSON is
            already written by the fitter.
        store: Optional override store. Defaults to the registered platform
            store from :func:`set_fit_store`.

    Returns:
        Stored optimization-run dict, or ``None`` when no store is available.
    """
    del experiment_dir  # JSON artefacts are written by PromptFitter itself.
    backend = store if store is not None else _FIT_STORE
    if backend is None:
        return None

    from agentomatic.logs.optimization_store import OptimizationRunStore

    run_store = OptimizationRunStore(backend)
    saved = await run_store.save_fit_result(result)
    if saved:
        logger.info(
            "💾 Optimization run persisted: id={} experiment={}",
            saved.get("id"),
            result.experiment_id,
        )
    return saved
