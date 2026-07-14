"""Keras-style training primitives for the agent lifecycle.

This module provides the objects returned by and passed to
:meth:`BaseGraphAgent.fit` — a :class:`History` record, a :class:`Callback`
protocol (plus a batteries-included :class:`EarlyStopping`), and a small
:class:`Loss` abstraction that turns any metric (or callable) into a scalar
objective to minimise.

The design mirrors ``keras``: ``fit()`` returns a ``History`` whose
``.history`` attribute maps metric/loss names to per-epoch values, callbacks
receive ``on_epoch_end(epoch, logs)`` dicts, and ``EarlyStopping`` halts
training by flipping ``agent.stop_training``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import AgentExample, Metric


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class History:
    """Record of metric/loss values collected during ``fit()``.

    Attributes:
        history: Mapping of log key (e.g. ``"loss"``, ``"val_loss"``,
            ``"exact_match"``) to the list of values, one per epoch.
        epoch: The list of epoch indices recorded.
        params: Training parameters (epochs, optimizer, metric names, …).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.history: dict[str, list[float]] = {}
        self.epoch: list[int] = []
        self.params: dict[str, Any] = dict(params or {})

    def record(self, epoch: int, logs: dict[str, float]) -> None:
        """Append one epoch's ``logs`` to the history."""
        self.epoch.append(epoch)
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))

    def __getitem__(self, key: str) -> list[float]:
        return self.history[key]

    def __contains__(self, key: str) -> bool:
        return key in self.history

    def keys(self):  # noqa: ANN201
        """Return the recorded log keys."""
        return self.history.keys()

    def final(self, key: str) -> float | None:
        """Return the last recorded value for ``key`` (or ``None``)."""
        values = self.history.get(key)
        return values[-1] if values else None

    def best(self, key: str, mode: str = "max") -> tuple[int, float] | None:
        """Return the ``(epoch, value)`` of the best value for ``key``.

        Args:
            key: The log key to inspect.
            mode: ``"max"`` (higher is better) or ``"min"`` (lower is better).

        Returns:
            The best ``(epoch, value)`` pair, or ``None`` if ``key`` is absent.
        """
        values = self.history.get(key)
        if not values:
            return None
        chooser = max if mode == "max" else min
        best_value = chooser(values)
        idx = values.index(best_value)
        return self.epoch[idx], best_value

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "params": self.params,
            "epoch": list(self.epoch),
            "history": {k: list(v) for k, v in self.history.items()},
        }

    def summary(self) -> str:
        """Return a compact, human-readable training summary."""
        if not self.epoch:
            return "History(empty)"
        lines = [f"History over {len(self.epoch)} epoch(s):"]
        for key, values in self.history.items():
            mode = "min" if "loss" in key else "max"
            best = self.best(key, mode=mode)
            best_str = f" (best {best[1]:.4f} @ epoch {best[0] + 1})" if best else ""
            lines.append(f"  {key}: {values[-1]:.4f}{best_str}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return f"History(epochs={len(self.epoch)}, keys={sorted(self.history)})"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class Callback:
    """Base class for training callbacks (Keras-style hooks).

    Subclass and override any of the ``on_*`` hooks. The active agent is
    injected via :meth:`set_agent` before training, so callbacks can inspect
    or halt it (``self.agent.stop_training = True``).
    """

    def __init__(self) -> None:
        self.agent: Any | None = None
        self.params: dict[str, Any] = {}

    def set_agent(self, agent: Any) -> None:
        """Attach the agent being trained."""
        self.agent = agent

    def set_params(self, params: dict[str, Any]) -> None:
        """Attach the training parameters."""
        self.params = params

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        """Called once before training starts."""

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        """Called once after training ends."""

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Called at the end of each epoch with the epoch's ``logs``."""


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Log key to watch (default ``"loss"``).
        mode: ``"min"``, ``"max"``, or ``"auto"`` (inferred from ``monitor``).
        patience: Epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best: If true, no-op placeholder for API parity (config is not
            snapshotted); kept for forward compatibility.
    """

    def __init__(
        self,
        monitor: str = "loss",
        mode: str = "auto",
        patience: int = 0,
        min_delta: float = 0.0,
        restore_best: bool = False,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.restore_best = restore_best
        if mode == "auto":
            mode = "min" if "loss" in monitor else "max"
        self.mode = mode
        self._best: float | None = None
        self._wait = 0
        self.stopped_epoch: int | None = None

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        self._best = None
        self._wait = 0
        self.stopped_epoch = None

    def _is_improvement(self, current: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return current < self._best - self.min_delta
        return current > self._best + self.min_delta

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        if self.monitor not in logs:
            logger.warning(f"EarlyStopping: monitor '{self.monitor}' not in logs {sorted(logs)}")
            return
        current = logs[self.monitor]
        if self._is_improvement(current):
            self._best = current
            self._wait = 0
            return
        self._wait += 1
        if self._wait > self.patience:
            self.stopped_epoch = epoch
            if self.agent is not None:
                self.agent.stop_training = True
            logger.info(
                f"EarlyStopping: no improvement in '{self.monitor}' for "
                f"{self._wait} epoch(s) — stopping at epoch {epoch + 1}"
            )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class Loss:
    """Scalar training objective (lower is better).

    Concrete losses implement :meth:`compute`. Use :func:`resolve_loss` to
    coerce a metric or callable into a ``Loss``.
    """

    name: str = "loss"

    def compute(self, example: AgentExample, prediction: dict[str, Any]) -> float:
        """Return the (non-negative) loss for one prediction."""
        raise NotImplementedError


class MetricLoss(Loss):
    """Turn a 0..1 (higher-better) metric into a loss (``1 - score``)."""

    def __init__(self, metric: Metric, name: str | None = None) -> None:
        self.metric = metric
        self.name = name or f"{getattr(metric, 'name', 'metric')}_loss"

    def compute(self, example: AgentExample, prediction: dict[str, Any]) -> float:
        score = float(self.metric.score(example, prediction))
        return max(0.0, 1.0 - score)


class CallableLoss(Loss):
    """Wrap a ``(example, prediction) -> float`` callable as a loss."""

    def __init__(
        self,
        fn: Callable[[AgentExample, dict[str, Any]], float],
        name: str = "loss",
    ) -> None:
        self.fn = fn
        self.name = name

    def compute(self, example: AgentExample, prediction: dict[str, Any]) -> float:
        return float(self.fn(example, prediction))


def resolve_loss(obj: Any) -> Loss | None:
    """Coerce ``obj`` into a :class:`Loss`.

    Accepts ``None`` (→ ``None``), an existing ``Loss``, a metric-like object
    with ``.score`` (→ :class:`MetricLoss`), or a plain callable
    (→ :class:`CallableLoss`).
    """
    if obj is None:
        return None
    if isinstance(obj, Loss):
        return obj
    if hasattr(obj, "score"):
        return MetricLoss(obj)
    if callable(obj):
        return CallableLoss(obj, name=getattr(obj, "__name__", "loss"))
    raise TypeError(f"Cannot interpret {type(obj).__name__} as a Loss")
