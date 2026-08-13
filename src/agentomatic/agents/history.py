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

import difflib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from .types import AgentExample

if TYPE_CHECKING:
    from .types import Metric


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


class EpochDiffCallback(Callback):
    """Print a full per-epoch report: loss + prompt diff + config changes.

    Snapshots the agent's compiled configuration at ``on_epoch_begin``
    (before the optimizer runs) and, at ``on_epoch_end``, renders a
    Keras-style report with:

    - the epoch's ``loss`` / ``val_loss`` (and improvement vs. previous epoch)
    - a unified diff of the system prompt (``-`` removed / ``+`` added lines)
    - a summary of config changes (temperature, few-shot count, …)
    - the current best prompt (first lines)

    Every epoch record is also stored in :attr:`per_epoch` (JSON-friendly)
    so scripts can persist the full change history — the proof that loss
    descends while the prompt evolves.

    Example::

        from agentomatic import EpochDiffCallback
        agent.fit(..., callbacks=[EpochDiffCallback(epochs=10)])
    """

    # Config keys tracked for the changes summary (order matters).
    _TRACKED_KEYS = (
        "temperature",
        "top_p",
        "max_tokens",
        "model_choice",
        "output_contract",
        "few_shot_examples",
    )
    _DEFAULT_LINE = ""

    def __init__(self, epochs: int = 1, prompt_key: str = "system_prompt") -> None:
        super().__init__()
        self.epochs = epochs
        self.prompt_key = prompt_key
        self.per_epoch: list[dict[str, Any]] = []
        self._snapshot: dict[str, Any] = {}
        self._previous_loss: float | None = None

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        self._snapshot = self._config_snapshot()

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        current = self._config_snapshot()
        changes = self._diff_snapshot(self._snapshot, current)
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        improvement: float | None = None
        if loss is not None and self._previous_loss is not None:
            improvement = float(self._previous_loss) - float(loss)
        self._previous_loss = float(loss) if loss is not None else self._previous_loss

        self._render(epoch, logs, changes, current, loss, val_loss, improvement)
        self.per_epoch.append(
            {
                "epoch": epoch,
                "loss": round(float(loss), 6) if loss is not None else None,
                "val_loss": round(float(val_loss), 6) if val_loss is not None else None,
                "improvement": round(improvement, 6) if improvement is not None else None,
                "changes": changes,
                "prompt": current.get(self.prompt_key, ""),
            }
        )

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        self._snapshot = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _config_snapshot(self) -> dict[str, Any]:
        """Capture the agent's current prompt + tracked config keys."""
        agent = self.agent
        snapshot: dict[str, Any] = {}
        if agent is None:
            return snapshot
        compiled = getattr(agent, "compiled_config", None) or {}
        if isinstance(compiled, dict):
            prompt = compiled.get(self.prompt_key)
            if isinstance(prompt, str) and prompt.strip():
                snapshot[self.prompt_key] = prompt
            for key in self._TRACKED_KEYS:
                if key in compiled:
                    snapshot[key] = compiled[key]
        attr_prompt = getattr(agent, self.prompt_key, None)
        if isinstance(attr_prompt, str) and attr_prompt.strip():
            snapshot.setdefault(self.prompt_key, attr_prompt)
        return snapshot

    def _diff_snapshot(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two snapshots → prompt diff + scalar/list change summary."""
        changes: dict[str, Any] = {"prompt_diff": [], "params": {}}
        old_prompt = str(before.get(self.prompt_key, "") or "")
        new_prompt = str(after.get(self.prompt_key, "") or "")
        if old_prompt != new_prompt:
            changes["prompt_diff"] = list(
                difflib.unified_diff(
                    old_prompt.splitlines(),
                    new_prompt.splitlines(),
                    fromfile="before",
                    tofile="after",
                    lineterm="",
                    n=1,
                )
            )
        for key in self._TRACKED_KEYS:
            old_val = before.get(key)
            new_val = after.get(key)
            if old_val == new_val:
                continue
            changes["params"][key] = {"old": old_val, "new": new_val}
        return changes

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        epoch: int,
        logs: dict[str, float],
        changes: dict[str, Any],
        current: dict[str, Any],
        loss: float | None,
        val_loss: float | None,
        improvement: float | None,
    ) -> None:
        total = max(1, self.epochs)
        parts: list[str] = [f"\n{'─' * 70}", f"Epoch {epoch + 1}/{total} — report"]

        # Keras-style loss line
        loss_bits = []
        if loss is not None:
            loss_bits.append(f"loss: {loss:.4f}")
        if val_loss is not None:
            loss_bits.append(f"val_loss: {val_loss:.4f}")
        if improvement is not None:
            arrow = "↓" if improvement > 1e-9 else ("↑" if improvement < -1e-9 else "→")
            loss_bits.append(f"improvement: {improvement:+.4f} {arrow}")
        parts.append("  " + "   ".join(loss_bits) if loss_bits else "  (no loss recorded)")

        # Prompt diff (added / removed lines)
        diff = changes.get("prompt_diff") or []
        if diff:
            parts.append("  ── prompt changes (what was modified) ──")
            for line in diff[:40]:
                if line.startswith("+") and not line.startswith("+++"):
                    parts.append(f"    \033[32m{line[:160]}\033[0m")
                elif line.startswith("-") and not line.startswith("---"):
                    parts.append(f"    \033[31m{line[:160]}\033[0m")
                elif line.startswith("@@"):
                    parts.append(f"    \033[90m{line[:160]}\033[0m")
                else:
                    parts.append(f"    {line[:160]}")
        else:
            parts.append("  ── prompt changes ── (unchanged)")

        # Config change summary
        params = changes.get("params") or {}
        if params:
            parts.append("  ── config changes ──")
            for key, delta in params.items():
                old_v, new_v = delta["old"], delta["new"]
                if isinstance(old_v, list) or isinstance(new_v, list):
                    parts.append(
                        f"    {key}: {len(old_v or [])} → {len(new_v or [])} items"
                    )
                else:
                    parts.append(f"    {key}: {old_v!r} → {new_v!r}")

        # Current best prompt (first lines)
        prompt = current.get(self.prompt_key, "")
        if prompt:
            first_lines = " | ".join(str(prompt).splitlines()[:2])[:200]
            parts.append(f"  ── current prompt ── {first_lines}")

        logger.info("\n".join(parts))


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
        return CallableLoss(
            cast(Callable[[AgentExample, dict[str, Any]], float], obj),
            name=getattr(obj, "__name__", "loss"),
        )
    raise TypeError(f"Cannot interpret {type(obj).__name__} as a Loss")
