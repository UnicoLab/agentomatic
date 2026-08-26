#!/usr/bin/env python3
"""Keras-style optimization showcase for Agentomatic (local omlx stack).

Proves the optimization engine end-to-end: a deliberately-bad agent
(vague prompt + hot temperature) is progressively improved by
``agent.compile() → agent.fit()`` with a Keras-like ``loss`` that gets
smaller every epoch.

Run (from the repo root, with the local omlx server running)::

    OMLX_API_KEY=… OMLX_BASE_URL=http://127.0.0.1:8000/v1 \\
      uv run python examples/keras_optimize_showcase/train.py --mode rewrite

    # every registered fitter optimizer mode:
    #   rewrite | gepa_like | mipro_like | few_shot_bootstrap | param_search

    OMLX_API_KEY=… OMLX_BASE_URL=http://127.0.0.1:8000/v1 \\
      uv run python examples/keras_optimize_showcase/train.py --mode all

What you should see:

    Epoch 1/3 - banana: 0.0000 - temp_ok: 0.0000 - quality: 0.0000 - loss: 1.0000
    Epoch 2/3 - banana: 1.0000 - temp_ok: 0.0000 - quality: 0.5000 - loss: 0.5000
    ...
    Best val_loss: 0.0000 @ epoch 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Public API only (the exact Keras-style imports from the docs) ──────
from agentomatic import (  # noqa: E402
    CallableMetric,
    EarlyStopping,
    EpochDiffCallback,
    ExactKeyMatchMetric,
    History,
    MetricLoss,
    PromptFitterBridge,
    WeightedMetric,
)
from agentomatic.agents import Callback  # noqa: E402
from agentomatic.optimize import (  # noqa: E402
    CustomMetric,
    PromptSearchSpace,
)
from examples.keras_optimize_showcase.agent import (  # noqa: E402
    BAD_PROMPT,
    MARKER,
    MARKER2,
    MARKER3,
    MARKER4,
    TEMP_RUNGS,
    MarkerAgent,
)
from examples.keras_optimize_showcase.dataset import build_dataset  # noqa: E402

# Local omlx server (OpenAI-compatible). Set OMLX_BASE_URL / OMLX_API_KEY
# or rely on the defaults used by the agentomatic optimize engine. Pick a
# faster local model (e.g. the DeepSeek stack) via AGENTOMATIC_LIVE_MODEL.
MODEL = os.getenv("AGENTOMATIC_LIVE_MODEL", "omlx/Qwen3.5-9B-MLX-4bit")
BASE_URL = os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("OMLX_API_KEY", "local")

MODES = ["rewrite", "gepa_like", "mipro_like", "few_shot_bootstrap", "param_search"]


# ---------------------------------------------------------------------------
# Objective / metrics
# ---------------------------------------------------------------------------


def quality_score(query: str, response: str, expected: str | None = None, context=None) -> float:
    """Fitter objective: 1/7 per satisfied curriculum signal + tiny base.

    The agent only emits tokens for the signals required by the current
    difficulty level (see ``MarkerAgent.respond``), so the fitter's
    candidate ranking and the epoch ``loss`` move in lock-step. The tiny
    base score for any non-empty response keeps optimizers that bootstrap
    from scored results (e.g. few-shot) working even when the baseline
    satisfies nothing yet.
    """
    del query, expected, context
    resp = (response or "").lower()
    score = 1e-3 if resp else 0.0
    for token in (MARKER, MARKER2, MARKER3, MARKER4, "r1", "r2", "r3"):
        if token in resp:
            score += 1 / 7
    return score


class _MarkerMetric(CustomMetric):
    """CustomMetric with a rich reason so the rewriter sees exactly what failed."""

    async def evaluate(self, query, response, expected=None, context=None):
        import asyncio

        from agentomatic.optimize.metrics import EvalResult

        if asyncio.iscoroutinefunction(self.fn):
            score = await self.fn(query, response, expected, context)
        else:
            score = await asyncio.to_thread(self.fn, query, response, expected, context)
        reasons = []
        resp = (response or "").lower()
        if MARKER not in resp:
            reasons.append(
                f"response is missing the required marker token '{MARKER}' "
                "(expected output contains 'OPT, banana')"
            )
        for token in (MARKER2, MARKER3, MARKER4):
            if token not in resp:
                reasons.append(
                    f"response is missing the required marker token '{token}' "
                    "(judge guidance lists it as part of the ideal answer)"
                )
        for rung, threshold in zip(("r1", "r2", "r3"), TEMP_RUNGS, strict=True):
            if rung not in resp:
                reasons.append(
                    f"response is missing the '{rung}' marker (temperature must be <= {threshold})"
                )
        return EvalResult(
            metric_name=self.name,
            score=float(score),
            reason="; ".join(reasons) or "response contains all required markers",
        )


def build_metrics() -> tuple[list[Any], MetricLoss, _MarkerMetric]:
    """Return (metrics, loss, fit_metric) — the Keras-style metric stack.

    Seven equal quality signals (1/7 each): the four prompt markers
    (``banana`` / ``strawberry`` / ``blueberry`` / ``kiwi``) plus three
    temperature rungs (``<=0.5`` / ``<=0.3`` / ``<=0.15``). Marker metrics
    read the response text only (the curriculum-gated tokens), so each
    metric flips from 0 → 1 exactly when its signal is discovered.
    """

    def _resp_contains(token: str) -> Any:
        return CallableMetric(
            token,
            (lambda t: lambda ex, pred: 1.0 if t in str(pred.get("response", "")) else 0.0)(token),
        )

    banana_m = _resp_contains(MARKER)
    strawberry_m = _resp_contains(MARKER2)
    blueberry_m = _resp_contains(MARKER3)
    kiwi_m = _resp_contains(MARKER4)
    temp_m = CallableMetric("temp_ok", lambda ex, pred: 1.0 if pred.get("temp_ok") else 0.0)
    rung_metrics = [_resp_contains(f"r{k}") for k in (1, 2, 3)]
    key_m = ExactKeyMatchMetric(["response"], name="exact_key_match")
    quality_m = WeightedMetric(
        [
            ("banana", banana_m, 1 / 7),
            ("strawberry", strawberry_m, 1 / 7),
            ("blueberry", blueberry_m, 1 / 7),
            ("kiwi", kiwi_m, 1 / 7),
            ("temp_r1", rung_metrics[0], 1 / 7),
            ("temp_r2", rung_metrics[1], 1 / 7),
            ("temp_r3", rung_metrics[2], 1 / 7),
        ],
        name="quality",
    )
    metrics: list[Any] = [
        banana_m,
        strawberry_m,
        blueberry_m,
        kiwi_m,
        temp_m,
        *rung_metrics,
        quality_m,
        key_m,
    ]
    loss = MetricLoss(quality_m)
    fit_metric = _MarkerMetric(fn=quality_score, name="marker_quality")
    return metrics, loss, fit_metric


# Temperature grid for parameter search: denser than the sampler's per-epoch
# budget so ``random`` sampling descends the rungs progressively across
# epochs instead of jumping to the coldest value in a single round.
PARAM_GRID: dict[str, list[float]] = {
    "temperature": [
        0.7,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.05,
        0.02,
        0.0,
    ]
}


def search_space_for(mode: str) -> PromptSearchSpace:
    """Search space per optimizer mode (only the knobs that mode can tune)."""
    if mode == "rewrite":
        return PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_user_template=True,
            optimize_few_shot=False,
            optimize_model_params=False,
        )
    if mode == "gepa_like":
        return PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_few_shot=False,
            optimize_model_params=False,
        )
    if mode == "mipro_like":
        return PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_few_shot=True,
            optimize_model_params=True,
            model_param_space=PARAM_GRID,
        )
    if mode == "few_shot_bootstrap":
        return PromptSearchSpace(
            optimize_system_prompt=False,
            optimize_few_shot=True,
            optimize_model_params=False,
        )
    if mode == "param_search":
        return PromptSearchSpace(
            optimize_system_prompt=False,
            optimize_few_shot=False,
            optimize_model_params=True,
            # Denser grid + random sampling → rung-by-rung descent per epoch.
            model_param_space=PARAM_GRID,
            search_method="random",
        )
    raise ValueError(f"Unknown mode {mode!r}")


# Curriculum caps per mode: how many of the seven quality signals the mode
# can actually satisfy (prompt markers vs. temperature rungs), and the
# starting difficulty. Difficulty 1-4 = prompt markers, 5-7 = temp rungs.
MAX_DIFFICULTY: dict[str, int] = {
    "rewrite": 4,  # prompt markers only
    "gepa_like": 4,
    "mipro_like": 7,  # markers + temperature rungs
    "few_shot_bootstrap": 1,  # few-shot blocks carry the primary marker
    "param_search": 7,  # temperature rungs only (starts at 5)
}
CURRICULUM_START: dict[str, int] = {
    "rewrite": 1,
    "gepa_like": 1,
    "mipro_like": 1,
    "few_shot_bootstrap": 1,
    "param_search": 5,  # prompt markers are out of scope — start at the rungs
}


class CurriculumCallback(Callback):
    """Raise the agent's difficulty by one after every epoch.

    Each new epoch demands one more quality signal from the agent, so the
    optimizer must keep discovering new markers / colder temperatures —
    producing a Keras-style loss that descends every epoch and a prompt
    that keeps evolving. The fitter's compounding (each epoch starts from
    the previous best config) is what makes this possible.
    """

    def __init__(self, max_difficulty: int = 7, start: int = 1) -> None:
        super().__init__()
        self.max_difficulty = max_difficulty
        self.start = start

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        if self.agent is not None:
            self.agent.difficulty = self.start

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        if self.agent is None:
            return
        current = max(0, int(getattr(self.agent, "difficulty", self.start)))
        self.agent.difficulty = min(current + 1, self.max_difficulty)


def print_history(history: History) -> None:
    """Print a Keras-style loss table."""
    print("\n┌──────────────────────────────────────────────────────────────┐")
    print("│  Keras-style training history                                │")
    print("├───────────────┬──────────────────────────────────────────────┤")
    keys = [k for k in history.keys() if not k.startswith("val_")]
    header = "  Epoch  │ " + "   ".join(f"{k:>10}" for k in keys)
    print(header)
    print("├───────────────┼──────────────────────────────────────────────┤")
    for i, epoch in enumerate(history.epoch):
        values = "   ".join(f"{history[k][i]:>10.4f}" for k in keys)
        print(f"     {epoch + 1:>3}  │ {values}")
    print("└───────────────┴──────────────────────────────────────────────┘")

    loss_key = "val_loss" if "val_loss" in history else "loss"
    best = history.best(loss_key, mode="min")
    if best:
        print(f"\n📉 Best {loss_key}: {best[1]:.4f} @ epoch {best[0] + 1}")
    print(history.summary())


def run_mode(
    mode: str,
    *,
    epochs: int = 10,
    max_trials: int = 6,
    seed: int = 42,
    report_dir: str | None = None,
    model: str | None = None,
    augment: bool = False,
    n_examples: int = 30,
) -> History:
    """Run the full Keras-style workflow for one optimizer mode."""
    model = model or MODEL
    print(f"\n{'=' * 78}\n🚀 Mode: {mode}  (epochs={epochs}, max_trials={max_trials})\n{'=' * 78}")

    # ── 1. Fake training dataset (+ optional LLM augmentation) ─────────
    dataset = build_dataset(seed=seed)
    if augment:
        from agentomatic.optimize import prepare_dataset

        dataset, written = prepare_dataset(
            dataset,
            augment=True,
            n_examples=n_examples,
            persist=True,
            persist_path=ROOT / "optimization_results" / "showcase" / "dataset.augmented.jsonl",
            seed_path=ROOT / "examples" / "keras_optimize_showcase" / "dataset.jsonl",
            model=model,
            llm_base_url=BASE_URL,
            llm_api_key=API_KEY,
        )
        print(
            f"🧬 Augmented dataset: {len(dataset.examples)} examples "
            f"(train={len(dataset.train)}, val={len(dataset.validation)}, "
            f"test={len(dataset.test)}) → {written}"
        )
    else:
        print(
            f"📊 Dataset: {len(dataset.examples)} examples "
            f"(train={len(dataset.train)}, val={len(dataset.validation)}, "
            f"test={len(dataset.test)})"
        )

    agent = MarkerAgent()
    print(f"Baseline prompt : {BAD_PROMPT!r}")
    print(f"Baseline temp    : {agent.temperature} (rungs <=0.5 / <=0.3 / <=0.15)")

    # ── 2. Metrics / loss / fitter objective ───────────────────────────
    metrics, loss, fit_metric = build_metrics()

    # ── 3. Keras-style compile ─────────────────────────────────────────
    agent.compile(
        dataset=dataset,
        metrics=metrics,
        loss=loss,
        optimizer=PromptFitterBridge(
            agent_name=agent.agent_name,
            task_model=model,
            rewrite_model=model,
            metric=fit_metric,
            optimizer=mode,
            max_trials=max_trials,
            search_space=search_space_for(mode),
            concurrency=1,
            auto_report=False,
            experiment_dir=str(ROOT / "optimization_results" / ".fit"),
            llm_base_url=BASE_URL,
            llm_api_key=API_KEY,
            min_absolute_improvement=0.001,
            patience=1,
        ),
    )

    # ── 4. Keras-style fit (epochs + validation + early stopping) ─────
    t0 = time.perf_counter()
    curriculum = CurriculumCallback(
        max_difficulty=MAX_DIFFICULTY.get(mode, 4),
        start=CURRICULUM_START.get(mode, 1),
    )
    diff_report = EpochDiffCallback(epochs=epochs)
    history = agent.fit(
        dataset,
        epochs=epochs,
        verbose=1,
        validation_data=dataset.validation,
        callbacks=[
            curriculum,
            diff_report,
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        optimize_mode=mode,
        search_space=search_space_for(mode),
        max_trials=max_trials,
    )
    print(f"⏱️  Fit took {time.perf_counter() - t0:.1f}s")
    print(
        f"🧗 Curriculum: difficulty reached {curriculum.agent.difficulty if curriculum.agent else '?'}"
    )
    print_history(history)

    # ── 5. What did the optimizer change? ──────────────────────────────
    print("\n🔧 Compiled config after fit:")
    for key, value in agent.compiled_config.items():
        if key in ("system_prompt", "temperature", "few_shot_examples"):
            preview = value if isinstance(value, str) else json.dumps(value)
            print(f"   {key}: {str(preview)[:120]}")
    result = getattr(agent, "_last_fit_result", None)
    if result is not None:
        print(f"\n📊 Fitter status : {getattr(agent, '_last_optimize_status', '?')!r}")
        print(f"   baseline→best : {result.baseline_score:.3f} → {result.best_score:.3f}")
        print(f"   holdout score : {result.holdout_score}")
        if result.suggestions:
            print(f"   suggestions    : {result.suggestions[0][:120]}")

    # ── 6. Evaluate on held-out test split ─────────────────────────────
    report = agent.evaluate(dataset.test, metrics)
    print(
        f"\n✅ Test evaluation: {json.dumps({k: round(v, 3) for k, v in report.scores.items()})}"
    )

    # ── 7. Interactive HolySheet report (chart + prompt evolution) ─────
    fit_result = getattr(agent, "_last_fit_result", None)
    report_path: str | None = None
    if fit_result is not None:
        from agentomatic.optimize import generate_fit_report

        reports_dir = (
            Path(report_dir)
            if report_dir
            else ROOT / "optimization_results" / "showcase" / "reports"
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = generate_fit_report(
            fit_result,
            output_path=reports_dir / f"fit_{mode}.html",
            keras_history=history.history,
            eval_scores=report.scores,
            dataset_sizes={
                "train": len(dataset.train),
                "validation": len(dataset.validation),
                "test": len(dataset.test),
            },
            optimizer_name=mode,
            stack_name="omlx",
            model_name=model,
            run_config={
                "epochs": epochs,
                "max_trials": max_trials,
                "loss": loss.name,
                "metrics": [m.name for m in metrics],
            },
        )
        print(f"📄 Fit report: {report_path}")

        # Dump the Keras history + prompt evolution as JSON (docs tooling).
        evolution = []
        for entry in getattr(fit_result, "prompt_history", None) or []:
            if not isinstance(entry, dict):
                continue
            evolution.append(
                {
                    "round_idx": entry.get("round_idx"),
                    "score": round(float(entry.get("score", 0.0)), 4),
                    "accepted": bool(entry.get("accepted", False)),
                    "what_worked": entry.get("what_worked") or [],
                    "what_failed": entry.get("what_failed") or [],
                    "next_focus": entry.get("next_focus") or [],
                    "prompt": (entry.get("prompt_snapshot") or "")[:2000],
                }
            )
        summary = {
            "mode": mode,
            "model": model,
            "history": history.to_dict(),
            "compiled_config": {
                k: (str(v)[:2000] if not isinstance(v, (int, float, bool)) else v)
                for k, v in agent.compiled_config.items()
            },
            "per_epoch_changes": diff_report.per_epoch,
            "prompt_evolution": evolution,
            "test_scores": {k: round(v, 4) for k, v in report.scores.items()},
        }
        (reports_dir / f"summary_{mode}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── 8. Save / reload (Keras-style model persistence) ───────────────
    out_dir = ROOT / "optimization_results" / "showcase" / mode
    agent.save(out_dir)
    restored = MarkerAgent()
    restored.load(out_dir)
    assert restored.compiled_config == agent.compiled_config
    print(f"💾 Saved + reloaded compiled state: {out_dir}")
    run_mode_history_cache[mode] = history
    return history


run_mode_history_cache: dict[str, History] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keras-style optimization showcase (omlx)")
    parser.add_argument("--mode", choices=MODES + ["all"], default="all", help="Optimizer mode")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", default=None, help="Directory for the HTML report(s)")
    parser.add_argument(
        "--augment", action="store_true", help="LLM-augment the seed dataset first"
    )
    parser.add_argument("--n-examples", type=int, default=30, help="Augmented dataset target size")
    parser.add_argument(
        "--model",
        default=os.getenv("AGENTOMATIC_LIVE_MODEL", "omlx/Qwen3.5-9B-MLX-4bit"),
        help="Local OpenAI-compatible model spec (e.g. omlx/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx)",
    )
    args = parser.parse_args(argv)

    modes = MODES if args.mode == "all" else [args.mode]
    for mode in modes:
        run_mode(
            mode,
            epochs=args.epochs,
            max_trials=args.max_trials,
            seed=args.seed,
            report_dir=args.report_dir,
            model=args.model,
            augment=args.augment,
            n_examples=args.n_examples,
        )

    # Cross-mode summary: prove every method drives loss down.
    print("\n" + "=" * 78)
    print("📈 LOSS PER EPOCH PER METHOD (lower is better)")
    print("=" * 78)
    for mode in modes:
        hist = run_mode_history_cache.get(mode)
        if hist and "loss" in hist:
            curve = [f"{v:.3f}" for v in hist["loss"]]
            print(f"   {mode:<20} " + " → ".join(curve))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
