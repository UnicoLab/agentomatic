"""Prompt optimizer — the main orchestration engine.

Like ``model.fit()`` but for prompts. Iteratively improves prompts
based on evaluation metrics, tracks all experiments, and provides
rich progress display.

Features:
- Prompt versioning (branching: v1 → v1_opt_1 → v1_opt_2)
- Rich live progress with per-iteration metrics table
- Experiment tracking (local JSON + extensible to MLflow/W&B)
- A/B comparison between prompt versions
- Early stopping with patience
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from agentomatic.optimize.dataset import Dataset
from agentomatic.optimize.metrics import BaseMetric, EvalResult, resolve_metrics
from agentomatic.optimize.runner import AgentRunner
from agentomatic.optimize.strategies import (
    IterationResult,
    OptimizationStrategy,
    resolve_strategy,
)

# =====================================================================
# Experiment Tracking
# =====================================================================


@dataclass
class ExperimentLog:
    """Tracks all optimization experiments to a local JSON file.

    Each experiment records: prompt versions, metrics, scores, and timing.
    Extensible — override ``log_iteration`` and ``save`` for MLflow/W&B.
    """

    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str = ""
    started_at: str = ""
    iterations: list[dict[str, Any]] = field(default_factory=list)
    best_iteration: int = 0
    best_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def log_iteration(self, result: IterationResult) -> None:
        """Record an iteration."""
        self.iterations.append(
            {
                "iteration": result.iteration,
                "prompt_version": f"opt_{result.iteration}",
                "avg_score": result.avg_score,
                "per_metric": result.per_metric_scores,
                "num_failures": len(result.failures),
                "improvements": result.improvements,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        if result.avg_score > self.best_score:
            self.best_score = result.avg_score
            self.best_iteration = result.iteration

    def save(self, path: str | Path) -> None:
        """Save experiment log to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing experiments
        existing: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, ValueError):
                existing = []

        existing.append(
            {
                "experiment_id": self.experiment_id,
                "agent": self.agent,
                "started_at": self.started_at,
                "best_iteration": self.best_iteration,
                "best_score": self.best_score,
                "total_iterations": len(self.iterations),
                "iterations": self.iterations,
                "metadata": self.metadata,
            }
        )

        path.write_text(json.dumps(existing, indent=2, default=str))
        logger.info(f"📊 Experiment log saved to {path}")


# =====================================================================
# Optimization Result
# =====================================================================


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run.

    Contains the best prompt, all iteration history, and methods
    to apply the result or generate reports.
    """

    best_prompt: str
    best_score: float
    best_iteration: int
    baseline_prompt: str
    baseline_score: float
    history: list[IterationResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    experiment_id: str = ""
    agent: str = ""

    @property
    def improvement(self) -> float:
        """Score improvement over baseline."""
        if self.baseline_score == 0:
            return float("inf") if self.best_score > 0 else 0.0
        return ((self.best_score - self.baseline_score) / self.baseline_score) * 100

    def apply(self, agent_dir: str | None = None, version: str | None = None) -> str:
        """Write the optimized prompt to the agent's prompts.json.

        Creates a new version (e.g., ``v1_optimized``) rather than
        overwriting existing versions.

        Args:
            agent_dir: Path to agent directory. Auto-detected if None.
            version: Custom version name. Auto-generated if None.

        Returns:
            The version key written.
        """
        if agent_dir is None:
            agent_dir = f"agents/{self.agent}"

        prompts_path = Path(agent_dir) / "prompts.json"

        # Load existing prompts
        if prompts_path.exists():
            data = json.loads(prompts_path.read_text())
        else:
            data = {}

        # Generate version name
        if version is None:
            existing_opt = [k for k in data if "opt" in k or "optimized" in k]
            idx = len(existing_opt) + 1
            version = f"v{idx}_optimized"

        # Write new version
        data[version] = {
            "system": self.best_prompt,
            "user_template": "{query}",
            "_metadata": {
                "optimization": {
                    "experiment_id": self.experiment_id,
                    "score": self.best_score,
                    "baseline_score": self.baseline_score,
                    "improvement_pct": round(self.improvement, 2),
                    "iterations": self.best_iteration,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            },
        }

        prompts_path.write_text(json.dumps(data, indent=2))
        logger.info(
            f"✅ Optimized prompt saved as '{version}' in {prompts_path} "
            f"(score: {self.best_score:.3f}, +{self.improvement:.1f}%)"
        )
        return version

    def compare(self) -> str:
        """Generate a comparison table between baseline and best prompt."""
        lines = [
            "┌─────────────────────────────────────────────────────────┐",
            "│              Prompt Optimization Comparison             │",
            "├──────────────────┬──────────────┬──────────────┬────────┤",
            "│ Version          │ Avg Score    │ Iteration    │ Status │",
            "├──────────────────┼──────────────┼──────────────┼────────┤",
            f"│ Baseline         │ {self.baseline_score:>10.4f}   │ {'0':>10}   │   📌   │",
            f"│ Best (optimized) │ {self.best_score:>10.4f}   │ {self.best_iteration:>10}   │   🏆   │",
            "├──────────────────┴──────────────┴──────────────┴────────┤",
            f"│ Improvement: {self.improvement:>+.1f}%                                     │",
            f"│ Total iterations: {len(self.history)}                                        │",
            f"│ Duration: {self.duration_seconds:.1f}s                                          │",
            "└─────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def report(self) -> str:
        """Generate a detailed Rich-formatted report."""
        try:
            return self._rich_report()
        except ImportError:
            return self._plain_report()

    def _rich_report(self) -> str:
        """Rich formatted report."""
        from rich.console import Console
        from rich.table import Table

        console = Console(record=True)

        # Summary
        console.print("\n[bold magenta]⚡ Optimization Report[/bold magenta]")
        console.print(f"   Agent: [cyan]{self.agent}[/cyan]")
        console.print(f"   Experiment: [dim]{self.experiment_id}[/dim]")
        console.print(f"   Duration: [yellow]{self.duration_seconds:.1f}s[/yellow]")
        console.print(f"   Improvement: [bold green]+{self.improvement:.1f}%[/bold green]\n")

        # Iteration table
        table = Table(title="📈 Iteration History", show_lines=True)
        table.add_column("#", justify="center", style="dim")
        table.add_column("Avg Score", justify="center")
        table.add_column("Metrics", justify="left")
        table.add_column("Status", justify="center")

        for it in self.history:
            score_color = "green" if it.avg_score >= self.baseline_score else "red"
            metrics_str = " | ".join(f"{k}: {v:.3f}" for k, v in it.per_metric_scores.items())
            is_best = "🏆" if it.iteration == self.best_iteration else ""
            table.add_row(
                str(it.iteration),
                f"[{score_color}]{it.avg_score:.4f}[/{score_color}]",
                metrics_str,
                is_best,
            )

        console.print(table)

        # Prompt comparison
        console.print("\n[bold]Baseline prompt:[/bold]")
        console.print(
            f"[dim]{self.baseline_prompt[:200]}...[/dim]"
            if len(self.baseline_prompt) > 200
            else f"[dim]{self.baseline_prompt}[/dim]"
        )
        console.print("\n[bold]Best prompt:[/bold]")
        console.print(
            f"[green]{self.best_prompt[:300]}...[/green]"
            if len(self.best_prompt) > 300
            else f"[green]{self.best_prompt}[/green]"
        )

        return console.export_text()

    def _plain_report(self) -> str:
        """Plain text report."""
        lines = [
            "\n⚡ Optimization Report",
            f"   Agent: {self.agent}",
            f"   Improvement: +{self.improvement:.1f}%",
            f"   Best score: {self.best_score:.4f} (iteration {self.best_iteration})",
            f"   Baseline: {self.baseline_score:.4f}",
            f"   Duration: {self.duration_seconds:.1f}s",
            "\nIteration History:",
        ]
        for it in self.history:
            marker = " 🏆" if it.iteration == self.best_iteration else ""
            lines.append(f"   #{it.iteration}: {it.avg_score:.4f}{marker}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "agent": self.agent,
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
            "improvement_pct": round(self.improvement, 2),
            "best_iteration": self.best_iteration,
            "total_iterations": len(self.history),
            "duration_seconds": self.duration_seconds,
            "best_prompt": self.best_prompt,
            "history": [
                {
                    "iteration": it.iteration,
                    "avg_score": it.avg_score,
                    "per_metric": it.per_metric_scores,
                }
                for it in self.history
            ],
        }


# =====================================================================
# Prompt Optimizer — Main Engine
# =====================================================================


class PromptOptimizer:
    """Iterative prompt optimization engine.

    Like ``model.fit()`` but for prompts. Evaluates prompts against a
    dataset using configurable metrics, then applies optimization
    strategies to iteratively improve them.

    Example::

        optimizer = PromptOptimizer(
            agent="my_agent",
            metrics=["answer_relevancy", "exact_match"],
            strategy="iterative_rewrite",
        )
        result = await optimizer.optimize(
            dataset=Dataset.from_jsonl("qa.jsonl"),
            max_iterations=10,
            target_score=0.9,
        )
        logger.info(result.report())
        result.apply()

    Args:
        agent: Agent name (must be registered in the platform).
        metrics: List of metric names or BaseMetric instances.
        llm: Default LLM for both evaluation and rewriting.
        rewrite_llm: LLM specifically for prompt rewriting/generation.
            Overrides ``llm`` for the rewrite strategy only.
        eval_llm: LLM specifically for metric evaluation.
            Overrides ``llm`` for evaluation metrics only.
        strategy: Optimization strategy name or instance.
        api_base: Platform API base URL.
        api_prefix: API prefix path.
        experiment_dir: Directory for experiment logs.
        concurrency: Max concurrent agent invocations.
        auto_report: Generate an HTML report after optimization.

    Example with separate models::

        optimizer = PromptOptimizer(
            agent="my_agent",
            rewrite_llm="ollama/llama3:70b",   # powerful model for rewriting
            eval_llm="ollama/mistral:7b",      # fast model for evaluation
            metrics=["answer_relevancy"],
        )
    """

    def __init__(
        self,
        agent: str,
        metrics: list[str | BaseMetric] | None = None,
        llm: str = "ollama/mistral:7b",
        rewrite_llm: str | None = None,
        eval_llm: str | None = None,
        strategy: str | OptimizationStrategy = "iterative_rewrite",
        api_base: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        experiment_dir: str = ".optimize",
        concurrency: int = 5,
        auto_report: bool = True,
    ):
        self.agent = agent
        self.llm = llm
        self.rewrite_llm = rewrite_llm or llm
        self.eval_llm = eval_llm or llm
        self.api_base = api_base
        self.api_prefix = api_prefix
        self.experiment_dir = Path(experiment_dir)
        self.concurrency = concurrency
        self.auto_report = auto_report

        # Resolve metrics (use eval_llm for evaluation)
        self._raw_metrics = metrics or ["exact_match"]
        self._metrics: list[BaseMetric] = resolve_metrics(self._raw_metrics, model=self.eval_llm)

        # Resolve strategy (use rewrite_llm for prompt generation)
        self._strategy: OptimizationStrategy = resolve_strategy(strategy, model=self.rewrite_llm)

        # Runner
        self._runner = AgentRunner(
            agent=agent,
            api_base=api_base,
            api_prefix=api_prefix,
        )

    async def optimize(
        self,
        dataset: Dataset,
        initial_prompt: str | None = None,
        max_iterations: int = 10,
        target_score: float = 0.9,
        patience: int = 3,
        min_improvement: float = 0.001,
        save_experiments: bool = True,
    ) -> OptimizationResult:
        """Run iterative prompt optimization.

        Args:
            dataset: Dataset of Q/A pairs to optimize against.
            initial_prompt: Starting prompt. If None, reads from agent's prompts.json.
            max_iterations: Maximum optimization iterations.
            target_score: Stop when average score reaches this threshold.
            patience: Stop after N consecutive iterations without improvement.
            min_improvement: Minimum score improvement to count as progress.
            save_experiments: Whether to save experiment logs to disk.

        Returns:
            OptimizationResult with best prompt and full history.
        """
        experiment = ExperimentLog(agent=self.agent)
        experiment.started_at = datetime.now(UTC).isoformat()
        experiment.metadata = {
            "strategy": self._strategy.name,
            "metrics": [m.name for m in self._metrics],
            "llm": self.llm,
            "rewrite_llm": self.rewrite_llm,
            "eval_llm": self.eval_llm,
            "max_iterations": max_iterations,
            "target_score": target_score,
            "dataset_size": len(dataset),
        }

        # Load initial prompt
        current_prompt = initial_prompt or self._load_prompt()

        # Initialize progress display
        progress = _ProgressDisplay(max_iterations)

        t0 = time.perf_counter()
        history: list[IterationResult] = []
        best_prompt = current_prompt
        best_score = 0.0
        best_iteration = 0
        no_improvement_count = 0

        # === Baseline evaluation ===
        progress.start(self.agent, len(dataset), len(self._metrics))
        baseline_result = await self._evaluate_iteration(dataset, current_prompt, 0, progress)
        history.append(baseline_result)
        experiment.log_iteration(baseline_result)

        baseline_score = baseline_result.avg_score
        best_score = baseline_score
        best_prompt = current_prompt

        progress.update_iteration(baseline_result, is_best=True, is_baseline=True)

        # === Optimization loop ===
        for i in range(1, max_iterations + 1):
            progress.set_iteration(i)

            # Check early stopping
            if best_score >= target_score:
                progress.info(f"🎯 Target score {target_score} reached!")
                break

            if no_improvement_count >= patience:
                progress.info(f"⏹️  Patience exhausted ({patience} iterations)")
                break

            # Generate improved prompt
            progress.phase("Generating improved prompt...")
            dataset_sample = [p.to_dict() for p in dataset.points[:10]]
            eval_for_strategy = self._results_to_strategy_input(history[-1], dataset)

            new_prompt = await self._strategy.step(
                current_prompt=current_prompt,
                eval_results=eval_for_strategy,
                dataset_sample=dataset_sample,
                iteration=i,
            )

            # Evaluate new prompt
            iter_result = await self._evaluate_iteration(dataset, new_prompt, i, progress)

            # Track improvement
            is_better = iter_result.avg_score > best_score + min_improvement
            if is_better:
                iter_result.improvements = (
                    f"Score improved: {best_score:.4f} → {iter_result.avg_score:.4f} "
                    f"(+{((iter_result.avg_score - best_score) / max(best_score, 0.001)) * 100:.1f}%)"
                )
                best_score = iter_result.avg_score
                best_prompt = new_prompt
                best_iteration = i
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            history.append(iter_result)
            experiment.log_iteration(iter_result)
            current_prompt = new_prompt  # Always move forward

            progress.update_iteration(iter_result, is_best=is_better)

        duration = time.perf_counter() - t0
        progress.finish(best_score, baseline_score, duration)

        # Save experiment log
        if save_experiments:
            log_path = self.experiment_dir / self.agent / "experiments.json"
            experiment.save(log_path)

        result = OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            best_iteration=best_iteration,
            baseline_prompt=initial_prompt or current_prompt,
            baseline_score=baseline_score,
            history=history,
            duration_seconds=duration,
            experiment_id=experiment.experiment_id,
            agent=self.agent,
        )

        # Auto-generate HTML report
        if self.auto_report:
            try:
                from agentomatic.optimize.report import generate_html_report

                report_path = generate_html_report(result)
                logger.info(f"📊 HTML report: {report_path}")
            except Exception as exc:
                logger.warning(f"Report generation failed: {exc}")

        return result

    async def evaluate(
        self,
        dataset: Dataset,
        prompt: str | None = None,
    ) -> IterationResult:
        """Evaluate a prompt without optimization (single pass).

        Useful for benchmarking a specific prompt version.
        """
        prompt = prompt or self._load_prompt()
        progress = _ProgressDisplay(1)
        progress.start(self.agent, len(dataset), len(self._metrics))
        result = await self._evaluate_iteration(dataset, prompt, 0, progress)
        progress.finish(result.avg_score, 0.0, 0.0)
        return result

    async def compare_prompts(
        self,
        dataset: Dataset,
        prompts: dict[str, str],
    ) -> dict[str, IterationResult]:
        """Evaluate multiple prompts and compare scores.

        Args:
            dataset: Evaluation dataset.
            prompts: Dict of version_name → prompt_text.

        Returns:
            Dict of version_name → IterationResult.
        """
        results: dict[str, IterationResult] = {}
        progress = _ProgressDisplay(len(prompts))
        progress.start(self.agent, len(dataset), len(self._metrics))

        for idx, (version, prompt) in enumerate(prompts.items()):
            progress.phase(f"Evaluating '{version}'...")
            result = await self._evaluate_iteration(dataset, prompt, idx, progress)
            result.prompt = prompt
            results[version] = result
            progress.update_iteration(result, is_best=False, label=version)

        # Print comparison
        self._print_comparison(results)
        return results

    # -----------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------

    async def _evaluate_iteration(
        self,
        dataset: Dataset,
        prompt: str,
        iteration: int,
        progress: _ProgressDisplay,
    ) -> IterationResult:
        """Run dataset through agent and evaluate with all metrics."""
        # Run agent
        progress.phase(f"Running {len(dataset)} queries...")
        points = [p.to_dict() for p in dataset.points]
        run_results = await self._runner.run_dataset(
            points, prompt_override=prompt, concurrency=self.concurrency
        )

        # Evaluate each response
        progress.phase(f"Evaluating with {len(self._metrics)} metrics...")
        all_scores: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        failures: list[dict[str, Any]] = []

        for run_result in run_results:
            if run_result.error:
                for m in self._metrics:
                    all_scores[m.name].append(0.0)
                failures.append(
                    {
                        "query": run_result.query,
                        "error": run_result.error,
                        "avg_score": 0.0,
                    }
                )
                continue

            point_scores: list[float] = []
            point_details: list[dict] = []

            for metric in self._metrics:
                try:
                    eval_result: EvalResult = await metric.evaluate(
                        query=run_result.query,
                        response=run_result.response,
                        expected=run_result.expected,
                        context=(run_result.retrieval_context or run_result.context or None),
                    )
                    all_scores[metric.name].append(eval_result.score)
                    point_scores.append(eval_result.score)
                    point_details.append(
                        {
                            "metric": eval_result.metric_name,
                            "score": eval_result.score,
                            "reason": eval_result.reason,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"Metric {metric.name} failed: {exc}")
                    all_scores[metric.name].append(0.0)
                    point_scores.append(0.0)

            avg = sum(point_scores) / len(point_scores) if point_scores else 0.0
            if avg < 0.5:
                failures.append(
                    {
                        "query": run_result.query,
                        "response": run_result.response,
                        "expected": run_result.expected,
                        "avg_score": avg,
                        "details": point_details,
                    }
                )

            progress.tick()

        # Compute per-metric averages
        per_metric = {
            name: (sum(scores) / len(scores)) if scores else 0.0
            for name, scores in all_scores.items()
        }
        avg_score = sum(per_metric.values()) / len(per_metric) if per_metric else 0.0

        return IterationResult(
            iteration=iteration,
            prompt=prompt,
            avg_score=avg_score,
            per_metric_scores=per_metric,
            failures=failures[:10],  # Keep top failures
        )

    def _results_to_strategy_input(
        self,
        iteration_result: IterationResult,
        dataset: Dataset,
    ) -> list[dict[str, Any]]:
        """Convert iteration result to strategy-compatible input."""
        return iteration_result.failures + [
            {
                "query": p.query,
                "expected": p.expected_answer,
                "response": "(not re-run)",
                "avg_score": 1.0,
            }
            for p in dataset.points[:5]
        ]

    def _load_prompt(self) -> str:
        """Load current prompt from agent's prompts.json."""
        for search_dir in [Path(f"agents/{self.agent}"), Path(self.agent)]:
            prompts_file = search_dir / "prompts.json"
            if prompts_file.exists():
                data = json.loads(prompts_file.read_text())
                # Get the first available version
                for version in ["v1", "default"]:
                    if version in data and "system" in data[version]:
                        return data[version]["system"]
                # Fallback: first entry
                for key, val in data.items():
                    if isinstance(val, dict) and "system" in val:
                        return val["system"]

        return "You are a helpful AI assistant."

    def _print_comparison(self, results: dict[str, IterationResult]) -> None:
        """Print prompt version comparison."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="📊 Prompt Version Comparison", show_lines=True)
            table.add_column("Version", style="bold cyan")
            table.add_column("Avg Score", justify="center")
            for m in self._metrics:
                table.add_column(m.name, justify="center")
            table.add_column("Rank", justify="center")

            sorted_results = sorted(results.items(), key=lambda x: x[1].avg_score, reverse=True)
            for rank, (version, result) in enumerate(sorted_results, 1):
                row = [
                    version,
                    f"{result.avg_score:.4f}",
                ]
                for m in self._metrics:
                    score = result.per_metric_scores.get(m.name, 0.0)
                    row.append(f"{score:.4f}")
                row.append("🏆" if rank == 1 else str(rank))
                table.add_row(*row)

            console.print(table)
        except ImportError:
            for version, result in results.items():
                logger.info(f"  {version}: {result.avg_score:.4f}")


# =====================================================================
# Progress Display
# =====================================================================


class _ProgressDisplay:
    """Rich progress display for optimization runs.

    Shows:
    - Overall progress bar
    - Per-iteration metrics table (live updating)
    - Phase indicators

    Falls back to plain tqdm/print if Rich is not available.
    """

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self._has_rich = False
        self._console: Any = None
        self._live: Any = None
        self._table: Any = None
        self._total_points = 0
        self._processed = 0

        try:
            from rich.console import Console

            self._has_rich = True
            self._console = Console()
        except ImportError:
            pass

    def start(self, agent: str, n_points: int, n_metrics: int) -> None:
        """Start progress display."""
        self._total_points = n_points

        if self._has_rich:
            from rich.panel import Panel

            self._console.print(
                Panel.fit(
                    f"[bold magenta]⚡ Prompt Optimizer[/bold magenta]\n"
                    f"[dim]Agent: {agent} | {n_points} points | {n_metrics} metrics | "
                    f"max {self.max_iterations} iterations[/dim]",
                    border_style="magenta",
                )
            )

            from rich.table import Table

            self._table = Table(title="📈 Optimization Progress", show_lines=True)
            self._table.add_column("#", justify="center", style="dim", width=4)
            self._table.add_column("Score", justify="center", width=10)
            self._table.add_column("Δ", justify="center", width=8)
            self._table.add_column("Metrics", width=40)
            self._table.add_column("Status", justify="center", width=8)
        else:
            logger.info(f"⚡ Optimizing: {agent} | {n_points} points | {n_metrics} metrics")

        self._prev_score = 0.0

    def set_iteration(self, i: int) -> None:
        """Mark the start of an iteration."""
        self._processed = 0
        if self._has_rich:
            self._console.print(
                f"\n[bold]Iteration {i}/{self.max_iterations}[/bold]",
                highlight=False,
            )
        else:
            logger.info(f"\n--- Iteration {i}/{self.max_iterations} ---")

    def phase(self, msg: str) -> None:
        """Show a phase indicator."""
        if self._has_rich:
            self._console.print(f"  [dim]{msg}[/dim]", highlight=False)
        else:
            logger.info(f"  {msg}")

    def tick(self) -> None:
        """Increment progress counter."""
        self._processed += 1

    def update_iteration(
        self,
        result: IterationResult,
        is_best: bool = False,
        is_baseline: bool = False,
        label: str = "",
    ) -> None:
        """Update display with iteration results."""
        delta = result.avg_score - self._prev_score
        self._prev_score = result.avg_score

        metrics_str = " | ".join(f"{k}: {v:.3f}" for k, v in result.per_metric_scores.items())

        if self._has_rich and self._table:
            status = ""
            if is_baseline:
                status = "📌"
            elif is_best:
                status = "🏆"
            else:
                status = "📉" if delta < 0 else "↗️"

            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            delta_color = "green" if delta >= 0 else "red"

            iter_label = label or ("baseline" if is_baseline else str(result.iteration))
            self._table.add_row(
                iter_label,
                f"[bold]{result.avg_score:.4f}[/bold]",
                f"[{delta_color}]{delta_str}[/{delta_color}]",
                metrics_str,
                status,
            )
            self._console.print(self._table)
        else:
            marker = " 🏆 BEST" if is_best else ""
            delta_s = f" (Δ {delta:+.4f})" if not is_baseline else ""
            logger.info(f"  Score: {result.avg_score:.4f}{delta_s}{marker}")
            if metrics_str:
                logger.info(f"  {metrics_str}")

    def info(self, msg: str) -> None:
        """Print info message."""
        if self._has_rich:
            self._console.print(f"[bold yellow]{msg}[/bold yellow]")
        else:
            logger.info(msg)

    def finish(self, best_score: float, baseline: float, duration: float) -> None:
        """Show final summary."""
        if baseline > 0:
            improvement = ((best_score - baseline) / baseline) * 100
        else:
            improvement = 0.0

        if self._has_rich:
            from rich.panel import Panel

            self._console.print(
                Panel.fit(
                    f"[bold green]✅ Optimization Complete[/bold green]\n\n"
                    f"  Baseline:    [dim]{baseline:.4f}[/dim]\n"
                    f"  Best:        [bold green]{best_score:.4f}[/bold green]\n"
                    f"  Improvement: [bold cyan]+{improvement:.1f}%[/bold cyan]\n"
                    f"  Duration:    [yellow]{duration:.1f}s[/yellow]",
                    border_style="green",
                )
            )
        else:
            logger.success(
                f"\n✅ Done! {baseline:.4f} → {best_score:.4f} (+{improvement:.1f}%) in {duration:.1f}s"
            )
