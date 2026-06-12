"""Local-first prompt optimization loop — like ``model.fit()`` for prompts.

Fully generic, framework-agnostic. Works with **any** LLM (LangChain,
OpenAI, custom callable) and **any** scoring function.

Features
--------
- **Iterative LLM rewrite** — analyses failures and rewrites prompts
- **Pluggable scoring** — sync or async, keyword or LLM-as-a-judge
- **Multiple strategies** — ``iterative``, ``adversarial``, ``structured``
- **Early stopping** — patience-based convergence detection
- **Prompt versioning** — tracks every prompt variant with scores
- **Rich HTML reports** — dark-themed evolution charts, per-step details
- **Experiment tracking** — JSON logs for reproducibility
- **Zero external deps** — only stdlib + loguru (optional)

Quickstart
----------
::

    from agentomatic.optimize.loop import PromptOptimizationLoop

    async def invoke_fn(datapoint: dict, prompt: str) -> dict:
        '''Run your agent with the given system prompt.'''
        response = await my_agent(datapoint["query"], system_prompt=prompt)
        return {"actual_response": response}

    def score_fn(expected: str, actual: str) -> float:
        '''Score 0..1 — higher is better.'''
        return len(set(expected.split()) & set(actual.split())) / max(len(expected.split()), 1)

    loop = PromptOptimizationLoop(
        agent_name="my_agent",
        invoke_fn=invoke_fn,
        score_fn=score_fn,
        dataset_path="eval_data.jsonl",
    )
    result = await loop.run(initial_prompt="You are a helpful assistant.", steps=5)
    result.save("reports/my_agent/")

With LLM-as-a-judge (async scorer)::

    async def llm_judge(expected: str, actual: str, dp: dict) -> float:
        # your LLM evaluation logic
        return score

    loop = PromptOptimizationLoop(
        agent_name="my_agent",
        invoke_fn=invoke_fn,
        score_fn=llm_judge,
        dataset_path="eval_data.jsonl",
    )

With custom rewrite LLM::

    from langchain_openai import ChatOpenAI
    rewrite_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    loop = PromptOptimizationLoop(
        ...,
        rewrite_llm=rewrite_llm,
    )
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from loguru import logger as _logger
except ImportError:
    import logging as _logging

    _logger = _logging.getLogger("agentomatic.optimize")  # type: ignore[assignment]


# =====================================================================
# Data types
# =====================================================================


@dataclass
class StepResult:
    """Result of a single optimization step."""

    step: int
    prompt: str
    avg_score: float
    accuracy: float
    results: list[dict[str, Any]]
    elapsed: float


@dataclass
class LoopResult:
    """Full result of an optimization run.

    Attributes:
        agent: Agent identifier.
        experiment_id: Unique run identifier.
        steps: History of all optimization steps.
        best_step: Index of the best-scoring step.
        best_score: Highest average score achieved.
        best_prompt: The prompt that produced the best score.
        total_elapsed: Total wall-clock time (seconds).
        config: Run configuration for reproducibility.
    """

    agent: str = ""
    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = ""
    steps: list[StepResult] = field(default_factory=list)
    best_step: int = 0
    best_score: float = 0.0
    best_prompt: str = ""
    total_elapsed: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    # -- Properties ---------------------------------------------------

    @property
    def baseline_score(self) -> float:
        """Score from the first (unoptimised) step."""
        return self.steps[0].avg_score if self.steps else 0.0

    @property
    def improvement(self) -> float:
        """Percentage improvement over baseline."""
        if self.baseline_score == 0:
            return 0.0
        return ((self.best_score - self.baseline_score) / self.baseline_score) * 100

    @property
    def scores(self) -> list[float]:
        """Score evolution across steps."""
        return [s.avg_score for s in self.steps]

    @property
    def improved(self) -> bool:
        """Whether optimization improved over baseline."""
        return self.best_score > self.baseline_score

    # -- Text summary -------------------------------------------------

    def summary(self) -> str:
        """Generate a human-readable text summary."""
        lines = [
            "",
            "━" * 60,
            f"⚡ OPTIMIZATION SUMMARY — {self.agent}",
            "━" * 60,
        ]
        for s in self.steps:
            marker = " 🏆" if s.step == self.best_step else ""
            delta = ""
            if s.step > 0:
                prev = self.steps[s.step - 1].avg_score
                diff = s.avg_score - prev
                delta = f" ({'+' if diff >= 0 else ''}{diff:.1%})"
            lines.append(
                f"  Step {s.step + 1}: avg={s.avg_score:.1%}  acc={s.accuracy:.0%}{delta}{marker}"
            )
        lines.extend(
            [
                "",
                f"  Baseline:    {self.baseline_score:.1%}",
                f"  Best:        {self.best_score:.1%} (step {self.best_step + 1})",
                f"  Improvement: {self.improvement:+.1f}%",
                f"  Total time:  {self.total_elapsed:.1f}s",
                "━" * 60,
            ]
        )
        return "\n".join(lines)

    # -- Persistence --------------------------------------------------

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        """Save results as JSON + HTML report.

        Args:
            directory: Output directory (created if needed).

        Returns:
            ``(json_path, html_path)`` tuple.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "optimization_results.json"
        json_path.write_text(
            json.dumps(self._to_dict(), indent=2, ensure_ascii=False, default=str)
        )
        _logger.info(f"📄 Results: {json_path}")

        html_path = out / "optimization_report.html"
        _write_html_report(self, str(html_path))
        _logger.info(f"📊 Report:  {html_path}")

        return json_path, html_path

    def _to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "agent": self.agent,
            "started_at": self.started_at,
            "baseline_score": round(self.baseline_score, 4),
            "best_score": round(self.best_score, 4),
            "best_step": self.best_step,
            "improvement_pct": round(self.improvement, 2),
            "total_elapsed": round(self.total_elapsed, 1),
            "config": self.config,
            "steps": [
                {
                    "step": s.step,
                    "avg_score": round(s.avg_score, 4),
                    "accuracy": round(s.accuracy, 4),
                    "elapsed": round(s.elapsed, 1),
                    "prompt_length": len(s.prompt),
                    "n_pass": sum(1 for r in s.results if r["score"] >= 0.5),
                    "n_fail": sum(1 for r in s.results if r["score"] < 0.5),
                }
                for s in self.steps
            ],
            "best_prompt": self.best_prompt,
        }

    @classmethod
    def load(cls, path: str | Path) -> LoopResult:
        """Load a previously saved result from JSON."""
        data = json.loads(Path(path).read_text())
        result = cls(
            agent=data["agent"],
            experiment_id=data.get("experiment_id", ""),
            started_at=data.get("started_at", ""),
            best_step=data["best_step"],
            best_score=data["best_score"],
            best_prompt=data.get("best_prompt", ""),
            total_elapsed=data.get("total_elapsed", 0.0),
            config=data.get("config", {}),
        )
        return result


# =====================================================================
# Rewrite strategies
# =====================================================================

_STRATEGY_PROMPTS: dict[str, str] = {
    "iterative": (
        "You are a world-class prompt engineer. Improve the system prompt "
        "based on evaluation feedback.\n\n"
        "Rules:\n"
        "- Preserve behaviours that scored well\n"
        "- Fix the specific failure patterns shown below\n"
        "- Make instructions more precise and actionable\n"
        "- Keep the same overall purpose and constraints\n"
        "- Return ONLY the improved system prompt, nothing else\n"
        "- Do NOT add explanations, commentary, or markdown fences"
    ),
    "adversarial": (
        "You are a world-class prompt engineer performing adversarial analysis.\n"
        "For each failure, identify the ROOT CAUSE — is it ambiguous instructions? "
        "missing constraints? wrong tone? conflicting rules? wrong language?\n"
        "Then rewrite the prompt to eliminate each root cause.\n\n"
        "Return ONLY the improved system prompt, no commentary."
    ),
    "structured": (
        "You are a world-class prompt engineer specializing in structured prompts.\n"
        "Rewrite the prompt using this structure:\n"
        "1. ROLE — who the assistant is\n"
        "2. TASK — what it must do\n"
        "3. CONSTRAINTS — rules it must follow (language, tone, format)\n"
        "4. FORMAT — how to format responses\n"
        "5. EXAMPLES — 1-2 inline examples if helpful\n\n"
        "Return ONLY the improved system prompt, no commentary."
    ),
    "minimal": (
        "You are a prompt engineer focused on clarity and conciseness.\n"
        "Rewrite the prompt to be shorter and clearer while fixing failures.\n"
        "Remove redundant instructions. Every sentence must earn its place.\n\n"
        "Return ONLY the improved system prompt."
    ),
}

AVAILABLE_STRATEGIES = list(_STRATEGY_PROMPTS.keys())


# =====================================================================
# Scoring helpers (generic, no external deps)
# =====================================================================


def keyword_overlap(expected: str, actual: str) -> float:
    """Simple keyword overlap scorer (no LLM needed).

    Computes the fraction of expected words found in the actual response.
    """
    if not expected or not actual:
        return 0.0
    exp = set(expected.lower().split())
    act = set(actual.lower().split())
    return len(exp & act) / len(exp) if exp else 0.0


def contains_score(expected: str, actual: str) -> float:
    """Check if comma-separated parts of expected appear in actual."""
    if not expected or not actual:
        return 0.0
    actual_lower = actual.lower()
    parts = [p.strip() for p in expected.lower().split(",") if p.strip()]
    if not parts:
        parts = [expected.lower()]
    found = sum(1 for p in parts if p in actual_lower)
    return found / len(parts)


# =====================================================================
# Core Loop
# =====================================================================


class PromptOptimizationLoop:
    """Iterative prompt optimization engine — fully generic.

    Evaluates a system prompt against a dataset, analyses failures,
    and uses an LLM to rewrite the prompt iteratively.

    This engine is **framework-agnostic**: it works with any LLM that
    implements ``ainvoke(messages)`` (LangChain protocol) or a simple
    async callable.

    Args:
        agent_name: Agent identifier (for reports and logging).
        invoke_fn: Async callable ``(datapoint, prompt) -> {"actual_response": str}``.
            ``datapoint`` is a dict from your JSONL dataset.
            ``prompt`` is the current system prompt string.
        score_fn: Scoring function — either:
            - ``(expected: str, actual: str) -> float`` (sync)
            - ``async (expected: str, actual: str, dp: dict) -> float`` (async)
            Must return a float in [0.0, 1.0].
        dataset_path: Path to a JSONL file. Each line must have at least
            ``"query"`` and ``"expected_response"`` keys.
        threshold: Score threshold for "pass" (default 0.3).
        language: Expected response language (used in rewrite hints).
        strategy: Rewrite strategy — one of
            ``"iterative"``, ``"adversarial"``, ``"structured"``, ``"minimal"``.
        rewrite_llm: LangChain-compatible LLM for prompt rewriting.
            Must implement ``ainvoke(messages) -> AIMessage``. If ``None``,
            rewriting is skipped and only evaluation runs.
    """

    def __init__(
        self,
        agent_name: str,
        invoke_fn: Callable[[dict, str], Awaitable[dict[str, Any]]],
        score_fn: Callable[..., Any],
        dataset_path: str | Path,
        threshold: float = 0.3,
        language: str = "English",
        strategy: str = "iterative",
        rewrite_llm: Any = None,
    ):
        self.agent_name = agent_name
        self.invoke_fn = invoke_fn
        self.score_fn = score_fn
        self.dataset_path = str(dataset_path)
        self.threshold = threshold
        self.language = language
        self.strategy = strategy
        self.rewrite_llm = rewrite_llm

        if strategy not in _STRATEGY_PROMPTS:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {AVAILABLE_STRATEGIES}")

    async def run(
        self,
        initial_prompt: str,
        steps: int = 5,
        target_score: float = 0.95,
        patience: int = 3,
        min_improvement: float = 0.005,
    ) -> LoopResult:
        """Run the full optimization loop.

        Args:
            initial_prompt: Starting system prompt text.
            steps: Maximum number of optimisation iterations.
            target_score: Stop early when avg score reaches this.
            patience: Stop after N consecutive steps without improvement.
            min_improvement: Minimum score delta to count as progress.

        Returns:
            :class:`LoopResult` with full history and the best prompt.
        """
        dataset = _load_dataset(self.dataset_path)

        result = LoopResult(
            agent=self.agent_name,
            started_at=datetime.now(UTC).isoformat(),
            config={
                "steps": steps,
                "target_score": target_score,
                "patience": patience,
                "min_improvement": min_improvement,
                "threshold": self.threshold,
                "strategy": self.strategy,
                "language": self.language,
                "dataset_size": len(dataset),
                "has_rewrite_llm": self.rewrite_llm is not None,
            },
        )

        current_prompt = initial_prompt
        no_improve_count = 0
        t_total = time.time()

        _logger.info("━" * 60)
        _logger.info(f"⚡ PROMPT OPTIMIZATION — {self.agent_name} ({steps} steps)")
        _logger.info("━" * 60)
        _logger.info(
            f"📋 Dataset: {len(dataset)} samples | Strategy: {self.strategy} | "
            f"Threshold: {self.threshold:.0%}"
        )

        for step_idx in range(steps):
            t_step = time.time()

            _logger.info("")
            _logger.info(f"┌─── Step {step_idx + 1}/{steps} " + "─" * 35)

            # ── 1. Evaluate ──────────────────────────────────────────
            step_results, avg_score, accuracy = await self._evaluate(dataset, current_prompt)
            elapsed = time.time() - t_step

            step = StepResult(
                step=step_idx,
                prompt=current_prompt,
                avg_score=avg_score,
                accuracy=accuracy,
                results=step_results,
                elapsed=elapsed,
            )
            result.steps.append(step)

            # Track best
            if avg_score > result.best_score + min_improvement:
                result.best_score = avg_score
                result.best_step = step_idx
                result.best_prompt = current_prompt
                no_improve_count = 0
            elif step_idx == 0:
                # Always save baseline as initial best
                result.best_score = avg_score
                result.best_prompt = current_prompt
            else:
                no_improve_count += 1

            n_pass = sum(1 for r in step_results if r["score"] >= self.threshold)
            _logger.info(
                f"│ Avg Score: {avg_score:.1%} | Accuracy: {accuracy:.0%} "
                f"({n_pass}/{len(step_results)}) | Time: {elapsed:.1f}s"
            )

            # ── 2. Early stopping ────────────────────────────────────
            if avg_score >= target_score:
                _logger.info(f"│ 🎯 Target score {target_score:.0%} reached!")
                _logger.info("└" + "─" * 45)
                break

            if no_improve_count >= patience and step_idx > 0:
                _logger.info(f"│ ⏹️  Patience exhausted ({patience} steps w/o improvement)")
                _logger.info("└" + "─" * 45)
                break

            if step_idx == steps - 1:
                _logger.info("└" + "─" * 45)
                break

            # ── 3. Rewrite prompt ────────────────────────────────────
            if self.rewrite_llm is None:
                _logger.info("│ ⚠️  No rewrite LLM — skipping prompt rewrite")
                _logger.info("└" + "─" * 45)
                continue

            failures = [r for r in step_results if r["score"] < self.threshold]
            successes = [r for r in step_results if r["score"] >= self.threshold]

            _logger.info(
                f"│ 📝 Analysing {len(failures)} failures, {len(successes)} successes — rewriting…"
            )

            new_prompt = await self._rewrite_prompt(
                current_prompt, failures, successes, avg_score, step_idx
            )

            if new_prompt and new_prompt.strip() != current_prompt.strip():
                current_prompt = new_prompt
                _logger.info(f"│ ✅ Prompt updated ({len(new_prompt)} chars)")
            else:
                _logger.info("│ ⚠️  Rewrite unchanged — keeping current prompt")

            _logger.info("└" + "─" * 45)

        result.total_elapsed = time.time() - t_total
        _logger.info(result.summary())

        return result

    # -----------------------------------------------------------------
    # Evaluate all samples
    # -----------------------------------------------------------------

    async def _evaluate(
        self,
        dataset: list[dict],
        prompt: str,
    ) -> tuple[list[dict], float, float]:
        """Evaluate every dataset sample with the given prompt."""
        results: list[dict] = []
        correct = 0
        is_async_scorer = inspect.iscoroutinefunction(self.score_fn)

        for dp in dataset:
            expected = dp.get("expected_response", "")
            try:
                output = await self.invoke_fn(dp, prompt)
                actual = output.get("actual_response", "")
            except Exception as exc:
                _logger.debug(f"  Invoke error: {exc}")
                actual = f"[ERROR] {exc}"

            # Score — sync or async
            try:
                if is_async_scorer:
                    score = await self.score_fn(expected, actual, dp)
                else:
                    score = self.score_fn(expected, actual)
                score = max(0.0, min(1.0, float(score)))
            except Exception as exc:
                _logger.debug(f"  Score error: {exc}")
                score = 0.0

            if score >= self.threshold:
                correct += 1

            results.append(
                {
                    "query": dp.get("query", ""),
                    "expected": expected,
                    "actual": actual[:500],
                    "category": dp.get("category", "general"),
                    "score": round(score, 4),
                }
            )

        n = len(results)
        avg = sum(r["score"] for r in results) / n if n else 0.0
        acc = correct / n if n else 0.0
        return results, avg, acc

    # -----------------------------------------------------------------
    # Prompt rewriting via LLM
    # -----------------------------------------------------------------

    async def _rewrite_prompt(
        self,
        current_prompt: str,
        failures: list[dict],
        successes: list[dict],
        avg_score: float,
        step: int,
    ) -> str | None:
        """Use the rewrite LLM to improve the prompt."""
        if self.rewrite_llm is None:
            return None

        try:
            strategy_system = _STRATEGY_PROMPTS[self.strategy]

            # Build analysis text
            failure_text = (
                "\n\n".join(
                    f"  Query: {f['query'][:100]}\n"
                    f"  Expected: {f['expected'][:150]}\n"
                    f"  Got: {f['actual'][:150]}\n"
                    f"  Score: {f['score']:.0%}"
                    for f in failures[:5]
                )
                or "(none — all samples passed)"
            )

            success_text = (
                "\n\n".join(
                    f"  Query: {s['query'][:100]}\n"
                    f"  Response: {s['actual'][:150]}\n"
                    f"  Score: {s['score']:.0%}"
                    for s in successes[:3]
                )
                or "(none)"
            )

            human_msg = (
                f"Response language constraint: ALL responses MUST be in {self.language}.\n\n"
                f"CURRENT PROMPT:\n```\n{current_prompt}\n```\n\n"
                f"CURRENT SCORE: {avg_score:.0%} (step {step + 1})\n\n"
                f"FAILURES ({len(failures)} samples below {self.threshold:.0%} threshold):\n"
                f"{failure_text}\n\n"
                f"SUCCESSES ({len(successes)} good examples):\n"
                f"{success_text}\n\n"
                f"Write the improved system prompt:"
            )

            # Use LangChain protocol if available
            new_prompt = await self._call_llm(strategy_system, human_msg)

            if not new_prompt or len(new_prompt.strip()) < 20:
                return None

            # Clean markdown fences
            new_prompt = new_prompt.strip()
            if new_prompt.startswith("```"):
                lines = new_prompt.split("\n")
                end = -1 if lines[-1].strip() == "```" else len(lines)
                new_prompt = "\n".join(lines[1:end]).strip()

            return new_prompt

        except Exception as exc:
            _logger.warning(f"Prompt rewrite failed: {exc}")
            return None

    async def _call_llm(self, system: str, human: str) -> str:
        """Call the rewrite LLM using LangChain protocol or raw callable."""
        llm = self.rewrite_llm

        # LangChain protocol: ainvoke with message list
        if hasattr(llm, "ainvoke"):
            from langchain_core.messages import HumanMessage, SystemMessage

            result = await llm.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=human)]
            )
            return result.content if hasattr(result, "content") else str(result)

        # Sync invoke fallback
        if hasattr(llm, "invoke"):
            from langchain_core.messages import HumanMessage, SystemMessage

            result = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            return result.content if hasattr(result, "content") else str(result)

        # Raw async callable
        if callable(llm) and inspect.iscoroutinefunction(llm):
            return await llm(system + "\n\n" + human)

        # Raw sync callable
        if callable(llm):
            return llm(system + "\n\n" + human)

        raise TypeError(
            f"rewrite_llm must implement ainvoke/invoke (LangChain) "
            f"or be callable. Got: {type(llm)}"
        )


# =====================================================================
# Dataset loader
# =====================================================================


def _load_dataset(path: str) -> list[dict]:
    """Load a JSONL dataset.

    Each line must be a JSON object with at least ``"query"`` key.
    Optional keys: ``"expected_response"``, ``"category"``, ``"context"``.
    """
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise ValueError(f"Empty dataset: {path}")
    return items


# =====================================================================
# HTML Report Generator
# =====================================================================


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_html_report(result: LoopResult, path: str) -> None:
    """Write a self-contained dark-themed HTML optimisation report."""
    scores = result.scores

    # ── Step table rows ──
    step_rows = []
    for s in result.steps:
        marker = "🏆" if s.step == result.best_step else ""
        delta_html = ""
        if s.step > 0:
            prev = result.steps[s.step - 1].avg_score
            diff = s.avg_score - prev
            color = "#34d399" if diff >= 0 else "#f87171"
            sign = "+" if diff >= 0 else ""
            delta_html = f"<span style='color:{color}'>{sign}{diff:.1%}</span>"

        sc = "#34d399" if s.avg_score >= 0.7 else "#fbbf24" if s.avg_score >= 0.4 else "#f87171"
        n_pass = sum(1 for r in s.results if r["score"] >= 0.5)
        n_fail = len(s.results) - n_pass
        step_rows.append(
            f"<tr>"
            f"<td>{s.step + 1} {marker}</td>"
            f"<td style='color:{sc};font-weight:700'>{s.avg_score:.1%}</td>"
            f"<td>{s.accuracy:.0%}</td>"
            f"<td>{delta_html}</td>"
            f"<td>{n_pass}✅ / {n_fail}❌</td>"
            f"<td>{s.elapsed:.1f}s</td>"
            f"<td>{len(s.prompt)}</td>"
            f"</tr>"
        )

    # ── SVG evolution chart ──
    w, h = 600, 200
    if len(scores) > 1:
        x_step = w / (len(scores) - 1)
        y_lo = max(min(scores) - 0.1, 0)
        y_hi = min(max(scores) + 0.1, 1.0)
        y_range = y_hi - y_lo if y_hi > y_lo else 1.0

        pts, dots, labels = [], [], []
        for i, sc in enumerate(scores):
            x = i * x_step
            y = h - ((sc - y_lo) / y_range) * h
            pts.append(f"{x:.0f},{y:.0f}")
            fill = "#34d399" if i == result.best_step else "#818cf8"
            dots.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='6' fill='{fill}'/>")
            labels.append(
                f"<text x='{x:.0f}' y='{y - 12:.0f}' fill='#e2e8f0' "
                f"font-size='11' text-anchor='middle'>{sc:.0%}</text>"
            )

        grid_lines = ""
        for pct in [0.25, 0.5, 0.75, 1.0]:
            gy = h - ((pct - y_lo) / y_range) * h
            if 0 <= gy <= h:
                grid_lines += (
                    f"<line x1='0' y1='{gy:.0f}' x2='{w}' y2='{gy:.0f}' "
                    f"stroke='#334155' stroke-dasharray='4'/>"
                    f"<text x='-8' y='{gy + 4:.0f}' fill='#94a3b8' "
                    f"font-size='10' text-anchor='end'>{pct:.0%}</text>"
                )

        svg = (
            f"<svg viewBox='-40 -25 {w + 60} {h + 40}' "
            f"style='width:100%;max-width:750px;background:#1e293b;"
            f"border-radius:12px;padding:1rem;border:1px solid #334155'>"
            f"{grid_lines}"
            f"<polyline points='{' '.join(pts)}' fill='none' stroke='#818cf8' "
            f"stroke-width='3' stroke-linejoin='round'/>"
            f"{''.join(dots)}{''.join(labels)}"
            f"</svg>"
        )
    else:
        svg = "<p style='color:#94a3b8'>Single step — no evolution chart.</p>"

    # ── Category breakdown ──
    best_results = result.steps[result.best_step].results if result.steps else []
    cats: dict[str, list[float]] = {}
    for r in best_results:
        cats.setdefault(r.get("category", "general"), []).append(r["score"])

    cat_rows = []
    for cat, ss in sorted(cats.items()):
        avg = sum(ss) / len(ss)
        bar_w = int(avg * 100)
        c = "#34d399" if avg >= 0.7 else "#fbbf24" if avg >= 0.4 else "#f87171"
        cat_rows.append(
            f"<tr><td>{_esc(cat)}</td><td>{len(ss)}</td>"
            f"<td><div class='bar-bg'><div class='bar' "
            f"style='width:{bar_w}%;background:{c}'></div></div></td>"
            f"<td>{avg:.0%}</td></tr>"
        )

    imp_c = "#34d399" if result.improvement >= 0 else "#f87171"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>⚡ Optimization — {_esc(result.agent)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0f172a;--card:#1e293b;--accent:#818cf8;--green:#34d399;--red:#f87171;
--yellow:#fbbf24;--text:#e2e8f0;--muted:#94a3b8;--border:#334155;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);
padding:2rem;max-width:1300px;margin:auto;}}
h1{{color:var(--accent);font-size:1.8rem;margin-bottom:0.3rem;}}
h2{{color:var(--accent);margin:2rem 0 1rem;font-size:1.3rem;}}
.sub{{color:var(--muted);margin-bottom:2rem;font-size:0.9rem;}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
gap:1rem;margin-bottom:2rem;}}
.kpi{{background:var(--card);border-radius:12px;padding:1.5rem;text-align:center;
border:1px solid var(--border);}}
.kpi .v{{font-size:2rem;font-weight:700;}}
.kpi .l{{color:var(--muted);font-size:0.8rem;margin-top:0.3rem;}}
.kpi.green .v{{color:var(--green);}}.kpi.accent .v{{color:var(--accent);}}
table{{width:100%;border-collapse:collapse;background:var(--card);border-radius:12px;
overflow:hidden;margin-bottom:2rem;border:1px solid var(--border);}}
th{{background:#334155;text-align:left;padding:0.75rem 1rem;font-size:0.8rem;
color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;}}
td{{padding:0.75rem 1rem;border-top:1px solid var(--border);font-size:0.85rem;}}
tr:hover{{background:#334155;}}
.bar-bg{{background:#334155;border-radius:4px;height:20px;width:140px;overflow:hidden;}}
.bar{{height:100%;border-radius:4px;transition:width 0.5s;}}
.chart{{margin:1rem 0 2rem;}}
pre{{background:var(--card);padding:1rem;border-radius:8px;overflow-x:auto;font-size:0.8rem;
border:1px solid var(--border);white-space:pre-wrap;word-break:break-word;max-height:400px;
overflow-y:auto;}}
.footer{{text-align:center;color:#475569;margin-top:3rem;font-size:0.8rem;padding:1rem;}}
</style>
</head>
<body>
<h1>⚡ Prompt Optimization — {_esc(result.agent)}</h1>
<p class="sub">ID: <code>{result.experiment_id}</code> · {len(result.steps)} steps · {result.total_elapsed:.1f}s · Strategy: {result.config.get("strategy", "?")}</p>

<div class="kpi-grid">
  <div class="kpi accent"><div class="v">{result.baseline_score:.0%}</div><div class="l">Baseline</div></div>
  <div class="kpi green"><div class="v">{result.best_score:.0%}</div><div class="l">Best Score</div></div>
  <div class="kpi" style="border-color:{imp_c}"><div class="v" style="color:{imp_c}">{result.improvement:+.1f}%</div><div class="l">Improvement</div></div>
  <div class="kpi accent"><div class="v">{result.best_step + 1}</div><div class="l">Best Step</div></div>
  <div class="kpi accent"><div class="v">{result.config.get("dataset_size", "?")}</div><div class="l">Samples</div></div>
</div>

<h2>📈 Score Evolution</h2>
<div class="chart">{svg}</div>

<h2>📋 Step Details</h2>
<table>
<tr><th>Step</th><th>Avg Score</th><th>Accuracy</th><th>Δ</th><th>Pass/Fail</th><th>Time</th><th>Prompt Len</th></tr>
{"".join(step_rows)}
</table>

{"<h2>📂 Category Breakdown (Best Step)</h2>" if cat_rows else ""}
{"<table><tr><th>Category</th><th>Samples</th><th>Score</th><th>Avg</th></tr>" + "".join(cat_rows) + "</table>" if cat_rows else ""}

<h2>🏆 Best Prompt (Step {result.best_step + 1})</h2>
<pre>{_esc(result.best_prompt[:3000])}</pre>

<div class="footer">Agentomatic Prompt Optimization Engine v1.0</div>
</body>
</html>"""

    Path(path).write_text(html, encoding="utf-8")
