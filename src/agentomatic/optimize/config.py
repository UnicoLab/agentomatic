"""Core data models for the PromptFitter system.

Defines the optimizable configuration surface and structured result types
used throughout the prompt-fitting pipeline.  Every class uses
``@dataclass(slots=True)`` for memory efficiency and faster attribute
access.

Key types:

- **PromptRuntimeConfig** — full snapshot of everything that influences an
  LLM call: system prompt, user template, few-shot examples, output
  contract, model params (temperature …), RAG params (top_k …), and tool
  params (max_tool_calls …).

- **ParamDelta** — a single parameter change with before/after values and
  a human-readable reason.

- **PromptCandidate** — a named configuration produced by an optimization
  strategy, with lineage tracking and per-metric scores.

- **PromptFitResult** — the final output of an optimization run:
  best/baseline configs, metric deltas, suggestions, failure clusters,
  and an ``apply()`` method to persist changes.

Example::

    from agentomatic.optimize.config import (
        PromptRuntimeConfig,
        PromptCandidate,
        PromptFitResult,
    )

    cfg = PromptRuntimeConfig(
        system_prompt="You are a helpful assistant.",
        model_params={"temperature": 0.3},
    )
    print(cfg.to_dict())
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


def escape_braces(text: str, preserve: list[str] | None = None) -> str:
    """Escape literal braces while preserving specific template variables.

    Replaces '{' with '{{' and '}' with '}}', except for variables listed in `preserve`
    (e.g., 'query' for '{query}'). This prevents KeyError during subsequent .format() calls
    on system prompts that contain literal JSON or markdown code blocks.
    """
    if not text:
        return text

    # First escape all
    escaped = text.replace("{", "{{").replace("}", "}}")

    # Then unescape the specific variables we want to preserve
    if preserve:
        for var in preserve:
            escaped = escaped.replace(f"{{{{{var}}}}}", f"{{{var}}}")

    return escaped


# =====================================================================
# Runtime configuration
# =====================================================================


@dataclass(slots=True)
class PromptRuntimeConfig:
    """Full snapshot of everything that influences a single LLM call.

    Captures the *entire* controllable surface: prompt text, few-shot
    examples, output format constraints, model hyper-parameters, RAG
    retrieval settings, and tool-use policy.

    Examples::

        cfg = PromptRuntimeConfig(
            system_prompt="Answer concisely.",
            user_template="Question: {query}\\nContext: {context}",
            few_shot_examples=[
                {"query": "What is 2+2?", "response": "4"},
            ],
            model_params={"temperature": 0.2, "max_tokens": 512},
            rag_params={"top_k": 5, "min_similarity": 0.75},
        )
        serialized = cfg.to_dict()
        restored  = PromptRuntimeConfig.from_dict(serialized)
    """

    system_prompt: str
    user_template: str | None = None
    few_shot_examples: list[dict[str, Any]] = field(default_factory=list)
    output_contract: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    rag_params: dict[str, Any] = field(default_factory=dict)
    tool_params: dict[str, Any] = field(default_factory=dict)

    # ── Deployment-first fields ────────────────────────────────────
    model_choice: str | None = None  # e.g. "ollama/qwen2.5:7b"
    fallback_model: str | None = None  # automatic fallback model
    routing_config: dict[str, Any] = field(default_factory=dict)  # A/B weights, local/remote

    @property
    def safe_system_prompt(self) -> str:
        """Returns the system prompt with literal braces safely escaped.

        Assumes the system prompt does not contain runtime format variables.
        Useful when passing the system prompt into LangChain or other formatters
        that might crash on literal JSON blocks.
        """
        return escape_braces(self.system_prompt)

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full configuration to a plain dictionary.

        Returns:
            Nested dict suitable for ``json.dumps``.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptRuntimeConfig:
        """Reconstruct a config from a dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict` or equivalent.

        Returns:
            A new ``PromptRuntimeConfig`` instance.
        """
        return cls(
            system_prompt=data.get("system_prompt", ""),
            user_template=data.get("user_template"),
            few_shot_examples=data.get("few_shot_examples", []),
            output_contract=data.get("output_contract"),
            model_params=data.get("model_params", {}),
            rag_params=data.get("rag_params", {}),
            tool_params=data.get("tool_params", {}),
            model_choice=data.get("model_choice"),
            fallback_model=data.get("fallback_model"),
            routing_config=data.get("routing_config", {}),
        )

    # -- comparison ------------------------------------------------------

    def diff(self, other: PromptRuntimeConfig) -> dict[str, Any]:
        """Compute field-level differences between two configs.

        Args:
            other: The config to compare against (usually the baseline).

        Returns:
            Dict mapping field names to ``{"old": …, "new": …}`` pairs
            for every field whose value changed.

        Example::

            delta = new_cfg.diff(old_cfg)
            # {"model_params": {"old": {"temperature": 0.7},
            #                   "new": {"temperature": 0.3}}}
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        changes: dict[str, Any] = {}
        for key in self_dict:
            if self_dict[key] != other_dict.get(key):
                changes[key] = {"old": other_dict.get(key), "new": self_dict[key]}
        return changes

    # -- few-shot rendering ----------------------------------------------

    def format_few_shot_block(self) -> str:
        """Render ``few_shot_examples`` as a human-readable text block.

        Each example is formatted as a numbered Q/A pair::

            [Example 1]
            Q: What is 2+2?
            A: 4

        Returns:
            Formatted string, or empty string if no examples exist.
        """
        if not self.few_shot_examples:
            return ""
        lines: list[str] = []
        for idx, ex in enumerate(self.few_shot_examples, 1):
            lines.append(f"[Example {idx}]")
            lines.append(f"Q: {ex.get('query', '')}")
            lines.append(f"A: {ex.get('response', '')}")
            lines.append("")
        return "\n".join(lines).rstrip("\n")


# =====================================================================
# Parameter delta
# =====================================================================


@dataclass(slots=True)
class ParamDelta:
    """Describes a single parameter change between two configs.

    Useful for building human-readable changelogs and for auditing
    which knobs the optimizer decided to tweak.

    Examples::

        delta = ParamDelta(
            param_name="temperature",
            old_value=0.7,
            new_value=0.3,
            reason="Lower temperature reduces hallucination on factual QA.",
        )
        print(f"{delta.param_name}: {delta.old_value} → {delta.new_value}")
    """

    param_name: str
    old_value: Any
    new_value: Any
    reason: str = ""


# =====================================================================
# Prompt candidate
# =====================================================================


@dataclass(slots=True)
class PromptCandidate:
    """A named configuration produced by an optimization strategy.

    Tracks lineage (which parent it was derived from), what changed,
    and per-metric evaluation scores.

    Examples::

        candidate = PromptCandidate(
            name="gepa_rewrite_003",
            config=PromptRuntimeConfig(system_prompt="Be concise."),
            source="rewrite",
            parent="gepa_rewrite_002",
            mutation_notes="Shortened system prompt to reduce verbosity.",
            scores={"answer_relevancy": 0.91, "faithfulness": 0.88},
            composite_score=0.895,
        )
    """

    name: str
    config: PromptRuntimeConfig
    source: str
    parent: str | None = None
    mutation_notes: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0


# =====================================================================
# Fit result
# =====================================================================


@dataclass(slots=True)
class PromptFitResult:
    """Final output of a prompt-fitting run.

    Contains the best and baseline configurations, per-metric deltas,
    suggested parameter changes, failure clusters, and a full trial
    history.  Call :meth:`summary` for a human-readable report or
    :meth:`apply` to persist the optimised configuration to disk.

    Examples::

        result = PromptFitResult(
            best_config=optimised_cfg,
            baseline_config=original_cfg,
            best_score=0.92,
            baseline_score=0.78,
            metric_deltas={"answer_relevancy": 0.14},
            suggestions=["Add few-shot examples for edge cases."],
            agent="support_bot",
        )
        print(result.summary())
        result.apply(version="v3_fit", agent_dir="agents/support_bot")
    """

    best_config: PromptRuntimeConfig
    baseline_config: PromptRuntimeConfig
    best_score: float
    baseline_score: float
    metric_deltas: dict[str, float] = field(default_factory=dict)
    param_suggestions: dict[str, ParamDelta] = field(default_factory=dict)
    failure_clusters: list[dict[str, Any]] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    experiment_id: str = field(
        default_factory=lambda: __import__("uuid").uuid4().hex[:12],
    )
    agent: str = ""
    deployment_recommendation: Any = None  # DeploymentRecommendation
    score_history: list[float] = field(default_factory=list)
    """Per-round best scores (chronological). Populated by ``PromptFitter.fit()``."""

    # -- convenience properties ------------------------------------------

    @property
    def absolute_improvement(self) -> float:
        """Score lift: ``best_score − baseline_score``."""
        return self.best_score - self.baseline_score

    @property
    def history(self) -> list[float]:
        """Per-round best scores — Keras-style history list.

        Returns scores stored in :attr:`score_history`, falling back to
        extracting the best ``"full_val"`` score from each trial round
        when ``score_history`` is empty (backward compatibility).
        """
        if self.score_history:
            return list(self.score_history)
        # Fallback: derive from trials list for backward compat
        rounds: dict[int, float] = {}
        for t in self.trials:
            r = t.get("round", 0)
            s = t.get("score", 0.0)
            if t.get("phase") == "full_val":
                rounds[r] = max(rounds.get(r, 0.0), s)
        if rounds:
            return [rounds[k] for k in sorted(rounds)]
        return [t.get("score", 0.0) for t in self.trials]

    @property
    def best_prompt(self) -> str:
        """Shortcut to ``best_config.system_prompt``."""
        return self.best_config.system_prompt

    @property
    def best_params(self) -> dict[str, Any]:
        """Shortcut to ``best_config.model_params``."""
        return self.best_config.model_params

    @property
    def best_few_shot_examples(self) -> list[dict[str, Any]]:
        """Shortcut to ``best_config.few_shot_examples``."""
        return self.best_config.few_shot_examples

    # -- human-readable summary ------------------------------------------

    def summary(self) -> str:
        """Build a multi-line human-readable summary of the fit run.

        Includes baseline/best scores, absolute improvement, main
        improvements, regressions (negative metric deltas), and a
        final recommendation.

        Returns:
            Formatted string ready for ``print()`` or logging.
        """
        lines: list[str] = [
            "=" * 60,
            f"  PromptFit Result — experiment {self.experiment_id}",
            "=" * 60,
            f"  Agent:    {self.agent or '(not set)'}",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  Trials:   {len(self.trials)}",
            "",
            f"  Baseline score: {self.baseline_score:.4f}",
            f"  Best score:     {self.best_score:.4f}",
            f"  Improvement:    {self.absolute_improvement:+.4f}",
            "",
        ]

        # Improvements
        if self.suggestions:
            lines.append("  Main improvements:")
            for suggestion in self.suggestions:
                lines.append(f"    • {suggestion}")
            lines.append("")

        # Regressions
        regressions = {k: v for k, v in self.metric_deltas.items() if v < 0}
        if regressions:
            lines.append("  ⚠ Regressions:")
            for metric, delta in regressions.items():
                lines.append(f"    • {metric}: {delta:+.4f}")
            lines.append("")

        # Recommendation
        if self.absolute_improvement > 0.05:
            lines.append("  ✅ Recommendation: apply the optimised config.")
        elif self.absolute_improvement > 0:
            lines.append("  🔶 Recommendation: marginal gain — review before applying.")
        else:
            lines.append("  ❌ Recommendation: keep the baseline config.")

        lines.append("=" * 60)
        return "\n".join(lines)

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result to a JSON-compatible dictionary.

        Returns:
            Nested dict including both configs, scores, deltas, and
            all trial metadata.
        """
        return {
            "experiment_id": self.experiment_id,
            "agent": self.agent,
            "best_score": self.best_score,
            "baseline_score": self.baseline_score,
            "absolute_improvement": self.absolute_improvement,
            "score_history": self.score_history,
            "metric_deltas": self.metric_deltas,
            "param_suggestions": {k: asdict(v) for k, v in self.param_suggestions.items()},
            "best_config": self.best_config.to_dict(),
            "baseline_config": self.baseline_config.to_dict(),
            "failure_clusters": self.failure_clusters,
            "trials": self.trials,
            "suggestions": self.suggestions,
            "duration_seconds": self.duration_seconds,
            "deployment_recommendation": (
                self.deployment_recommendation.to_dict()
                if self.deployment_recommendation
                else None
            ),
        }

    # -- persistence -----------------------------------------------------

    def apply(
        self,
        version: str = "v2_fit",
        agent_dir: str | None = None,
    ) -> str:
        """Persist the optimised configuration to disk.

        Writes two files into *agent_dir* (defaults to ``"."``):

        1. ``prompts.json`` — adds a new version entry with the system
           prompt, user template, and metadata.
        2. ``runtime_config.json`` — full :class:`PromptRuntimeConfig`
           serialised as JSON.

        Args:
            version:   Version key for the new prompt entry.
            agent_dir: Directory to write files into.  Defaults to cwd.

        Returns:
            The *version* key that was written.
        """
        base = Path(agent_dir) if agent_dir else Path(".")
        base.mkdir(parents=True, exist_ok=True)

        # -- prompts.json ------------------------------------------------
        prompts_path = base / "prompts.json"
        prompts: dict[str, Any] = {}
        if prompts_path.exists():
            try:
                prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read existing prompts.json: {}", exc)

        prompts[version] = {
            "system_prompt": self.best_config.system_prompt,
            "user_template": self.best_config.user_template,
            "metadata": {
                "experiment_id": self.experiment_id,
                "baseline_score": self.baseline_score,
                "best_score": self.best_score,
                "absolute_improvement": self.absolute_improvement,
                "created_at": datetime.now(UTC).isoformat(),
            },
        }
        prompts_path.write_text(
            json.dumps(prompts, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Wrote prompt version '{}' to {}",
            version,
            prompts_path.resolve(),
        )

        # -- runtime_config.json -----------------------------------------
        config_path = base / "runtime_config.json"
        config_path.write_text(
            json.dumps(self.best_config.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote runtime config to {}", config_path.resolve())

        return version

    def save(self, directory: str | Path) -> Path:
        """Save the full result as JSON to *directory*.

        This is a convenience method that writes the entire
        :meth:`to_dict` output as ``fit_result.json``.  Use
        :meth:`apply` to additionally write ``prompts.json`` and
        ``runtime_config.json`` for the agent.

        Args:
            directory: Output directory (created if needed).

        Returns:
            Path to the written JSON file.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "fit_result.json"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved fit result to {}", json_path.resolve())
        return json_path
