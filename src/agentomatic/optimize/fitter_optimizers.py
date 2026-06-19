"""Fitter-specific optimizers for the PromptFitter pipeline.

Each optimizer implements a distinct search strategy and returns a list of
:class:`~agentomatic.optimize.config.PromptCandidate` objects ready for
evaluation.  Five concrete strategies are provided:

- **RewriteOptimizer** — LLM-driven full-prompt rewrite from failure analysis.
- **FewShotBootstrapOptimizer** — score-weighted random subsets of few-shot
  examples with diversity scoring.
- **MIPROLikeOptimizer** — multi-perspective instruction generation combined
  with few-shot and parameter sampling (inspired by MIPRO).
- **GEPALikeOptimizer** — feedback-guided prompt mutations targeting
  specific failure categories (inspired by GEPA).
- **ParamSearchOptimizer** — pure parameter-grid search without LLM calls.

All ``propose()`` methods are **async** (they may call LLMs internally).

Usage::

    from agentomatic.optimize.fitter_optimizers import resolve_fitter_optimizer

    optimizer = resolve_fitter_optimizer("rewrite", model="ollama/qwen2.5:7b")
    candidates = await optimizer.propose(
        current_config=baseline_config,
        eval_results=results,
        dataset_sample=samples,
        search_space=space,
        iteration=0,
    )

    # Or instantiate directly:
    from agentomatic.optimize.fitter_optimizers import MIPROLikeOptimizer

    mipro = MIPROLikeOptimizer(model="openai/gpt-4o-mini")
    candidates = await mipro.propose(...)
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.optimize.config import PromptCandidate, PromptRuntimeConfig
from agentomatic.optimize.llm_caller import LLMCaller
from agentomatic.optimize.search_space import PromptSearchSpace

if TYPE_CHECKING:
    from agentomatic.optimize.context import OptimizationContext

# =====================================================================
# Perspectives used by MIPRO-like instruction generation
# =====================================================================

_MIPRO_PERSPECTIVES: list[str] = [
    "Focus on making instructions more specific and actionable.",
    "Focus on adding output format constraints and structure.",
    "Focus on improving tone, clarity, and conciseness.",
    "Focus on adding edge case handling and error recovery.",
    "Focus on adding domain-specific knowledge and expertise.",
    "Focus on reducing ambiguity and adding explicit examples.",
    "Focus on improving step-by-step reasoning instructions.",
]

# =====================================================================
# ABC
# =====================================================================


@dataclass(slots=True)
class BaseFitterOptimizer(ABC):
    """Abstract base class for all fitter optimizers.

    Every concrete subclass must implement :meth:`propose`, which takes
    the current-best configuration and evaluation context and returns a
    list of :class:`PromptCandidate` objects for the fitter to evaluate.

    Attributes:
        name: Short identifier for the strategy (used in logging and
              candidate ``source`` fields).

    Examples::

        class MyOptimizer(BaseFitterOptimizer):
            name: str = "my_opt"

            async def propose(self, current_config, eval_results,
                              dataset_sample, search_space,
                              iteration=0, context=None):
                return [
                    PromptCandidate(
                        name=f"my_{iteration:03d}",
                        config=PromptRuntimeConfig(
                            system_prompt="Improved prompt",
                        ),
                        source=self.name,
                        mutation_notes="Added clarity.",
                    ),
                ]
    """

    name: str = "base"

    @abstractmethod
    async def propose(
        self,
        current_config: PromptRuntimeConfig,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: PromptSearchSpace,
        iteration: int = 0,
        context: OptimizationContext | None = None,
    ) -> list[PromptCandidate]:
        """Generate candidate configurations from evaluation context.

        Parameters
        ----------
        current_config:
            The current best configuration to improve upon.
        eval_results:
            Per-sample evaluation results from the most recent round.
            Each dict typically contains ``"query"``, ``"response"``,
            ``"expected"``, ``"score"``, and optionally ``"feedback"``,
            ``"reason"``, ``"retrieval_context"``, ``"tool_calls"``,
            ``"reasoning"``.
        dataset_sample:
            A sample of the evaluation dataset (raw examples).
        search_space:
            The parameter search space governing which knobs may be
            changed.
        iteration:
            Current iteration number in the outer optimisation loop.
        context:
            Rich :class:`OptimizationContext` with score history,
            failure clusters, dimensional scores, and pipeline
            metadata.  ``None`` for backward compatibility.

        Returns
        -------
        list[PromptCandidate]
            One or more candidates for the fitter to evaluate.
        """
        ...


# =====================================================================
# 1. RewriteOptimizer
# =====================================================================


@dataclass(slots=True)
class RewriteOptimizer(BaseFitterOptimizer):
    """LLM-driven full-prompt rewrite guided by failure analysis.

    Analyses the lowest-scoring evaluation results, identifies failure
    patterns, and asks an LLM to produce an improved system prompt that
    addresses those weaknesses while preserving strengths.

    Attributes:
        model:        LLM model identifier (``"provider/model_name"``).
        max_failures: Maximum number of low-scoring results to include
                      in the failure analysis sent to the LLM.

    Examples::

        opt = RewriteOptimizer(model="ollama/qwen2.5:7b", max_failures=5)
        candidates = await opt.propose(
            current_config=cfg, eval_results=results,
            dataset_sample=samples, search_space=space, iteration=1,
        )
        assert len(candidates) == 1
        assert candidates[0].source == "rewrite"
    """

    name: str = "rewrite"
    model: str = "ollama/qwen2.5:7b"
    max_failures: int = 5

    async def propose(
        self,
        current_config: PromptRuntimeConfig,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: PromptSearchSpace,
        iteration: int = 0,
        context: OptimizationContext | None = None,
    ) -> list[PromptCandidate]:
        """Rewrite the system prompt based on failure/success analysis.

        Workflow:
        1. Sort results by score (ascending) to surface failures.
        2. Build a detailed failure analysis string with pipeline context.
        3. Identify success patterns from top-scoring results.
        4. Include score history and dimensional trends (if available).
        5. Ask the LLM to reason step-by-step then produce an improved
           prompt.
        6. Return a single ``PromptCandidate``.

        Returns
        -------
        list[PromptCandidate]
            Always exactly one candidate with the rewritten system prompt.
        """
        logger.info(
            "RewriteOptimizer: analysing {} results for iteration {}",
            len(eval_results),
            iteration,
        )

        # -- sort by score to find failures and successes ----------------
        scored = sorted(eval_results, key=lambda r: r.get("score", 0.0))
        failures = scored[: self.max_failures]
        successes = scored[-max(3, self.max_failures) :]

        # -- build failure analysis with pipeline context ----------------
        failure_lines: list[str] = []
        for idx, fail in enumerate(failures, 1):
            failure_lines.append(f"### Failure {idx}")
            failure_lines.append(f"- Query: {fail.get('query', 'N/A')}")
            failure_lines.append(f"- Response: {fail.get('response', 'N/A')[:300]}")
            failure_lines.append(f"- Expected: {str(fail.get('expected', 'N/A'))[:300]}")
            failure_lines.append(f"- Score: {fail.get('score', 0.0):.3f}")
            issues = fail.get("feedback") or fail.get("reason") or fail.get("details", "")
            if issues:
                failure_lines.append(f"- Issues: {issues}")
            # Dimensional scores
            dims = fail.get("dimensions", {})
            if dims:
                dim_str = ", ".join(f"{k}={v:.3f}" for k, v in dims.items())
                failure_lines.append(f"- Dimensions: {dim_str}")
            # Retrieval context
            ret_ctx = fail.get("retrieval_context", [])
            if ret_ctx:
                docs = "; ".join(str(d)[:100] for d in ret_ctx[:3])
                failure_lines.append(f"- Retrieved docs: {docs}")
            # Tool calls
            tool_calls = fail.get("tool_calls", [])
            if tool_calls:
                tools = ", ".join(str(t.get("name", t)) for t in tool_calls[:3])
                failure_lines.append(f"- Tool calls: {tools}")
            # Reasoning
            reasoning = fail.get("reasoning", "")
            if reasoning:
                failure_lines.append(f"- Agent reasoning: {str(reasoning)[:200]}")
            failure_lines.append("")
        failure_analysis = "\n".join(failure_lines) if failure_lines else "No failures identified."

        # -- build success patterns --------------------------------------
        success_lines: list[str] = []
        for idx, suc in enumerate(successes, 1):
            success_lines.append(f"### Success {idx}")
            success_lines.append(f"- Query: {suc.get('query', 'N/A')}")
            success_lines.append(f"- Score: {suc.get('score', 0.0):.3f}")
            success_lines.append(f"- Response snippet: {suc.get('response', '')[:200]}")
            success_lines.append("")
        success_patterns = "\n".join(success_lines) if success_lines else "No strong successes."

        # -- build context sections (score history, dims, clusters) ------
        context_sections = ""
        if context is not None:
            parts: list[str] = []
            # Score history / trends
            history = context.format_score_history(max_rounds=5)
            if history and history != "No previous rounds.":
                parts.append(f"## Score History (recent rounds)\n{history}")
                parts.append(f"Trend: {context.format_score_sparkline()}")
            # Dimensional breakdown
            dim_table = context.format_dimension_table()
            if dim_table and "No per-dimension" not in dim_table:
                parts.append(f"## Per-Dimension Scores\n{dim_table}")
            # Failure clusters
            clusters = context.format_failure_clusters()
            if clusters and "No failure clusters" not in clusters:
                parts.append(f"## Failure Clusters\n{clusters}")
            if parts:
                context_sections = "\n\n".join(parts) + "\n\n"

        # -- build the rewrite prompt with chain-of-thought --------------
        prompt = (
            "You are an expert prompt engineer. Your task is to REWRITE a "
            "system prompt to improve its performance.\n\n"
            f"## Current System Prompt\n```\n"
            f"{current_config.system_prompt}\n```\n\n"
        )
        if context_sections:
            prompt += context_sections
        prompt += (
            f"## Failure Analysis (lowest-scoring results)\n"
            f"{failure_analysis}\n\n"
            f"## Success Patterns (highest-scoring results)\n"
            f"{success_patterns}\n\n"
            "## Instructions\n"
            "First, THINK step-by-step in a <thinking> block:\n"
            "1. What are the ROOT CAUSES of each failure type?\n"
            "2. What patterns do the successful cases share?\n"
            "3. What specific instructions are missing or "
            "ambiguous?\n"
            "4. How do the score trends inform what to change?\n"
            "5. What concrete changes will address each failure "
            "while preserving successes?\n\n"
            "Then write the IMPROVED system prompt that:\n"
            "- Addresses the root causes of each failure type\n"
            "- Preserves the patterns that lead to high scores\n"
            "- Adds explicit instructions for edge cases\n"
            "- Is clear, specific, and actionable\n"
            "- Keeps the core task/role the same\n\n"
            "Reply with your <thinking> analysis followed by the "
            "new system prompt after a '---' separator.\n"
        )

        raw_response = await LLMCaller.call(
            self.model,
            prompt,
            temperature=0.7,
            max_tokens=3000,
        )

        # Extract the prompt after the separator (or use full response)
        new_prompt = raw_response.strip()
        if "---" in new_prompt:
            parts_split = new_prompt.split("---", 1)
            new_prompt = parts_split[-1].strip()
        # Strip any remaining <thinking> tags
        if "<thinking>" in new_prompt:
            # Take content after closing tag
            if "</thinking>" in new_prompt:
                new_prompt = new_prompt.split("</thinking>", 1)[-1].strip()
        # Strip markdown fences
        if new_prompt.startswith("```"):
            lines = new_prompt.split("\n")
            end = -1 if lines[-1].strip() == "```" else len(lines)
            new_prompt = "\n".join(lines[1:end]).strip()

        if not new_prompt.strip():
            logger.warning("RewriteOptimizer: LLM returned empty rewrite — keeping current config")
            new_prompt = current_config.system_prompt

        # -- determine mutation notes ------------------------------------
        mutation_notes = (
            f"Full prompt rewrite at iteration {iteration}. "
            f"Analysed {len(failures)} failures (avg score "
            f"{sum(f.get('score', 0) for f in failures) / max(len(failures), 1):.3f}) "
            f"and {len(successes)} successes."
        )
        if context is not None:
            mutation_notes += (
                f" Context: {len(context.score_history)} rounds "
                f"history, baseline={context.baseline_score:.3f}, "
                f"current={context.current_score:.3f}."
            )

        candidate = PromptCandidate(
            name=f"rewrite_{iteration:03d}",
            config=PromptRuntimeConfig(
                system_prompt=new_prompt,
                user_template=current_config.user_template,
                few_shot_examples=list(current_config.few_shot_examples),
                output_contract=current_config.output_contract,
                model_params=dict(current_config.model_params),
                rag_params=dict(current_config.rag_params),
                tool_params=dict(current_config.tool_params),
            ),
            source="rewrite",
            mutation_notes=mutation_notes,
        )
        logger.debug(
            "RewriteOptimizer: produced candidate '{}'",
            candidate.name,
        )
        return [candidate]


# =====================================================================
# 2. FewShotBootstrapOptimizer
# =====================================================================


@dataclass(slots=True)
class FewShotBootstrapOptimizer(BaseFitterOptimizer):
    """Score-weighted few-shot example selection with diversity scoring.

    Generates candidate configurations by sampling subsets of high-scoring
    evaluation results to use as few-shot examples.  Candidates are ranked
    by a combined quality metric (70% average score, 30% diversity).

    Attributes:
        n_candidates: Number of random subsets to sample initially.
        k_examples:   Number of few-shot examples per candidate.

    Examples::

        opt = FewShotBootstrapOptimizer(n_candidates=8, k_examples=3)
        candidates = await opt.propose(
            current_config=cfg, eval_results=results,
            dataset_sample=samples, search_space=space,
        )
        assert len(candidates) <= 3  # returns top 3
        for c in candidates:
            assert c.config.few_shot_examples  # populated
    """

    name: str = "few_shot_bootstrap"
    n_candidates: int = 5
    k_examples: int = 4

    async def propose(
        self,
        current_config: PromptRuntimeConfig,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: PromptSearchSpace,
        iteration: int = 0,
        context: OptimizationContext | None = None,
    ) -> list[PromptCandidate]:
        """Generate few-shot candidates from scored evaluation results.

        Workflow:
        1. Sort results by score (highest first).
        2. Sample *n_candidates* subsets of *k_examples* results, weighted
           by score² (higher scores are more likely to be selected).
        3. Score each subset by combined quality = avg_score × 0.7 + diversity × 0.3.
        4. Return the top 3 subsets as ``PromptCandidate`` objects.

        Returns
        -------
        list[PromptCandidate]
            Up to 3 candidates, each with ``config.few_shot_examples``
            populated from high-scoring evaluation results.
        """
        logger.info(
            "FewShotBootstrapOptimizer: sampling from {} results (k={})",
            len(eval_results),
            self.k_examples,
        )

        # -- sort by score descending ------------------------------------
        scored = sorted(eval_results, key=lambda r: r.get("score", 0.0), reverse=True)

        # Need at least k_examples scored results with query+response
        usable = [
            r for r in scored if r.get("query") and r.get("response") and r.get("score", 0.0) > 0
        ]
        if len(usable) < self.k_examples:
            logger.warning(
                "FewShotBootstrapOptimizer: only {} usable results, need {} — skipping",
                len(usable),
                self.k_examples,
            )
            return []

        # -- compute weights (score²) ------------------------------------
        weights = [r.get("score", 0.0) ** 2 for r in usable]
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0] * len(usable)
            total_weight = float(len(usable))
        normalised_weights = [w / total_weight for w in weights]

        # -- sample n_candidates subsets ---------------------------------
        subsets: list[list[dict[str, Any]]] = []
        for _ in range(self.n_candidates):
            indices = _weighted_sample_without_replacement(
                population=list(range(len(usable))),
                weights=normalised_weights,
                k=self.k_examples,
            )
            subset = [usable[i] for i in indices]
            subsets.append(subset)

        # -- score each subset -------------------------------------------
        ranked: list[tuple[float, int, list[dict[str, Any]]]] = []
        for idx, subset in enumerate(subsets):
            avg_score = sum(r.get("score", 0.0) for r in subset) / len(subset)
            diversity = _compute_diversity(subset)
            combined = avg_score * 0.7 + diversity * 0.3
            ranked.append((combined, idx, subset))

        ranked.sort(key=lambda t: t[0], reverse=True)
        top_subsets = ranked[:3]

        # -- build candidates --------------------------------------------
        candidates: list[PromptCandidate] = []
        for rank, (quality, subset_idx, subset) in enumerate(top_subsets):
            few_shot = [
                {"query": r.get("query", ""), "response": r.get("response", "")} for r in subset
            ]
            avg_score = sum(r.get("score", 0.0) for r in subset) / len(subset)
            diversity = _compute_diversity(subset)

            candidate = PromptCandidate(
                name=f"fewshot_{iteration:03d}_{rank}",
                config=PromptRuntimeConfig(
                    system_prompt=current_config.system_prompt,
                    user_template=current_config.user_template,
                    few_shot_examples=few_shot,
                    output_contract=current_config.output_contract,
                    model_params=dict(current_config.model_params),
                    rag_params=dict(current_config.rag_params),
                    tool_params=dict(current_config.tool_params),
                ),
                source="few_shot_bootstrap",
                mutation_notes=(
                    f"Few-shot bootstrap (rank {rank}): "
                    f"{self.k_examples} examples, avg_score={avg_score:.3f}, "
                    f"diversity={diversity:.3f}, combined={quality:.3f}"
                ),
            )
            candidates.append(candidate)

        logger.debug(
            "FewShotBootstrapOptimizer: produced {} candidates",
            len(candidates),
        )
        return candidates


# =====================================================================
# 3. MIPROLikeOptimizer
# =====================================================================


@dataclass(slots=True)
class MIPROLikeOptimizer(BaseFitterOptimizer):
    """Multi-perspective instruction + few-shot + param optimisation.

    Inspired by the MIPRO (Multi-prompt Instruction Proposal) approach,
    this optimizer generates instruction variants from 7 distinct prompt-
    engineering perspectives, pairs them with few-shot bootstraps, and
    optionally explores model-parameter combinations.

    The final candidate list is the cross-product of top instructions ×
    top few-shot subsets (capped at 15 candidates).

    Attributes:
        model:                    LLM for instruction generation.
        n_instruction_candidates: Number of instruction variants to gen.
        n_few_shot_candidates:    Number of few-shot subsets to sample.
        fuse_top_k:               If > 0, fuse the top-K instructions
                                  via LLM into a single merged variant.

    Examples::

        opt = MIPROLikeOptimizer(
            model="ollama/qwen2.5:7b",
            n_instruction_candidates=5,
            n_few_shot_candidates=3,
        )
        candidates = await opt.propose(
            current_config=cfg, eval_results=results,
            dataset_sample=samples, search_space=space,
        )
        assert all(c.source == "mipro_like" for c in candidates)
    """

    name: str = "mipro_like"
    model: str = "ollama/qwen2.5:7b"
    n_instruction_candidates: int = 5
    n_few_shot_candidates: int = 3
    fuse_top_k: int = 2

    async def propose(
        self,
        current_config: PromptRuntimeConfig,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: PromptSearchSpace,
        iteration: int = 0,
        context: OptimizationContext | None = None,
    ) -> list[PromptCandidate]:
        """Generate candidates combining instructions, few-shot, and params.

        Workflow:
        1. Generate *n_instruction_candidates* instruction variants in
           parallel, each from a different MIPRO perspective.
        2. Generate *n_few_shot_candidates* few-shot subsets using the
           :class:`FewShotBootstrapOptimizer` logic.
        3. Optionally sample model-parameter combinations from the
           search space.
        4. Build the cross-product of instructions × few-shot subsets
           (capped at 15 candidates).
        5. Optionally fuse the top-K instructions into one merged variant.

        Returns
        -------
        list[PromptCandidate]
            Up to 15 combined candidates.
        """
        logger.info(
            "MIPROLikeOptimizer: iter={}, generating {} instructions × {} few-shot",
            iteration,
            self.n_instruction_candidates,
            self.n_few_shot_candidates,
        )

        # -- 1. Build failure summary for the LLM -----------------------
        failure_summary = _build_failure_summary(eval_results, max_items=5)

        # -- 2. Generate instruction variants in parallel ----------------
        instruction_tasks = [
            self._generate_instruction_variant(
                current_prompt=current_config.system_prompt,
                failure_summary=failure_summary,
                samples=dataset_sample,
                variant_idx=i,
                iteration=iteration,
            )
            for i in range(self.n_instruction_candidates)
        ]
        instruction_results = await asyncio.gather(*instruction_tasks, return_exceptions=True)

        instructions: list[str] = []
        for i, result in enumerate(instruction_results):
            if isinstance(result, Exception):
                logger.warning("MIPROLikeOptimizer: instruction variant {} failed: {}", i, result)
                continue
            text = str(result).strip()
            if text:
                instructions.append(text)
        if not instructions:
            logger.warning("MIPROLikeOptimizer: all instruction variants failed — using baseline")
            instructions = [current_config.system_prompt]

        logger.debug("MIPROLikeOptimizer: got {} valid instructions", len(instructions))

        # -- 3. Generate few-shot subsets --------------------------------
        few_shot_subsets: list[list[dict[str, str]]] = []
        bootstrap = FewShotBootstrapOptimizer(
            n_candidates=self.n_few_shot_candidates * 2,
            k_examples=min(4, len(eval_results)),
        )
        fs_candidates = await bootstrap.propose(
            current_config=current_config,
            eval_results=eval_results,
            dataset_sample=dataset_sample,
            search_space=search_space,
            iteration=iteration,
        )
        for fc in fs_candidates[: self.n_few_shot_candidates]:
            few_shot_subsets.append(fc.config.few_shot_examples)

        # Always include the empty few-shot option
        if not few_shot_subsets:
            few_shot_subsets.append([])

        # -- 4. Optionally sample model params ---------------------------
        param_combos: list[dict[str, Any]] = [{}]
        if search_space.optimize_model_params:
            param_combos = search_space.sample_params(5, "model")
            if not param_combos:
                param_combos = [{}]

        # -- 5. Fuse top-K instructions if requested ---------------------
        if self.fuse_top_k >= 2 and len(instructions) >= 2:
            fused = await self._fuse_instructions(
                instructions[: self.fuse_top_k],
                current_config.system_prompt,
            )
            if fused:
                instructions.insert(0, fused)

        # -- 6. Cross-product (capped at 15) -----------------------------
        candidates: list[PromptCandidate] = []
        candidate_idx = 0
        for instr in instructions:
            for fs_set in few_shot_subsets:
                if candidate_idx >= 15:
                    break

                # Pick a param combo (cycle through available ones)
                params = param_combos[candidate_idx % len(param_combos)]
                merged_params = {**current_config.model_params, **params}

                candidate = PromptCandidate(
                    name=f"mipro_{iteration:03d}_{candidate_idx:02d}",
                    config=PromptRuntimeConfig(
                        system_prompt=instr,
                        user_template=current_config.user_template,
                        few_shot_examples=list(fs_set),
                        output_contract=current_config.output_contract,
                        model_params=merged_params,
                        rag_params=dict(current_config.rag_params),
                        tool_params=dict(current_config.tool_params),
                    ),
                    source="mipro_like",
                    mutation_notes=(
                        f"MIPRO candidate {candidate_idx}: "
                        f"instruction variant (len={len(instr)}), "
                        f"{len(fs_set)} few-shot examples, "
                        f"params={params or 'baseline'}"
                    ),
                )
                candidates.append(candidate)
                candidate_idx += 1

            if candidate_idx >= 15:
                break

        logger.info("MIPROLikeOptimizer: produced {} candidates", len(candidates))
        return candidates

    # -- private helpers -------------------------------------------------

    async def _generate_instruction_variant(
        self,
        current_prompt: str,
        failure_summary: str,
        samples: list[dict[str, Any]],
        variant_idx: int,
        iteration: int,
    ) -> str:
        """Generate a single instruction variant from a specific perspective.

        Each variant is generated with a different prompt-engineering
        perspective from ``_MIPRO_PERSPECTIVES`` to maximise diversity.

        Parameters
        ----------
        current_prompt:
            The system prompt to improve.
        failure_summary:
            Formatted summary of evaluation failures.
        samples:
            Example dataset entries for context.
        variant_idx:
            Index selecting the perspective.
        iteration:
            Outer loop iteration number.

        Returns
        -------
        str
            The generated instruction text.
        """
        perspective = _MIPRO_PERSPECTIVES[variant_idx % len(_MIPRO_PERSPECTIVES)]

        # Include a few dataset samples for grounding
        sample_str = ""
        if samples:
            shown = samples[:3]
            sample_lines = []
            for s in shown:
                query = s.get("query", s.get("input", "N/A"))
                expected = s.get("expected", s.get("output", "N/A"))
                sample_lines.append(f"  Q: {str(query)[:200]}")
                sample_lines.append(f"  A: {str(expected)[:200]}")
            sample_str = "\n".join(sample_lines)

        prompt = (
            "You are a prompt engineering expert. Generate a VARIATION of "
            "the system prompt below.\n\n"
            f"## Perspective\n{perspective}\n\n"
            f"## Current Prompt\n```\n{current_prompt}\n```\n\n"
            f"## Known Failure Patterns\n{failure_summary}\n\n"
        )
        if sample_str:
            prompt += f"## Example Queries\n{sample_str}\n\n"
        prompt += (
            "## Rules\n"
            "- Create a DIFFERENT approach to the same task\n"
            "- Address the failure patterns identified above\n"
            "- Keep the core purpose and task intact\n"
            "- Be creative but practical\n"
            f"- This is variant {variant_idx} of iteration {iteration}\n\n"
            "Reply with ONLY the new system prompt text.\n"
        )

        return await LLMCaller.call(
            self.model,
            prompt,
            temperature=0.8,
            max_tokens=2000,
        )

    async def _fuse_instructions(
        self,
        instructions: list[str],
        original: str,
    ) -> str:
        """Fuse multiple instruction candidates into a single merged variant.

        Asks the LLM to combine the best elements of several candidate
        instructions into one coherent, improved prompt.

        Parameters
        ----------
        instructions:
            The top-K instruction variants to merge.
        original:
            The original baseline prompt for reference.

        Returns
        -------
        str
            The fused instruction text, or empty string on failure.
        """
        numbered = "\n\n".join(
            f"### Variant {i + 1}\n```\n{instr}\n```" for i, instr in enumerate(instructions)
        )
        prompt = (
            "You are a prompt engineering expert. Below are several "
            "candidate system prompts that each take a different approach "
            "to the same task.\n\n"
            f"## Original Prompt\n```\n{original}\n```\n\n"
            f"## Candidate Variants\n{numbered}\n\n"
            "## Task\n"
            "Merge the BEST elements of all variants into a single, "
            "coherent, improved system prompt. Combine complementary "
            "strengths and eliminate redundant or conflicting instructions.\n\n"
            "Reply with ONLY the fused system prompt text.\n"
        )

        result = await LLMCaller.call(
            self.model,
            prompt,
            temperature=0.5,
            max_tokens=2000,
        )
        return result.strip()


# =====================================================================
# 4. GEPALikeOptimizer
# =====================================================================


@dataclass(slots=True)
class GEPALikeOptimizer(BaseFitterOptimizer):
    """Feedback-guided prompt mutation targeting specific failure categories.

    Inspired by GEPA (Guided Evolution for Prompt Adaptation), this
    optimizer:
    1. Categorises evaluation feedback into distinct failure types.
    2. Generates targeted prompt mutations, each addressing a different
       failure category (e.g. completeness, factual grounding, format).

    Attributes:
        judge_model:   Model for feedback analysis / categorisation.
        rewrite_model: Model for generating prompt mutations.
        n_mutations:   Number of distinct mutations to produce.

    Examples::

        opt = GEPALikeOptimizer(
            judge_model="ollama/qwen2.5:7b",
            rewrite_model="ollama/qwen2.5:7b",
            n_mutations=3,
        )
        candidates = await opt.propose(
            current_config=cfg, eval_results=results,
            dataset_sample=samples, search_space=space,
        )
        for c in candidates:
            assert c.source == "gepa_like"
    """

    name: str = "gepa_like"
    judge_model: str = "ollama/qwen2.5:7b"
    rewrite_model: str = "ollama/qwen2.5:7b"
    n_mutations: int = 3

    # Default mutation aspects when fewer categories are found in feedback
    _DEFAULT_ASPECTS: list[str] = field(
        default_factory=lambda: [
            "completeness — ensure answers cover all parts of the question",
            "factual grounding — ensure answers are accurate and evidence-based",
            "format compliance — ensure answers follow the requested structure",
            "conciseness — remove unnecessary verbosity while keeping substance",
            "edge case handling — add instructions for unusual or ambiguous inputs",
        ]
    )

    async def propose(
        self,
        current_config: PromptRuntimeConfig,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: PromptSearchSpace,
        iteration: int = 0,
        context: OptimizationContext | None = None,
    ) -> list[PromptCandidate]:
        """Generate targeted prompt mutations from evaluation feedback.

        Workflow:
        1. Collect feedback strings from eval results (``feedback``,
           ``reason``, or ``details`` fields).
        2. Categorise feedback into failure types.
        3. Generate *n_mutations* prompt mutations, each targeting a
           different aspect.

        Returns
        -------
        list[PromptCandidate]
            One candidate per mutation, each with detailed ``mutation_notes``.
        """
        logger.info(
            "GEPALikeOptimizer: iter={}, generating {} mutations",
            iteration,
            self.n_mutations,
        )

        # -- 1. Collect feedback from results ----------------------------
        feedback_items: list[dict[str, Any]] = []
        for r in eval_results:
            fb = r.get("feedback") or r.get("reason") or r.get("details", "")
            if fb:
                feedback_items.append(
                    {
                        "query": r.get("query", "N/A"),
                        "score": r.get("score", 0.0),
                        "feedback": str(fb),
                        "response_snippet": str(r.get("response", ""))[:200],
                    }
                )

        # -- 2. Build feedback summary -----------------------------------
        feedback_summary = self._build_categorised_feedback(feedback_items)

        # -- 3. Select mutation aspects ----------------------------------
        aspects = self._select_mutation_aspects(feedback_items)

        # -- 4. Generate mutations in parallel ---------------------------
        mutation_tasks = [
            self._generate_mutation(
                current_prompt=current_config.system_prompt,
                feedback_summary=feedback_summary,
                aspect=aspects[i % len(aspects)],
                mutation_idx=i,
                iteration=iteration,
            )
            for i in range(self.n_mutations)
        ]
        mutation_results = await asyncio.gather(*mutation_tasks, return_exceptions=True)

        # -- 5. Build candidates -----------------------------------------
        candidates: list[PromptCandidate] = []
        for idx, result in enumerate(mutation_results):
            if isinstance(result, Exception):
                logger.warning("GEPALikeOptimizer: mutation {} failed: {}", idx, result)
                continue

            new_prompt = str(result).strip()
            if not new_prompt:
                logger.warning("GEPALikeOptimizer: mutation {} returned empty — skipping", idx)
                continue

            aspect = aspects[idx % len(aspects)]
            candidate = PromptCandidate(
                name=f"gepa_{iteration:03d}_{idx}",
                config=PromptRuntimeConfig(
                    system_prompt=new_prompt,
                    user_template=current_config.user_template,
                    few_shot_examples=list(current_config.few_shot_examples),
                    output_contract=current_config.output_contract,
                    model_params=dict(current_config.model_params),
                    rag_params=dict(current_config.rag_params),
                    tool_params=dict(current_config.tool_params),
                ),
                source="gepa_like",
                mutation_notes=(
                    f"GEPA mutation {idx} targeting: {aspect}. "
                    f"Based on {len(feedback_items)} feedback items."
                ),
            )
            candidates.append(candidate)

        logger.info("GEPALikeOptimizer: produced {} candidates", len(candidates))
        return candidates

    # -- private helpers -------------------------------------------------

    def _build_categorised_feedback(
        self,
        feedback_items: list[dict[str, Any]],
    ) -> str:
        """Build a categorised summary of evaluation feedback.

        Groups feedback by score ranges and identifies recurring patterns
        to give the LLM a structured view of what went wrong.

        Parameters
        ----------
        feedback_items:
            Extracted feedback dicts with ``query``, ``score``, ``feedback``
            and ``response_snippet`` keys.

        Returns
        -------
        str
            Formatted feedback summary for inclusion in the rewrite prompt.
        """
        if not feedback_items:
            return "No specific feedback available from evaluation results."

        # Group by score buckets
        critical: list[dict[str, Any]] = []  # score < 0.3
        moderate: list[dict[str, Any]] = []  # 0.3 <= score < 0.6
        minor: list[dict[str, Any]] = []  # 0.6 <= score < 0.8

        for item in feedback_items:
            score = item.get("score", 0.0)
            if score < 0.3:
                critical.append(item)
            elif score < 0.6:
                moderate.append(item)
            elif score < 0.8:
                minor.append(item)

        lines: list[str] = []
        if critical:
            lines.append(f"### Critical Failures ({len(critical)} items, score < 0.3)")
            for item in critical[:5]:
                lines.append(f"  - [{item['score']:.2f}] {item['feedback'][:200]}")
                lines.append(f"    Query: {item['query'][:100]}")
            lines.append("")

        if moderate:
            lines.append(f"### Moderate Issues ({len(moderate)} items, 0.3 ≤ score < 0.6)")
            for item in moderate[:5]:
                lines.append(f"  - [{item['score']:.2f}] {item['feedback'][:200]}")
            lines.append("")

        if minor:
            lines.append(f"### Minor Issues ({len(minor)} items, 0.6 ≤ score < 0.8)")
            for item in minor[:3]:
                lines.append(f"  - [{item['score']:.2f}] {item['feedback'][:150]}")
            lines.append("")

        return "\n".join(lines) if lines else "All results scored ≥ 0.8 — minor refinements only."

    def _select_mutation_aspects(
        self,
        feedback_items: list[dict[str, Any]],
    ) -> list[str]:
        """Choose which aspects each mutation should target.

        Tries to infer relevant aspects from feedback text.  Falls back
        to default aspects if not enough feedback is available.

        Parameters
        ----------
        feedback_items:
            Extracted feedback dicts.

        Returns
        -------
        list[str]
            At least ``n_mutations`` aspect descriptions.
        """
        # Try to detect categories from feedback keywords
        detected: list[str] = []
        keyword_map: dict[str, str] = {
            "incomplete": "completeness — ensure answers cover all parts of the question",
            "missing": "completeness — ensure answers cover all parts of the question",
            "inaccurate": "factual grounding — ensure answers are accurate and evidence-based",
            "wrong": "factual grounding — ensure answers are accurate and evidence-based",
            "hallucin": "factual grounding — reduce hallucination by requiring evidence",
            "format": "format compliance — ensure answers follow the requested structure",
            "structure": "format compliance — ensure answers follow the requested structure",
            "verbose": "conciseness — remove unnecessary verbosity while keeping substance",
            "long": "conciseness — remove unnecessary verbosity while keeping substance",
            "edge": "edge case handling — add instructions for unusual or ambiguous inputs",
            "ambig": "edge case handling — add instructions for unusual or ambiguous inputs",
        }

        all_feedback_text = " ".join(item.get("feedback", "").lower() for item in feedback_items)
        for keyword, aspect in keyword_map.items():
            if keyword in all_feedback_text and aspect not in detected:
                detected.append(aspect)

        # Pad with defaults if we don't have enough
        for default_aspect in self._DEFAULT_ASPECTS:
            if len(detected) >= self.n_mutations:
                break
            if default_aspect not in detected:
                detected.append(default_aspect)

        # Ensure we always have enough
        while len(detected) < self.n_mutations:
            detected.append(self._DEFAULT_ASPECTS[len(detected) % len(self._DEFAULT_ASPECTS)])

        return detected

    async def _generate_mutation(
        self,
        current_prompt: str,
        feedback_summary: str,
        aspect: str,
        mutation_idx: int,
        iteration: int,
    ) -> str:
        """Generate a single prompt mutation targeting a specific aspect.

        Parameters
        ----------
        current_prompt:
            The baseline system prompt to mutate.
        feedback_summary:
            Categorised feedback summary.
        aspect:
            The specific aspect this mutation should address.
        mutation_idx:
            Index of this mutation (for logging).
        iteration:
            Outer loop iteration number.

        Returns
        -------
        str
            The mutated system prompt text.
        """
        prompt = (
            "You are an expert prompt engineer performing targeted prompt "
            "mutation.\n\n"
            f"## Current System Prompt\n```\n{current_prompt}\n```\n\n"
            f"## Evaluation Feedback Summary\n{feedback_summary}\n\n"
            f"## Mutation Target\n"
            f"This mutation should specifically improve: **{aspect}**\n\n"
            "## Instructions\n"
            f"1. Analyse the current prompt for weaknesses related to: {aspect}\n"
            "2. Make TARGETED changes that specifically address this aspect\n"
            "3. Preserve all other instructions that are working well\n"
            "4. Add explicit guidance or constraints related to the target aspect\n"
            "5. The mutation should be a clear, focused improvement — not a "
            "complete rewrite\n\n"
            f"This is mutation {mutation_idx} of iteration {iteration}.\n\n"
            "Reply with ONLY the mutated system prompt text.\n"
        )

        return await LLMCaller.call(
            self.rewrite_model,
            prompt,
            temperature=0.7,
            max_tokens=2000,
        )


# =====================================================================
# 5. ParamSearchOptimizer
# =====================================================================


@dataclass(slots=True)
class ParamSearchOptimizer(BaseFitterOptimizer):
    """Pure parameter-grid search without LLM calls.

    Generates candidates by sampling from the parameter search space.
    System prompts and few-shot examples are inherited unchanged from the
    baseline — only model, RAG, and tool parameters are varied.

    This optimizer requires **no LLM calls** and is therefore very fast.

    Examples::

        from agentomatic.optimize.search_space import PromptSearchSpace

        space = PromptSearchSpace(
            optimize_model_params=True,
            model_param_space={"temperature": [0.0, 0.3, 0.7]},
        )
        opt = ParamSearchOptimizer()
        candidates = await opt.propose(
            baseline=cfg, eval_results=[],
            dataset_sample=[], search_space=space,
        )
        # Each candidate has a different temperature value
        temps = [c.config.model_params["temperature"] for c in candidates]
        assert len(set(temps)) == len(temps)
    """

    name: str = "param_search"

    async def propose(
        self,
        current_config: PromptRuntimeConfig,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: PromptSearchSpace,
        iteration: int = 0,
        context: OptimizationContext | None = None,
    ) -> list[PromptCandidate]:
        """Generate candidates by sampling parameter combinations.

        Workflow:
        1. Identify active parameter spaces (model, rag, tool).
        2. Sample up to 10 combinations from each active space.
        3. Create a ``PromptCandidate`` for each combination with the
           baseline prompt/few-shot unchanged.

        Returns
        -------
        list[PromptCandidate]
            One candidate per sampled parameter combination.
        """
        logger.info(
            "ParamSearchOptimizer: iter={}, active spaces={}",
            iteration,
            search_space.active_spaces(),
        )

        candidates: list[PromptCandidate] = []
        candidate_idx = 0

        # -- model params ------------------------------------------------
        if search_space.optimize_model_params:
            combos = search_space.sample_params(10, "model")
            for combo in combos:
                merged = {**current_config.model_params, **combo}
                changes = {
                    k: v for k, v in combo.items() if current_config.model_params.get(k) != v
                }
                if not changes:
                    continue

                candidate = PromptCandidate(
                    name=f"param_{iteration:03d}_m{candidate_idx:02d}",
                    config=PromptRuntimeConfig(
                        system_prompt=current_config.system_prompt,
                        user_template=current_config.user_template,
                        few_shot_examples=list(current_config.few_shot_examples),
                        output_contract=current_config.output_contract,
                        model_params=merged,
                        rag_params=dict(current_config.rag_params),
                        tool_params=dict(current_config.tool_params),
                    ),
                    source="param_search",
                    mutation_notes=(
                        "Model param change: " + ", ".join(f"{k}={v}" for k, v in changes.items())
                    ),
                )
                candidates.append(candidate)
                candidate_idx += 1

        # -- rag params --------------------------------------------------
        if search_space.optimize_rag_params:
            combos = search_space.sample_params(10, "rag")
            for combo in combos:
                merged = {**current_config.rag_params, **combo}
                changes = {k: v for k, v in combo.items() if current_config.rag_params.get(k) != v}
                if not changes:
                    continue

                candidate = PromptCandidate(
                    name=f"param_{iteration:03d}_r{candidate_idx:02d}",
                    config=PromptRuntimeConfig(
                        system_prompt=current_config.system_prompt,
                        user_template=current_config.user_template,
                        few_shot_examples=list(current_config.few_shot_examples),
                        output_contract=current_config.output_contract,
                        model_params=dict(current_config.model_params),
                        rag_params=merged,
                        tool_params=dict(current_config.tool_params),
                    ),
                    source="param_search",
                    mutation_notes=(
                        "RAG param change: " + ", ".join(f"{k}={v}" for k, v in changes.items())
                    ),
                )
                candidates.append(candidate)
                candidate_idx += 1

        # -- tool params -------------------------------------------------
        if search_space.optimize_tool_params:
            combos = search_space.sample_params(10, "tool")
            for combo in combos:
                merged = {**current_config.tool_params, **combo}
                changes = {
                    k: v for k, v in combo.items() if current_config.tool_params.get(k) != v
                }
                if not changes:
                    continue

                candidate = PromptCandidate(
                    name=f"param_{iteration:03d}_t{candidate_idx:02d}",
                    config=PromptRuntimeConfig(
                        system_prompt=current_config.system_prompt,
                        user_template=current_config.user_template,
                        few_shot_examples=list(current_config.few_shot_examples),
                        output_contract=current_config.output_contract,
                        model_params=dict(current_config.model_params),
                        rag_params=dict(current_config.rag_params),
                        tool_params=merged,
                    ),
                    source="param_search",
                    mutation_notes=(
                        "Tool param change: " + ", ".join(f"{k}={v}" for k, v in changes.items())
                    ),
                )
                candidates.append(candidate)
                candidate_idx += 1

        logger.info(
            "ParamSearchOptimizer: produced {} candidates across {} spaces",
            len(candidates),
            len(search_space.active_spaces()),
        )
        return candidates


# =====================================================================
# Module-level helpers
# =====================================================================


def _weighted_sample_without_replacement(
    population: list[int],
    weights: list[float],
    k: int,
) -> list[int]:
    """Sample *k* items from *population* without replacement, weighted.

    Uses sequential draws with weight renormalization after each pick.
    This is simpler than reservoir sampling and adequate for small *k*.

    Parameters
    ----------
    population:
        List of candidate indices.
    weights:
        Corresponding selection weights (must sum to ~1.0).
    k:
        Number of items to select.

    Returns
    -------
    list[int]
        Selected indices (order is randomised).

    Examples::

        indices = _weighted_sample_without_replacement(
            population=[0, 1, 2, 3, 4],
            weights=[0.4, 0.3, 0.15, 0.1, 0.05],
            k=3,
        )
        assert len(indices) == 3
        assert len(set(indices)) == 3
    """
    if k >= len(population):
        return list(population)

    selected: list[int] = []
    remaining = list(population)
    remaining_weights = list(weights)

    for _ in range(k):
        total = sum(remaining_weights)
        if total <= 0:
            # Fallback to uniform if all weights are zero
            idx_in_remaining = random.randrange(len(remaining))
        else:
            normalised = [w / total for w in remaining_weights]
            idx_in_remaining = random.choices(
                range(len(remaining)),
                weights=normalised,
                k=1,
            )[0]

        selected.append(remaining[idx_in_remaining])
        remaining.pop(idx_in_remaining)
        remaining_weights.pop(idx_in_remaining)

    return selected


def _compute_diversity(subset: list[dict[str, Any]]) -> float:
    """Compute diversity score for a subset of results.

    Diversity is measured as the ratio of unique query prefixes (first 50
    characters) to the total number of items.  A score of 1.0 means every
    item has a unique prefix; lower scores indicate redundancy.

    Parameters
    ----------
    subset:
        List of evaluation result dicts, each with a ``"query"`` key.

    Returns
    -------
    float
        Diversity score in [0.0, 1.0].

    Examples::

        items = [
            {"query": "What is Python?"},
            {"query": "What is Java?"},
            {"query": "How to sort a list?"},
        ]
        assert _compute_diversity(items) == 1.0  # all unique prefixes
    """
    if not subset:
        return 0.0
    prefixes = {str(r.get("query", ""))[:50] for r in subset}
    return len(prefixes) / len(subset)


def _build_failure_summary(
    eval_results: list[dict[str, Any]],
    max_items: int = 5,
) -> str:
    """Build a concise failure summary from evaluation results.

    Sorts results by score (ascending), selects the lowest-scoring items,
    and formats them into a readable string for inclusion in LLM prompts.

    Parameters
    ----------
    eval_results:
        Per-sample evaluation results with ``"score"`` keys.
    max_items:
        Maximum number of failure items to include.

    Returns
    -------
    str
        Formatted failure summary string.
    """
    if not eval_results:
        return "No evaluation results available."

    scored = sorted(eval_results, key=lambda r: r.get("score", 0.0))
    failures = scored[:max_items]

    lines: list[str] = []
    for idx, f in enumerate(failures, 1):
        score = f.get("score", 0.0)
        query = str(f.get("query", "N/A"))[:150]
        feedback = f.get("feedback") or f.get("reason") or f.get("details", "N/A")
        lines.append(f"{idx}. [score={score:.3f}] Q: {query}")
        if feedback and feedback != "N/A":
            lines.append(f"   Feedback: {str(feedback)[:200]}")

    return "\n".join(lines) if lines else "No failures identified."


# =====================================================================
# Factory
# =====================================================================


def resolve_fitter_optimizer(
    name: str | BaseFitterOptimizer,
    model: str = "ollama/qwen2.5:7b",
    rewrite_model: str | None = None,
    **kwargs: Any,
) -> BaseFitterOptimizer:
    """Resolve a fitter optimizer by name or pass through an existing instance.

    Maps human-friendly string names to concrete optimizer classes.  If
    *name* is already a :class:`BaseFitterOptimizer` instance it is
    returned unchanged.

    Parameters
    ----------
    name:
        Either a string identifier or an existing optimizer instance.

        Supported string identifiers:

        =========================  ================================
        String                     Optimizer class
        =========================  ================================
        ``"rewrite"``              :class:`RewriteOptimizer`
        ``"few_shot"``             :class:`FewShotBootstrapOptimizer`
        ``"few_shot_bootstrap"``   :class:`FewShotBootstrapOptimizer`
        ``"mipro_like"``           :class:`MIPROLikeOptimizer`
        ``"mipro"``                :class:`MIPROLikeOptimizer`
        ``"gepa_like"``            :class:`GEPALikeOptimizer`
        ``"gepa"``                 :class:`GEPALikeOptimizer`
        ``"param_search"``         :class:`ParamSearchOptimizer`
        =========================  ================================

    model:
        Default LLM model for optimizers that require one.
    rewrite_model:
        Separate model for rewrite calls (GEPALikeOptimizer).  Falls
        back to *model* if ``None``.
    **kwargs:
        Extra keyword arguments forwarded to the optimizer constructor.

    Returns
    -------
    BaseFitterOptimizer
        A ready-to-use optimizer instance.

    Raises
    ------
    ValueError
        If *name* is an unrecognised string.

    Examples::

        # By name
        opt = resolve_fitter_optimizer("rewrite")
        assert isinstance(opt, RewriteOptimizer)

        # With custom model
        opt = resolve_fitter_optimizer("mipro", model="openai/gpt-4o-mini")
        assert isinstance(opt, MIPROLikeOptimizer)

        # Pass-through existing instance
        existing = ParamSearchOptimizer()
        assert resolve_fitter_optimizer(existing) is existing
    """
    # -- pass-through for existing instances -----------------------------
    if isinstance(name, BaseFitterOptimizer):
        logger.debug("resolve_fitter_optimizer: got existing instance ({})", name.name)
        return name

    # -- string resolution -----------------------------------------------
    lookup: dict[str, type[BaseFitterOptimizer]] = {
        "rewrite": RewriteOptimizer,
        "few_shot": FewShotBootstrapOptimizer,
        "few_shot_bootstrap": FewShotBootstrapOptimizer,
        "mipro_like": MIPROLikeOptimizer,
        "mipro": MIPROLikeOptimizer,
        "gepa_like": GEPALikeOptimizer,
        "gepa": GEPALikeOptimizer,
        "param_search": ParamSearchOptimizer,
    }

    normalised = name.strip().lower()
    if normalised not in lookup:
        available = sorted(set(lookup.keys()))
        msg = f"Unknown fitter optimizer '{name}'. Available: {available}"
        raise ValueError(msg)

    cls = lookup[normalised]

    # -- build kwargs per optimizer type ---------------------------------
    if cls is RewriteOptimizer:
        return RewriteOptimizer(model=model, **kwargs)

    if cls is FewShotBootstrapOptimizer:
        return FewShotBootstrapOptimizer(**kwargs)

    if cls is MIPROLikeOptimizer:
        return MIPROLikeOptimizer(model=model, **kwargs)

    if cls is GEPALikeOptimizer:
        return GEPALikeOptimizer(
            judge_model=model,
            rewrite_model=rewrite_model or model,
            **kwargs,
        )

    if cls is ParamSearchOptimizer:
        return ParamSearchOptimizer(**kwargs)

    # Unreachable, but satisfies exhaustiveness checkers
    return cls(**kwargs)  # type: ignore[call-arg]
