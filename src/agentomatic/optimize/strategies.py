"""Optimization strategies — how to improve prompts iteratively.

Strategies:
- IterativeRewrite: LLM analyzes failures and rewrites the prompt
- FewShotBootstrap: Auto-selects best few-shot examples
- ChainOfThought: Adds/optimizes reasoning steps
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class IterationResult:
    """Result of a single optimization iteration."""

    iteration: int
    prompt: str
    avg_score: float
    per_metric_scores: dict[str, float] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    improvements: str = ""


class OptimizationStrategy(ABC):
    """Base class for prompt optimization strategies."""

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None: ...

    @abstractmethod
    async def step(
        self,
        current_prompt: str,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        """Produce an improved prompt based on evaluation results.

        Args:
            current_prompt: The current system prompt.
            eval_results: List of evaluation results from the last run.
            dataset_sample: Sample of dataset points for reference.
            iteration: Current iteration number.

        Returns:
            An improved prompt string.
        """
        ...


class IterativeRewrite(OptimizationStrategy):
    """LLM rewrites the prompt based on failure analysis.

    Process:
    1. Analyze low-scoring responses
    2. Identify failure patterns
    3. Ask LLM to rewrite prompt to address failures
    4. Preserve successful behaviors
    """

    name = "iterative_rewrite"

    def __init__(self, model: str = "ollama/mistral:7b", max_failures: int = 5):
        self.model = model
        self.max_failures = max_failures

    async def step(
        self,
        current_prompt: str,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        # Sort by score ascending (worst first)
        failures = sorted(eval_results, key=lambda r: r.get("avg_score", 0))[: self.max_failures]
        successes = sorted(eval_results, key=lambda r: r.get("avg_score", 0), reverse=True)[:3]

        failure_analysis = "\n".join(
            f"- Query: {f.get('query', '')}\n"
            f"  Response: {f.get('response', '')[:200]}\n"
            f"  Expected: {f.get('expected', 'N/A')}\n"
            f"  Score: {f.get('avg_score', 0):.2f}\n"
            f"  Issues: {'; '.join(r.get('reason', '') for r in f.get('details', []))}"
            for f in failures
        )

        success_summary = "\n".join(
            f"- Query: {s.get('query', '')} → Score: {s.get('avg_score', 0):.2f}"
            for s in successes
        )

        rewrite_prompt = (
            f"You are a prompt engineering expert. Your task is to improve a system prompt "
            f"for an AI agent based on evaluation feedback.\n\n"
            f"## Current Prompt (Iteration {iteration})\n"
            f"```\n{current_prompt}\n```\n\n"
            f"## Failure Analysis (Low-Scoring Responses)\n"
            f"{failure_analysis}\n\n"
            f"## Success Analysis (High-Scoring Responses)\n"
            f"{success_summary}\n\n"
            f"## Instructions\n"
            f"1. Analyze the failure patterns above\n"
            f"2. Identify what the prompt is missing or doing wrong\n"
            f"3. Write an IMPROVED system prompt that addresses the failures\n"
            f"4. PRESERVE behaviors that led to successful responses\n"
            f"5. Be specific and actionable in your instructions\n\n"
            f"Reply with ONLY the improved system prompt text, nothing else.\n"
        )

        return await self._call_llm(rewrite_prompt)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for prompt rewriting."""
        import httpx

        model_name = (
            self.model.replace("ollama/", "") if self.model.startswith("ollama/") else self.model
        )
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    return str(resp.json().get("response", "")).strip()
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}")

        # Fallback: return current prompt with a hint
        return prompt + "\n\n(Optimization LLM unavailable — manual revision needed)"


class FewShotBootstrap(OptimizationStrategy):
    """Automatically select best few-shot examples from the dataset.

    Process:
    1. Run all examples through the agent
    2. Score each response
    3. Select top-K as few-shot demonstrations
    4. Inject into prompt template
    """

    name = "few_shot_bootstrap"

    def __init__(self, n_examples: int = 5, model: str = "ollama/mistral:7b"):
        self.n_examples = n_examples
        self.model = model

    async def step(
        self,
        current_prompt: str,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        # Select top-scoring examples
        scored = sorted(eval_results, key=lambda r: r.get("avg_score", 0), reverse=True)
        top_examples = scored[: self.n_examples]

        # Build few-shot block
        examples_block = "\n\n".join(
            f"**Example {i + 1}:**\n"
            f"User: {ex.get('query', '')}\n"
            f"Assistant: {ex.get('response', '')}"
            for i, ex in enumerate(top_examples)
        )

        # Remove any existing examples section
        prompt_lines = current_prompt.split("\n")
        clean_lines = []
        in_examples = False
        for line in prompt_lines:
            if line.strip().startswith("## Examples") or line.strip().startswith("**Example"):
                in_examples = True
                continue
            if in_examples and line.strip() == "":
                in_examples = False
                continue
            if not in_examples:
                clean_lines.append(line)

        base_prompt = "\n".join(clean_lines).strip()

        return f"{base_prompt}\n\n## Examples\n\n{examples_block}"


class ChainOfThought(OptimizationStrategy):
    """Add or optimize chain-of-thought reasoning instructions.

    Process:
    1. Analyze responses for reasoning quality
    2. Generate step-by-step reasoning template
    3. Inject CoT instructions into system prompt
    """

    name = "chain_of_thought"

    def __init__(self, model: str = "ollama/mistral:7b"):
        self.model = model

    async def step(
        self,
        current_prompt: str,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        # Analyze which queries need step-by-step reasoning
        failures = [r for r in eval_results if r.get("avg_score", 0) < 0.7]

        query_types = set()
        for f in failures[:5]:
            query = f.get("query", "")
            if "?" in query:
                query_types.add("questions")
            if any(w in query.lower() for w in ["compare", "difference", "vs"]):
                query_types.add("comparisons")
            if any(w in query.lower() for w in ["how", "step", "process"]):
                query_types.add("procedures")
            if any(w in query.lower() for w in ["why", "explain", "reason"]):
                query_types.add("explanations")

        cot_instruction = (
            "\n\n## Reasoning Instructions\n\n"
            "When answering, follow these steps:\n"
            "1. **Understand** — Identify what the user is asking\n"
            "2. **Recall** — Gather relevant information from your knowledge\n"
            "3. **Reason** — Think through the answer step by step\n"
            "4. **Respond** — Provide a clear, structured answer\n"
            "5. **Verify** — Double-check your response for accuracy\n"
        )

        if "comparisons" in query_types:
            cot_instruction += "\nFor comparisons, explicitly list similarities and differences.\n"
        if "procedures" in query_types:
            cot_instruction += "\nFor procedures, provide numbered step-by-step instructions.\n"
        if "explanations" in query_types:
            cot_instruction += (
                "\nFor explanations, start with a summary then provide supporting details.\n"
            )

        # Remove existing reasoning section
        if "## Reasoning Instructions" in current_prompt:
            parts = current_prompt.split("## Reasoning Instructions")
            base = parts[0].strip()
        else:
            base = current_prompt.strip()

        return base + cot_instruction


class MIPRO(OptimizationStrategy):
    """Multi-prompt Instruction Proposal Optimizer (inspired by DSPy MIPROv2).

    Process:
    1. Generate N candidate prompts in parallel via LLM
    2. Evaluate each candidate against the dataset
    3. Select the best-scoring candidate
    4. Use it as the base for the next iteration
    5. Optionally combine top-K candidates via LLM fusion

    This explores a wider prompt space than iterative rewrite,
    reducing the risk of getting stuck in local optima.
    """

    name = "mipro"

    def __init__(
        self,
        model: str = "ollama/mistral:7b",
        n_candidates: int = 5,
        fuse_top_k: int = 2,
    ):
        self.model = model
        self.n_candidates = n_candidates
        self.fuse_top_k = fuse_top_k

    async def step(
        self,
        current_prompt: str,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        import asyncio

        # Analyze failures for context
        failures = sorted(eval_results, key=lambda r: r.get("avg_score", 0))[:5]
        failure_summary = "\n".join(
            f"- Q: {f.get('query', '')[:80]} → Score: {f.get('avg_score', 0):.2f}"
            for f in failures
        )

        # Generate N candidate prompts in parallel
        tasks = [
            self._generate_candidate(current_prompt, failure_summary, dataset_sample, i, iteration)
            for i in range(self.n_candidates)
        ]
        candidates = await asyncio.gather(*tasks)
        candidates = [c for c in candidates if c and c != current_prompt]

        if not candidates:
            return current_prompt

        # If we have enough candidates, fuse the top ones
        if len(candidates) >= self.fuse_top_k:
            return await self._fuse_candidates(
                candidates[: self.fuse_top_k], current_prompt, failure_summary
            )

        return candidates[0]

    async def _generate_candidate(
        self,
        current_prompt: str,
        failure_summary: str,
        samples: list[dict[str, Any]],
        variant_idx: int,
        iteration: int,
    ) -> str:
        """Generate a single candidate prompt variant."""
        perspectives = [
            "Focus on making instructions more specific and actionable.",
            "Focus on adding output format constraints and structure.",
            "Focus on improving tone, clarity, and conciseness.",
            "Focus on adding edge case handling and error recovery.",
            "Focus on adding domain-specific knowledge and expertise.",
            "Focus on reducing ambiguity and adding explicit examples.",
            "Focus on improving step-by-step reasoning instructions.",
        ]
        perspective = perspectives[variant_idx % len(perspectives)]

        prompt = (
            f"You are a prompt engineering expert. Generate a VARIATION of the system prompt below.\n\n"
            f"## Perspective\n{perspective}\n\n"
            f"## Current Prompt\n```\n{current_prompt}\n```\n\n"
            f"## Known Failure Patterns\n{failure_summary}\n\n"
            f"## Rules\n"
            f"- Create a DIFFERENT approach to the same task\n"
            f"- Address the failure patterns\n"
            f"- Keep the core purpose intact\n"
            f"- Be creative but practical\n\n"
            f"Reply with ONLY the new system prompt text.\n"
        )
        return await self._call_llm(prompt)

    async def _fuse_candidates(self, candidates: list[str], original: str, failures: str) -> str:
        """Fuse multiple candidate prompts into one optimal prompt."""
        candidates_text = "\n\n---\n\n".join(
            f"### Candidate {i + 1}\n```\n{c}\n```" for i, c in enumerate(candidates)
        )

        prompt = (
            f"You are a prompt engineering expert. Combine the best elements "
            f"from these candidate prompts into ONE optimal prompt.\n\n"
            f"{candidates_text}\n\n"
            f"## Known Failures to Address\n{failures}\n\n"
            f"Create the BEST possible prompt by fusing the strongest ideas.\n"
            f"Reply with ONLY the fused prompt text.\n"
        )
        return await self._call_llm(prompt)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM."""
        import httpx

        model_name = (
            self.model.replace("ollama/", "") if self.model.startswith("ollama/") else self.model
        )
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    return str(resp.json().get("response", "")).strip()
        except Exception as exc:
            logger.warning(f"MIPRO LLM call failed: {exc}")
        return ""


class BootstrapRandomSearch(OptimizationStrategy):
    """Bootstrap Few-Shot with Random Search (inspired by DSPy).

    Process:
    1. Run all dataset points through agent
    2. Score each response
    3. Generate N random subsets of K top-scoring examples
    4. For each subset, create a few-shot prompt
    5. Evaluate each prompt candidate
    6. Select the best-scoring configuration

    This is more robust than simple FewShotBootstrap because it
    explores multiple example combinations.
    """

    name = "bootstrap_random_search"

    def __init__(
        self,
        model: str = "ollama/mistral:7b",
        n_candidates: int = 8,
        k_examples: int = 4,
    ):
        self.model = model
        self.n_candidates = n_candidates
        self.k_examples = k_examples

    async def step(
        self,
        current_prompt: str,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        import random

        # Get scored examples
        scored = [r for r in eval_results if r.get("avg_score", 0) > 0.3 and r.get("response")]
        if len(scored) < self.k_examples:
            scored = eval_results[: self.k_examples]

        # Generate N random subsets
        best_subset: list[dict] = []
        best_diversity = 0.0

        for _ in range(self.n_candidates):
            if len(scored) <= self.k_examples:
                subset = scored
            else:
                # Weighted random: prefer higher-scoring examples
                weights = [r.get("avg_score", 0.5) ** 2 for r in scored]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]

                indices: set[int] = set()
                while len(indices) < self.k_examples and len(indices) < len(scored):
                    idx = random.choices(range(len(scored)), weights=weights, k=1)[0]
                    indices.add(idx)

                subset = [scored[i] for i in indices]

            # Compute diversity score (prefer diverse query types)
            queries = {s.get("query", "")[:30] for s in subset}
            diversity = len(queries) / max(len(subset), 1)

            avg_quality = sum(s.get("avg_score", 0) for s in subset) / max(len(subset), 1)
            combined = avg_quality * 0.7 + diversity * 0.3

            if combined > best_diversity:
                best_diversity = combined
                best_subset = subset

        # Build few-shot prompt
        examples_block = "\n\n".join(
            f"**Example {i + 1}:**\n"
            f"User: {ex.get('query', '')}\n"
            f"Assistant: {ex.get('response', ex.get('expected', ''))}"
            for i, ex in enumerate(best_subset)
        )

        # Clean existing examples
        base = self._remove_examples(current_prompt)
        return f"{base}\n\n## Few-Shot Examples\n\n{examples_block}"

    def _remove_examples(self, prompt: str) -> str:
        """Remove existing examples section."""
        for marker in ["## Few-Shot Examples", "## Examples", "**Example 1"]:
            if marker in prompt:
                return prompt.split(marker)[0].strip()
        return prompt.strip()


class EnsembleOptimizer(OptimizationStrategy):
    """Tries multiple strategies per iteration and picks the best.

    Inspired by DSPy's approach of exploring multiple optimization
    paths simultaneously. Each iteration:
    1. Run IterativeRewrite, FewShotBootstrap, and ChainOfThought
    2. Generate one candidate prompt per strategy
    3. Fuse the best ideas into a single prompt
    """

    name = "ensemble"

    def __init__(self, model: str = "ollama/mistral:7b"):
        self.model = model
        self._strategies = [
            IterativeRewrite(model=model),
            FewShotBootstrap(model=model),
            ChainOfThought(model=model),
        ]

    async def step(
        self,
        current_prompt: str,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        import asyncio

        # Run all strategies in parallel
        tasks = [
            strategy.step(current_prompt, eval_results, dataset_sample, iteration)
            for strategy in self._strategies
        ]
        candidates = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        valid = [c for c in candidates if isinstance(c, str) and c and c != current_prompt]

        if not valid:
            return current_prompt

        if len(valid) == 1:
            return valid[0]

        # Fuse the best elements from each strategy
        return await self._fuse(valid, current_prompt)

    async def _fuse(self, candidates: list[str], original: str) -> str:
        """Fuse strategy outputs into one optimal prompt."""
        parts = "\n\n---\n\n".join(
            f"Strategy {i + 1}:\n```\n{c[:500]}\n```" for i, c in enumerate(candidates)
        )

        prompt = (
            f"You are a prompt engineering expert. Three different optimization "
            f"strategies produced these prompt candidates:\n\n{parts}\n\n"
            f"Combine the BEST elements from each into ONE optimal prompt.\n"
            f"Preserve: specific instructions, examples, reasoning steps.\n"
            f"Reply with ONLY the fused prompt text.\n"
        )

        model_name = (
            self.model.replace("ollama/", "") if self.model.startswith("ollama/") else self.model
        )
        try:
            import httpx

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    return str(resp.json().get("response", "")).strip()
        except Exception:
            pass

        # Fallback: return the longest candidate (typically most detailed)
        return max(candidates, key=len)


# =====================================================================
# Strategy Factory
# =====================================================================

_STRATEGIES: dict[str, type[OptimizationStrategy]] = {
    "iterative_rewrite": IterativeRewrite,
    "few_shot_bootstrap": FewShotBootstrap,
    "few_shot": FewShotBootstrap,
    "chain_of_thought": ChainOfThought,
    "cot": ChainOfThought,
    "mipro": MIPRO,
    "bootstrap_random_search": BootstrapRandomSearch,
    "random_search": BootstrapRandomSearch,
    "ensemble": EnsembleOptimizer,
}


def resolve_strategy(
    name: str | OptimizationStrategy,
    model: str = "ollama/mistral:7b",
) -> OptimizationStrategy:
    """Resolve a strategy name to an instance."""
    if isinstance(name, OptimizationStrategy):
        return name
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: '{name}'. Available: {list(_STRATEGIES.keys())}")
    return _STRATEGIES[name](model=model)
