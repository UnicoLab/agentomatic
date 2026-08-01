"""APO — Automatic Prompt Optimization via textual gradients + beam search.

Ports the Agent Lightning APO loop (ProTeGi / TextGrad style) into
Agentomatic's :class:`BaseFitterOptimizer` interface:

1. Sample low-scoring rollouts with message-level traces
2. Ask a gradient LLM for a concrete critique (textual gradient)
3. Ask an edit LLM to revise the prompt under that critique
4. Emit ``branch_factor`` candidates; keep a local beam of top prompts

No POML / agentlightning dependency — plain Python prompt templates.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.optimize.config import PromptCandidate, PromptRuntimeConfig
from agentomatic.optimize.fitter_optimizers import BaseFitterOptimizer
from agentomatic.optimize.llm_caller import LLMCaller
from agentomatic.optimize.search_space import PromptSearchSpace
from agentomatic.optimize.trace_adapter import TraceToCritiqueContext

if TYPE_CHECKING:
    from agentomatic.optimize.context import OptimizationContext
    from agentomatic.optimize.llm_types import LLMSpec

_TEXT_GRADIENT_TEMPLATE = """\
You optimize a prompt template.

## Original Prompt Template
{prompt_template}

## Experiments with Original Prompt Template
{experiments}

## Your Task
Produce a brief critique listing specific causes for the error or ways to \
raise reward next time.
Return a bullet list with concrete, testable changes (format, constraints, \
ordering, definitions). Do not rewrite the full prompt yet — critique only.
"""

_APPLY_EDIT_TEMPLATE = """\
Revise the given prompt template using the critique as constraints and \
improvement guide.

## Revision Rules
1. Rewrite or restructure the prompt if critique implies it.
2. Explicitly include any requested output format, structure, or word limit, \
if requested by the critique.
3. Prioritize mechanism-first phrasing: define what to do, then how to do it.
4. Preserve placeholder variables inside curly brackets (e.g. {{query}}).

## Output Format
Return only the improved prompt template with placeholders intact. Do not \
include explanations, headers, or markdown fences.

## Prompt Template
{prompt_template}

## Critique
{critique}
"""


def _format_experiments(experiments: list[dict[str, Any]], *, max_chars: int = 6000) -> str:
    """Render critique experiments for the gradient prompt."""
    blocks: list[str] = []
    for idx, exp in enumerate(experiments, 1):
        reward = exp.get("final_reward")
        reward_s = f"{reward:.3f}" if isinstance(reward, (int, float)) else "n/a"
        lines = [
            f"### Experiment {idx}",
            f"Status: {exp.get('status', 'unknown')} | Final reward: {reward_s}",
            f"Query: {str(exp.get('query') or '')[:400]}",
            f"Response: {str(exp.get('response') or '')[:600]}",
        ]
        if exp.get("expected"):
            lines.append(f"Expected: {str(exp['expected'])[:400]}")
        if exp.get("feedback"):
            lines.append(f"Judge feedback: {str(exp['feedback'])[:400]}")
        messages = exp.get("messages") or []
        if messages:
            msg_lines = []
            for msg in messages[:8]:
                role = msg.get("role", "?")
                content = str(msg.get("content") or "")[:350]
                name = msg.get("name")
                prefix = f"{role}/{name}" if name else role
                msg_lines.append(f"- {prefix}: {content}")
            lines.append("Traces:\n" + "\n".join(msg_lines))
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    return text[:max_chars]


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped the prompt."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


@dataclass(slots=True)
class APOOptimizer(BaseFitterOptimizer):
    """Textual-gradient APO with local beam state.

    Each :meth:`propose` call expands the current beam by
    ``branch_factor`` edited prompts (or expands from the current config
    when the beam is empty).
    """

    name: str = "apo"
    gradient_model: LLMSpec = "ollama/qwen2.5:7b"
    apply_edit_model: LLMSpec | None = None
    diversity_temperature: float = 0.9
    gradient_batch_size: int = 4
    branch_factor: int = 4
    beam_width: int = 4
    node_match: str | None = None

    _beam: list[tuple[str, float | None]] = field(default_factory=list, repr=False)
    """List of ``(prompt, score)`` kept across fitter rounds."""

    def __post_init__(self) -> None:
        if self.apply_edit_model is None:
            self.apply_edit_model = self.gradient_model

    def update_beam(self, prompt: str, score: float | None) -> None:
        """Record a scored prompt into the local beam and trim to width."""
        self._beam.append((prompt, score))
        scored: list[tuple[str, float | None]] = [(p, s) for p, s in self._beam if s is not None]
        unscored: list[tuple[str, float | None]] = [(p, s) for p, s in self._beam if s is None]
        scored.sort(key=lambda item: float(item[1] or 0.0), reverse=True)
        self._beam = (scored + unscored)[: self.beam_width]

    async def compute_textual_gradient(
        self,
        prompt_template: str,
        experiments: list[dict[str, Any]],
    ) -> str:
        """Ask the gradient model for a critique of the prompt."""
        prompt = _TEXT_GRADIENT_TEMPLATE.format(
            prompt_template=prompt_template,
            experiments=_format_experiments(experiments),
        )
        critique = await LLMCaller.call(
            self.gradient_model,
            prompt,
            temperature=self.diversity_temperature,
            max_tokens=1200,
        )
        return (critique or "").strip()

    async def apply_edit(self, prompt_template: str, critique: str) -> str:
        """Apply a critique to produce an improved prompt."""
        if not critique.strip():
            return ""
        prompt = _APPLY_EDIT_TEMPLATE.format(
            prompt_template=prompt_template,
            critique=critique,
        )
        edited = await LLMCaller.call(
            self.apply_edit_model or self.gradient_model,
            prompt,
            temperature=self.diversity_temperature,
            max_tokens=3000,
        )
        return _strip_fences(edited or "")

    async def textual_gradient_and_apply_edit(
        self,
        prompt_template: str,
        experiments: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Critique then edit; return ``(new_prompt, critique)``."""
        critique = await self.compute_textual_gradient(prompt_template, experiments)
        if not critique:
            logger.warning("APOOptimizer: empty textual gradient")
            return "", ""
        new_prompt = await self.apply_edit(prompt_template, critique)
        return new_prompt, critique

    async def propose(
        self,
        current_config: PromptRuntimeConfig,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: PromptSearchSpace,
        iteration: int = 0,
        context: OptimizationContext | None = None,
    ) -> list[PromptCandidate]:
        """Generate ``branch_factor`` APO candidates from textual gradients."""
        del dataset_sample  # unused — traces come from eval_results
        if not search_space.optimize_system_prompt:
            logger.info("APOOptimizer: system prompt optimisation disabled — skipping")
            return []

        critique_ctx = TraceToCritiqueContext()
        critique_ctx.adapter.node_match = self.node_match or search_space.node_match
        experiments = [
            exp.to_dict()
            for exp in critique_ctx.from_eval_details(
                eval_results,
                max_items=self.gradient_batch_size,
            )
        ]
        if not experiments:
            # Fallback synthetic experiment from current prompt only
            experiments = [
                {
                    "status": "succeeded",
                    "final_reward": None,
                    "messages": [],
                    "query": "",
                    "response": "",
                    "feedback": "No eval traces available; improve clarity and constraints.",
                }
            ]

        parents: list[str]
        if self._beam:
            parents = [p for p, _ in self._beam[: self.beam_width]]
        else:
            parents = [current_config.system_prompt]

        # Expand each parent up to branch_factor total candidates
        tasks: list[asyncio.Task[tuple[str, str, str]]] = []

        async def _branch(parent_prompt: str, branch_idx: int) -> tuple[str, str, str]:
            new_prompt, critique = await self.textual_gradient_and_apply_edit(
                parent_prompt,
                experiments,
            )
            return parent_prompt, new_prompt, critique

        branch_idx = 0
        while branch_idx < self.branch_factor:
            for parent in parents:
                if branch_idx >= self.branch_factor:
                    break
                tasks.append(asyncio.create_task(_branch(parent, branch_idx)))
                branch_idx += 1

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[PromptCandidate] = []
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("APOOptimizer branch failed: {}", result)
                continue
            parent_prompt, new_prompt, critique = result
            if not new_prompt.strip() or new_prompt.strip() == parent_prompt.strip():
                continue
            # Seed beam with unscored candidate prompts
            self.update_beam(new_prompt, None)
            candidates.append(
                PromptCandidate(
                    name=f"apo_{iteration:03d}_b{idx:02d}",
                    config=PromptRuntimeConfig(
                        system_prompt=new_prompt,
                        user_template=current_config.user_template,
                        few_shot_examples=list(current_config.few_shot_examples),
                        output_contract=current_config.output_contract,
                        model_params=dict(current_config.model_params),
                        rag_params=dict(current_config.rag_params),
                        tool_params=dict(current_config.tool_params),
                        model_choice=current_config.model_choice,
                        fallback_model=current_config.fallback_model,
                        routing_config=dict(current_config.routing_config),
                    ),
                    source="apo",
                    mutation_notes=(
                        f"APO textual-gradient edit (branch {idx}). Critique: {critique[:400]}"
                    ),
                    metadata={"critique": critique, "parent_prompt": parent_prompt[:500]},
                )
            )

        logger.info(
            "APOOptimizer: iter={} produced {} candidates (beam={})",
            iteration,
            len(candidates),
            len(self._beam),
        )
        return candidates
