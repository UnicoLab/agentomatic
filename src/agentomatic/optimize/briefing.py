"""Full-context briefing builder for prompt optimizers (SLMs and LLMs).

Both small and frontier models rewrite better with *explicit* structure:
current config, search space, dataset samples, metric names, and
per-example I/O.  This module assembles that briefing once so rewrite /
GEPA / MIPRO / multi-pass refiners all see the same rich picture.

Multi-pass refine defaults:

* SLMs / local providers → 3 passes (draft → critique → revise)
* Frontier LLMs → 2 passes (draft → self-check revise)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agentomatic.optimize.config import PromptRuntimeConfig
    from agentomatic.optimize.context import OptimizationContext
    from agentomatic.optimize.search_space import PromptSearchSpace

# Heuristic: local / small models benefit from deeper multi-pass refine.
_SLM_PROVIDER_PREFIXES = ("omlx/", "ollama/", "lmstudio/", "vllm/")
_SLM_SIZE_RE = re.compile(
    r"(?i)(?:^|[-_/])(?:[1-9]|1[0-4])\s*b(?:[-_/]|$)|"
    r"(?:small|mini|tiny|mlx|qwen2\.5|qwen3\.5|mistral|phi|gemma|llama3\.2)"
)

RefineStyle = Literal["slm", "llm"]


def looks_like_slm(model: Any) -> bool:
    """Return True when *model* looks like a small / local LLM.

    Callables are treated as unknown (not SLM).  String specs matching
    local providers (``omlx/``, ``ollama/``, …) or small-size tokens
    (``7b``, ``9B``, ``mlx``, …) return True.
    """
    if not isinstance(model, str):
        return False
    lowered = model.lower().strip()
    if any(lowered.startswith(p) for p in _SLM_PROVIDER_PREFIXES):
        return True
    return bool(_SLM_SIZE_RE.search(lowered))


def refine_style_for(model: Any) -> RefineStyle:
    """Return ``\"slm\"`` or ``\"llm\"`` style for rewrite prompts."""
    return "slm" if looks_like_slm(model) else "llm"


def resolve_rewrite_passes(
    model: Any,
    *,
    rewrite_passes: int | None = None,
    multipass: bool = True,
    slm_multipass: bool = True,
    llm_multipass: bool = True,
    slm_default_passes: int = 3,
    llm_default_passes: int = 2,
) -> int:
    """Resolve how many draft→critique→revise passes to run.

    Args:
        model: Rewrite model spec.
        rewrite_passes: Explicit override (``None`` = auto).
        multipass: Master switch; when False always returns 1.
        slm_multipass: Enable auto multi-pass for SLM-like models.
        llm_multipass: Enable auto multi-pass for frontier / cloud LLMs.
        slm_default_passes: Pass count for auto-detected SLMs (default 3).
        llm_default_passes: Pass count for non-SLM LLMs (default 2).

    Returns:
        Pass count ≥ 1.
    """
    if rewrite_passes is not None:
        return max(1, int(rewrite_passes))
    if not multipass:
        return 1
    if looks_like_slm(model):
        if slm_multipass:
            return max(1, int(slm_default_passes))
        return 1
    if llm_multipass:
        return max(1, int(llm_default_passes))
    return 1


def briefing_limits_for(model: Any) -> dict[str, int]:
    """Return richer I/O limits for frontier LLMs, compact for SLMs."""
    if looks_like_slm(model):
        return {
            "max_failures": 6,
            "max_successes": 3,
            "max_samples": 8,
        }
    return {
        "max_failures": 10,
        "max_successes": 5,
        "max_samples": 12,
    }


def _clip(value: Any, limit: int = 400) -> str:
    text = str(value) if value is not None else "N/A"
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _fmt_json(data: Any, limit: int = 800) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(data)
    return _clip(text, limit)


def format_runtime_config(config: PromptRuntimeConfig) -> str:
    """Serialize the full runtime config surface for LLM consumption."""
    lines = [
        "### System prompt",
        "```",
        config.system_prompt or "(empty)",
        "```",
    ]
    if config.user_template:
        lines += ["### User template", "```", config.user_template, "```"]
    if config.few_shot_examples:
        lines += [
            f"### Few-shot examples ({len(config.few_shot_examples)})",
            _fmt_json(config.few_shot_examples[:8], 1200),
        ]
    if config.output_contract:
        lines += ["### Output contract", _fmt_json(config.output_contract)]
    if config.model_params:
        lines += ["### Model params", _fmt_json(config.model_params)]
    if config.rag_params:
        lines += ["### RAG params", _fmt_json(config.rag_params)]
    if config.tool_params:
        lines += ["### Tool params", _fmt_json(config.tool_params)]
    if config.model_choice:
        lines += [f"### Model choice: `{config.model_choice}`"]
    if getattr(config, "fallback_model", None):
        lines += [f"### Fallback model: `{config.fallback_model}`"]
    return "\n".join(lines)


def format_search_space(space: PromptSearchSpace | None) -> str:
    """Summarize which surfaces are being optimized."""
    if space is None:
        return "Search space: (default / unspecified)"
    active = []
    if getattr(space, "optimize_system_prompt", False):
        active.append("system_prompt")
    if getattr(space, "optimize_few_shot", False):
        active.append("few_shot")
    if getattr(space, "optimize_model_params", False):
        active.append("model_params")
    if getattr(space, "optimize_rag_params", False):
        active.append("rag_params")
    if getattr(space, "optimize_tool_params", False):
        active.append("tool_params")
    if getattr(space, "optimize_model_choice", False):
        active.append("model_choice")
    lines = [f"Active surfaces: {', '.join(active) or '(none)'}"]
    model_space = getattr(space, "model_param_space", None) or {}
    if model_space:
        lines.append(f"Model param grid: {_fmt_json(model_space, 500)}")
    return "\n".join(lines)


def format_dataset_samples(samples: list[Any], *, max_items: int = 8) -> str:
    """Format train/val sample queries + expected answers."""
    if not samples:
        return "No dataset samples provided."
    lines: list[str] = []
    for i, raw in enumerate(samples[:max_items], 1):
        if isinstance(raw, dict):
            query = raw.get("query") or raw.get("input") or ""
            expected = raw.get("expected_answer") or raw.get("expected") or raw.get("output") or ""
        else:
            query = getattr(raw, "query", "")
            expected = getattr(raw, "expected_answer", None) or getattr(raw, "expected", "")
        lines.append(f"{i}. Q: {_clip(query, 220)}")
        lines.append(f"   Expected: {_clip(expected, 220)}")
    return "\n".join(lines)


def format_eval_io(
    eval_results: list[dict[str, Any]],
    *,
    max_failures: int = 6,
    max_successes: int = 3,
    clip: int = 350,
) -> str:
    """Format per-example inputs/outputs/scores for the rewriter."""
    if not eval_results:
        return "No evaluation results yet."

    scored = sorted(
        eval_results,
        key=lambda r: float(r.get("score", r.get("avg_score", 0.0)) or 0.0),
    )
    failures = scored[:max_failures]
    successes = scored[-max_successes:] if scored else []

    lines: list[str] = ["### Failures (lowest scores)"]
    for idx, fail in enumerate(failures, 1):
        score = float(fail.get("score", fail.get("avg_score", 0.0)) or 0.0)
        lines.append(f"\n**Failure {idx}** (score={score:.3f})")
        lines.append(f"- Input/query: {_clip(fail.get('query'), clip)}")
        lines.append(f"- Expected: {_clip(fail.get('expected'), clip)}")
        lines.append(f"- Actual output: {_clip(fail.get('response'), clip)}")
        issues = fail.get("feedback") or fail.get("reason") or fail.get("details")
        if issues:
            lines.append(f"- Judge/feedback: {_clip(issues, min(clip, 280))}")
        dims = fail.get("dimensions") or {}
        if dims:
            lines.append(
                "- Dimensions: " + ", ".join(f"{k}={float(v):.3f}" for k, v in dims.items())
            )
        ret_ctx = fail.get("retrieval_context") or []
        if ret_ctx:
            lines.append("- Retrieval: " + "; ".join(_clip(d, 100) for d in ret_ctx[:3]))
        tools = fail.get("tool_calls") or []
        if tools:
            names = ", ".join(
                str(t.get("name", t) if isinstance(t, dict) else t) for t in tools[:4]
            )
            lines.append(f"- Tools: {names}")
        reasoning = fail.get("reasoning") or ""
        if reasoning:
            lines.append(f"- Reasoning: {_clip(reasoning, 200)}")
        meta = fail.get("metadata") or {}
        if isinstance(meta, dict) and meta:
            useful = {
                k: meta[k] for k in ("used_prompt", "temperature", "model", "error") if k in meta
            }
            if useful:
                lines.append(f"- Run metadata: {_fmt_json(useful, 300)}")

    if successes:
        lines.append("\n### Successes (highest scores)")
        for idx, suc in enumerate(successes, 1):
            score = float(suc.get("score", suc.get("avg_score", 0.0)) or 0.0)
            lines.append(f"\n**Success {idx}** (score={score:.3f})")
            lines.append(f"- Input/query: {_clip(suc.get('query'), min(clip, 220))}")
            lines.append(f"- Expected: {_clip(suc.get('expected'), min(clip, 220))}")
            lines.append(f"- Actual output: {_clip(suc.get('response'), min(clip, 220))}")

    return "\n".join(lines)


def build_full_optimization_briefing(
    *,
    current_config: PromptRuntimeConfig,
    eval_results: list[dict[str, Any]],
    dataset_sample: list[Any] | None = None,
    search_space: PromptSearchSpace | None = None,
    context: OptimizationContext | None = None,
    metric_names: list[str] | None = None,
    agent_name: str = "",
    rewrite_model: str = "",
    max_failures: int | None = None,
    max_successes: int | None = None,
    max_samples: int | None = None,
) -> str:
    """Assemble a structured briefing for rewrite / multi-pass refine.

    Includes config, params, search space, metrics, dataset samples,
    full eval I/O, and optimization history so rewrite models (SLM or
    LLM) do not have to infer missing context.

    When *max_failures* / *max_successes* / *max_samples* are omitted,
    limits are chosen from :func:`briefing_limits_for` based on
    *rewrite_model*.
    """
    limits = briefing_limits_for(rewrite_model or "")
    if max_failures is None:
        max_failures = limits["max_failures"]
    if max_successes is None:
        max_successes = limits["max_successes"]
    if max_samples is None:
        max_samples = limits["max_samples"]
    clip = 350 if looks_like_slm(rewrite_model or "") else 700

    metrics = metric_names
    if metrics is None and context is not None:
        metrics = list(context.metric_names or [])

    sections: list[str] = [
        "# Optimization briefing",
        "Use ALL sections below. Do not invent missing data — improve the "
        "prompt/config using the observed inputs, outputs, expected answers, "
        "scores, parameters, and judge feedback.",
    ]
    if agent_name:
        sections.append(f"**Agent:** `{agent_name}`")
    if rewrite_model:
        style = refine_style_for(rewrite_model)
        sections.append(f"**Rewrite model:** `{rewrite_model}` ({style} style)")
    if context is not None:
        sections.append(
            f"**Progress:** round {context.round_idx + 1}/{max(context.total_rounds, 1)} | "
            f"baseline={context.baseline_score:.4f} | current_best={context.current_score:.4f}"
        )
        history = context.format_score_history(max_rounds=6)
        if history and "No previous" not in history:
            sections += ["## Score history", history, context.format_score_sparkline()]
        dims = context.format_dimension_table()
        if dims and "No per-dimension" not in dims:
            sections += ["## Dimension deltas", dims]
        clusters = context.format_failure_clusters()
        if clusters and "No failure clusters" not in clusters:
            sections += ["## Failure clusters", clusters]

    if metrics:
        sections += ["## Metrics", ", ".join(metrics)]

    sections += [
        "## Current runtime config (prompt + params)",
        format_runtime_config(current_config),
        "## Search space",
        format_search_space(search_space),
        "## Dataset samples (ground truth)",
        format_dataset_samples(list(dataset_sample or []), max_items=max_samples),
        "## Evaluation I/O (what the agent actually did)",
        format_eval_io(
            eval_results,
            max_failures=max_failures,
            max_successes=max_successes,
            clip=clip,
        ),
    ]
    return "\n\n".join(sections)


def extract_prompt_text(raw: str, *, fallback: str = "") -> str:
    """Pull the final system-prompt text out of a free-form LLM reply."""
    new_prompt = (raw or "").strip()
    if "---" in new_prompt:
        new_prompt = new_prompt.split("---", 1)[-1].strip()
    if "<thinking>" in new_prompt and "</thinking>" in new_prompt:
        new_prompt = new_prompt.split("</thinking>", 1)[-1].strip()
    # Prefer fenced blocks when present
    fence = re.search(r"```(?:text|markdown|prompt)?\s*\n(.*?)```", new_prompt, re.S | re.I)
    if fence:
        new_prompt = fence.group(1).strip()
    elif new_prompt.startswith("```"):
        lines = new_prompt.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        new_prompt = "\n".join(lines[1:end]).strip()
    # Drop common labels
    for prefix in ("Improved system prompt:", "New system prompt:", "System prompt:"):
        if new_prompt.lower().startswith(prefix.lower()):
            new_prompt = new_prompt[len(prefix) :].strip()
    return new_prompt.strip() or fallback


def _draft_instructions(style: RefineStyle, passes: int) -> str:
    if style == "slm":
        return (
            f"## Task (pass 1/{passes}: DRAFT)\n"
            "Produce an improved system prompt that:\n"
            "1. Fixes root causes visible in the failure I/O\n"
            "2. Preserves patterns from successes\n"
            "3. Mentions any required keywords / output constraints from Expected\n"
            "4. Keeps the same core task/role\n"
            "5. Is concrete and short enough for a small model to follow\n\n"
            "Think briefly, then output ONLY the new system prompt after a '---' line.\n"
        )
    return (
        f"## Task (pass 1/{passes}: DRAFT)\n"
        "Produce an improved system prompt that:\n"
        "1. Diagnoses root causes from failure I/O and judge feedback\n"
        "2. Preserves high-scoring success patterns\n"
        "3. Encodes expected keywords, format, and output contracts explicitly\n"
        "4. Reflects relevant model/RAG/tool parameters as operational guidance\n"
        "5. Adds crisp edge-case / ambiguity handling without bloating the prompt\n"
        "6. Keeps the same core task/role and measurable acceptance criteria\n\n"
        "Reason carefully, then output ONLY the new system prompt after a '---' line.\n"
    )


def _revise_instructions(style: RefineStyle, turn: int, passes: int) -> str:
    if style == "slm":
        return (
            f"## Task (pass {turn}/{passes}: REVISE)\n"
            "Ensure every failure I/O issue and expected keyword/constraint is "
            "covered. Reply with ONLY the improved system prompt after '---'.\n"
        )
    return (
        f"## Task (pass {turn}/{passes}: REVISE)\n"
        "Tighten the draft against the briefing: close every failure gap, keep "
        "success patterns, strengthen format/constraints, and remove fluff. "
        "Reply with ONLY the improved system prompt after '---'.\n"
    )


def _critique_instructions(style: RefineStyle, turn: int, passes: int) -> str:
    if style == "slm":
        return (
            f"## Task (pass {turn}/{passes}: CRITIQUE)\n"
            "List concrete gaps vs the failure I/O and expected answers. "
            "Check: missing constraints, ignored keywords, unclear format, "
            "lost success patterns, and whether model/RAG/tool params should "
            "be reflected in instructions.\n"
            "Reply with a short bullet critique only (no full prompt).\n"
        )
    return (
        f"## Task (pass {turn}/{passes}: CRITIQUE)\n"
        "Act as a senior prompt reviewer. List concrete gaps vs failure I/O, "
        "expected answers, metrics, and params. Check: missing constraints, "
        "ignored keywords, weak format contracts, lost success patterns, "
        "underspecified edge cases, and unused RAG/tool/model settings.\n"
        "Prioritize the highest-impact issues. Reply with bullets only "
        "(no full prompt).\n"
    )


async def multipass_refine_prompt(
    *,
    model: Any,
    briefing: str,
    current_prompt: str,
    passes: int = 3,
    temperature: float = 0.5,
    max_tokens: int = 3000,
    style: RefineStyle | None = None,
) -> tuple[str, list[str]]:
    """Run draft → critique → revise turns for high-quality rewrites.

    Works for both SLMs and frontier LLMs. Prompt wording adapts via
    *style* (auto-detected from *model* when omitted).

    With ``passes=1`` this is a single draft call. With ``passes>=3`` the
    loop is draft, then repeating (critique, revise) pairs. With
    ``passes=2`` it runs draft then a self-check revise against the briefing.

    Returns:
        ``(final_prompt, pass_notes)``.
    """
    from agentomatic.optimize.llm_caller import LLMCaller

    passes = max(1, int(passes))
    notes: list[str] = []
    style = style or refine_style_for(model)
    # Frontier models can use a bit more output budget on the draft.
    draft_tokens = max_tokens if style == "slm" else max(max_tokens, 4000)

    role = (
        "You are an expert prompt engineer optimizing with a SMALL language model."
        if style == "slm"
        else "You are an expert prompt engineer optimizing with a frontier LLM."
    )
    draft_prompt = (
        f"{role} Rewrite the system prompt using the FULL briefing below.\n\n"
        f"{briefing}\n\n"
        f"{_draft_instructions(style, passes)}"
    )
    raw = await LLMCaller.call(
        model,
        draft_prompt,
        temperature=temperature,
        max_tokens=draft_tokens,
    )
    draft = extract_prompt_text(raw, fallback=current_prompt)
    notes.append(f"pass1_draft chars={len(draft)} style={style}")

    if passes == 1:
        return draft, notes

    if passes == 2:
        revise_prompt = (
            "Self-check and revise the DRAFT using the briefing.\n\n"
            f"{briefing}\n\n"
            f"## Draft\n```\n{draft}\n```\n\n"
            f"{_revise_instructions(style, 2, passes)}"
        )
        raw = await LLMCaller.call(
            model,
            revise_prompt,
            temperature=temperature,
            max_tokens=draft_tokens,
        )
        draft = extract_prompt_text(raw, fallback=draft)
        notes.append(f"pass2_revise chars={len(draft)}")
        return draft, notes

    # passes >= 3: critique/revise pairs after the draft
    remaining = passes - 1
    turn = 2
    critique = ""
    while remaining > 0:
        critique_prompt = (
            "You are reviewing a DRAFT system prompt for quality.\n\n"
            f"{briefing}\n\n"
            f"## Draft system prompt\n```\n{draft}\n```\n\n"
            f"{_critique_instructions(style, turn, passes)}"
        )
        critique = (
            await LLMCaller.call(
                model,
                critique_prompt,
                temperature=min(temperature, 0.4),
                max_tokens=min(max_tokens, 1500 if style == "slm" else 2000),
            )
        ).strip()
        notes.append(f"pass{turn}_critique chars={len(critique)}")
        remaining -= 1
        turn += 1
        if remaining <= 0:
            break

        revise_prompt = (
            "Revise the DRAFT system prompt using the critique and briefing.\n\n"
            f"{briefing}\n\n"
            f"## Draft\n```\n{draft}\n```\n\n"
            f"## Critique\n{critique}\n\n"
            f"{_revise_instructions(style, turn, passes)}"
        )
        raw = await LLMCaller.call(
            model,
            revise_prompt,
            temperature=temperature,
            max_tokens=draft_tokens,
        )
        draft = extract_prompt_text(raw, fallback=draft)
        notes.append(f"pass{turn}_revise chars={len(draft)}")
        remaining -= 1
        turn += 1

    return draft, notes
