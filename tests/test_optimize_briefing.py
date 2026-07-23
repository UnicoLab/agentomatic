"""Tests for SLM/LLM briefing + multi-pass rewrite helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentomatic.optimize.briefing import (
    briefing_limits_for,
    build_full_optimization_briefing,
    extract_prompt_text,
    looks_like_slm,
    multipass_refine_prompt,
    refine_style_for,
    resolve_rewrite_passes,
)
from agentomatic.optimize.config import PromptRuntimeConfig
from agentomatic.optimize.context import OptimizationContext
from agentomatic.optimize.fitter_optimizers import RewriteOptimizer, resolve_fitter_optimizer
from agentomatic.optimize.search_space import PromptSearchSpace


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("omlx/Qwen3.5-9B-MLX-4bit", True),
        ("ollama/qwen2.5:7b", True),
        ("openai/gpt-4.1", False),
        ("openai/gpt-4.1-mini", False),  # cloud provider wins over "mini"
        ("anthropic/claude-sonnet-4", False),
        ("gemini/gemini-3.1-flash-lite", False),  # must not match "mini" in gemini
        (lambda x: x, False),
        ("vllm/llama-3.2-3b", True),
        ("lmstudio/phi-3-mini", True),
        ("ollama/phi-3-mini", True),
    ],
)
def test_looks_like_slm(model: object, expected: bool) -> None:
    assert looks_like_slm(model) is expected


def test_refine_style_for() -> None:
    assert refine_style_for("omlx/qwen") == "slm"
    assert refine_style_for("openai/gpt-4.1") == "llm"


def test_briefing_limits_richer_for_llms() -> None:
    slm = briefing_limits_for("omlx/qwen")
    llm = briefing_limits_for("openai/gpt-4.1")
    assert llm["max_failures"] > slm["max_failures"]
    assert llm["max_samples"] > slm["max_samples"]


def test_resolve_rewrite_passes_auto_slm_and_llm() -> None:
    assert resolve_rewrite_passes("omlx/qwen3.5-9b") == 3
    assert resolve_rewrite_passes("openai/gpt-4.1") == 2
    assert resolve_rewrite_passes("anthropic/claude-sonnet-4") == 2
    assert resolve_rewrite_passes("omlx/qwen", rewrite_passes=2) == 2
    assert resolve_rewrite_passes("omlx/qwen", slm_multipass=False) == 1
    assert resolve_rewrite_passes("openai/gpt-4.1", llm_multipass=False) == 1
    assert resolve_rewrite_passes("openai/gpt-4.1", multipass=False) == 1
    assert resolve_rewrite_passes("openai/gpt-4.1", llm_default_passes=4) == 4


def test_build_full_optimization_briefing_includes_io_and_params() -> None:
    cfg = PromptRuntimeConfig(
        system_prompt="You are a helpful assistant.",
        model_params={"temperature": 0.2, "max_tokens": 256},
        rag_params={"top_k": 4},
    )
    results = [
        {
            "query": "What is 2+2?",
            "expected": "4",
            "response": "maybe five",
            "score": 0.1,
            "feedback": "wrong arithmetic",
        },
        {
            "query": "Capital of France?",
            "expected": "Paris",
            "response": "Paris",
            "score": 0.95,
        },
    ]
    samples = [{"query": "hi", "expected_answer": "hello"}]
    space = PromptSearchSpace(optimize_model_params=True)
    ctx = OptimizationContext(
        agent_name="demo",
        round_idx=0,
        total_rounds=3,
        baseline_score=0.4,
        current_score=0.5,
        metric_names=["exact_match"],
    )

    briefing = build_full_optimization_briefing(
        current_config=cfg,
        eval_results=results,
        dataset_sample=samples,
        search_space=space,
        context=ctx,
        rewrite_model="openai/gpt-4.1",
        agent_name="demo",
    )

    assert "Optimization briefing" in briefing
    assert "You are a helpful assistant." in briefing
    assert "temperature" in briefing
    assert "What is 2+2?" in briefing
    assert "maybe five" in briefing
    assert "Paris" in briefing
    assert "exact_match" in briefing or "Metrics" in briefing
    assert "demo" in briefing
    assert "openai/gpt-4.1" in briefing
    assert "llm style" in briefing


def test_extract_prompt_text_strips_fence_and_separator() -> None:
    raw = "Some thinking\n---\n```\nBe concise.\n```\n"
    assert extract_prompt_text(raw) == "Be concise."


@pytest.mark.asyncio
async def test_multipass_refine_prompt_calls_llm_three_times_for_slm() -> None:
    calls: list[str] = []

    async def fake_call(model, prompt, **kwargs):  # noqa: ANN001, ANN003
        calls.append(prompt)
        if "CRITIQUE" in prompt:
            return "- Missing keyword FOUR\n"
        return "---\nImproved prompt mentioning FOUR\n"

    with patch(
        "agentomatic.optimize.llm_caller.LLMCaller.call",
        new=AsyncMock(side_effect=fake_call),
    ):
        prompt, notes = await multipass_refine_prompt(
            model="omlx/qwen",
            briefing="# briefing\nquery=2+2 expected=4",
            current_prompt="old",
            passes=3,
        )

    assert prompt == "Improved prompt mentioning FOUR"
    assert len(calls) == 3  # draft, critique, revise
    assert any("DRAFT" in c for c in calls)
    assert any("CRITIQUE" in c for c in calls)
    assert any("REVISE" in c for c in calls)
    assert any("SMALL language model" in c for c in calls)
    assert len(notes) == 3


@pytest.mark.asyncio
async def test_multipass_refine_prompt_two_passes_for_frontier_llm() -> None:
    calls: list[str] = []

    async def fake_call(model, prompt, **kwargs):  # noqa: ANN001, ANN003
        calls.append(prompt)
        return "---\nFrontier revised prompt\n"

    with patch(
        "agentomatic.optimize.llm_caller.LLMCaller.call",
        new=AsyncMock(side_effect=fake_call),
    ):
        prompt, notes = await multipass_refine_prompt(
            model="openai/gpt-4.1",
            briefing="# briefing\nquery=2+2 expected=4",
            current_prompt="old",
            passes=2,
            style="llm",
        )

    assert prompt == "Frontier revised prompt"
    assert len(calls) == 2
    assert any("frontier LLM" in c for c in calls)
    assert any("acceptance criteria" in c for c in calls)
    assert notes[0].endswith("style=llm")


@pytest.mark.asyncio
async def test_rewrite_optimizer_uses_multipass_for_omlx() -> None:
    opt = RewriteOptimizer(model="omlx/Qwen3.5-9B", rewrite_passes=2)
    cfg = PromptRuntimeConfig(system_prompt="baseline prompt")
    results = [
        {
            "query": "q",
            "expected": "yes",
            "response": "no",
            "score": 0.0,
        }
    ]

    async def fake_call(model, prompt, **kwargs):  # noqa: ANN001, ANN003
        return "---\nrewritten with yes\n"

    with patch(
        "agentomatic.optimize.llm_caller.LLMCaller.call",
        new=AsyncMock(side_effect=fake_call),
    ) as mocked:
        cands = await opt.propose(
            current_config=cfg,
            eval_results=results,
            dataset_sample=[{"query": "q", "expected_answer": "yes"}],
            search_space=PromptSearchSpace(optimize_few_shot=False),
            iteration=1,
        )

    assert len(cands) == 1
    assert "yes" in cands[0].config.system_prompt
    assert mocked.await_count == 2  # draft + revise
    assert "2 pass" in cands[0].mutation_notes


@pytest.mark.asyncio
async def test_rewrite_optimizer_auto_two_passes_for_openai() -> None:
    opt = RewriteOptimizer(model="openai/gpt-4.1")
    cfg = PromptRuntimeConfig(system_prompt="baseline prompt")
    results = [{"query": "q", "expected": "yes", "response": "no", "score": 0.0}]

    with patch(
        "agentomatic.optimize.llm_caller.LLMCaller.call",
        new=AsyncMock(return_value="---\nllm rewrite\n"),
    ) as mocked:
        cands = await opt.propose(
            current_config=cfg,
            eval_results=results,
            dataset_sample=[{"query": "q", "expected_answer": "yes"}],
            search_space=PromptSearchSpace(),
            iteration=0,
        )

    assert cands[0].config.system_prompt == "llm rewrite"
    assert mocked.await_count == 2  # auto llm_default_passes=2
    assert any("frontier LLM" in str(c.args[1]) for c in mocked.await_args_list)


def test_resolve_fitter_optimizer_keeps_rewrite_knobs() -> None:
    opt = resolve_fitter_optimizer(
        "rewrite",
        model="openai/gpt-4.1",
        rewrite_passes=4,
        llm_multipass=True,
        llm_default_passes=2,
    )
    assert isinstance(opt, RewriteOptimizer)
    assert opt.rewrite_passes == 4
    assert opt.llm_default_passes == 2

    # GEPA should not explode when rewrite knobs are forwarded
    gepa = resolve_fitter_optimizer(
        "gepa",
        model="openai/gpt-4.1",
        rewrite_passes=3,
        llm_multipass=True,
        n_mutations=2,
    )
    assert gepa.n_mutations == 2
