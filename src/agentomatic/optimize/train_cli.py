"""Auto-generate per-agent training CLIs.

Provides :func:`create_train_cli` which produces a fully-featured
Click CLI for any agent, with commands for training, evaluation,
experiment listing, model listing, and more.

The agent just needs a ``train.py`` like::

    # agents/my_agent/train.py
    from agentomatic.optimize.train_cli import create_train_cli
    from agents.my_agent.agent import agent
    from agents.my_agent.evals import get_test_cases

    cli = create_train_cli(agent=agent, test_cases_fn=get_test_cases)

    if __name__ == "__main__":
        cli()
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


def create_train_cli(
    agent: Any,
    test_cases_fn: Callable[[], list[Any]] | None = None,
    description: str = "Train and evaluate an agent.",
) -> Any:
    """Create a Click CLI for agent training and evaluation.

    Returns a Click command group with subcommands:
    - ``train`` — run prompt optimisation
    - ``evaluate`` — evaluate without optimising
    - ``experiments`` — list past experiments
    - ``list-models`` — show available model options
    - ``generate`` — generate synthetic test cases

    Args:
        agent: The agent instance to train/evaluate.
        test_cases_fn: A callable returning a list of test cases.
        description: CLI description.

    Returns:
        A Click group that can be called as ``cli()``.
    """
    try:
        import click
    except ImportError:
        raise ImportError(
            "click is required for create_train_cli(). Install with: pip install click"
        ) from None

    @click.group(name="train", help=description)
    def cli() -> None:
        pass

    @cli.command(name="train")
    @click.option(
        "-m", "--model", default="ollama/mistral:7b", help="Model for rewriting (LiteLLM format)"
    )
    @click.option("-i", "--iterations", default=5, help="Max optimisation rounds")
    @click.option("-t", "--target", default=0.85, type=float, help="Target composite score")
    @click.option(
        "-s",
        "--strategy",
        default="iterative_refinement",
        type=click.Choice(["iterative_refinement", "bootstrap_few_shot", "mipro", "combined"]),
        help="Optimisation strategy",
    )
    @click.option("--quick", is_flag=True, help="Fast 2-iteration run")
    @click.option("--augment", is_flag=True, help="Enable data augmentation")
    @click.option(
        "--generate", "gen_count", default=0, type=int, help="Generate N synthetic test cases"
    )
    @click.option("--verbose/--quiet", default=True, help="Show progress")
    @click.option("--output", default="optimization_results", help="Output directory")
    def train_cmd(
        model: str,
        iterations: int,
        target: float,
        strategy: str,
        quick: bool,
        augment: bool,
        gen_count: int,
        verbose: bool,
        output: str,
    ) -> None:
        """Run prompt optimisation."""
        if quick:
            iterations = 2
            strategy = "bootstrap_few_shot"

        if test_cases_fn is None:
            click.echo("❌ No test_cases_fn provided — cannot train without test data.")
            return

        test_cases = test_cases_fn()
        if gen_count > 0:
            test_cases = _generate_cases(agent, gen_count, model, test_cases)
        if augment:
            test_cases = _augment_cases(test_cases, model)

        click.echo(f"\n🧠 Training agent: {getattr(agent, 'agent_name', 'unknown')}")
        click.echo(f"   Model: {model}")
        click.echo(f"   Rounds: {iterations}")
        click.echo(f"   Strategy: {strategy}")
        click.echo(f"   Target: {target}")
        click.echo(f"   Test cases: {len(test_cases)}")
        click.echo()

        result = asyncio.run(
            _do_train(
                agent=agent,
                test_cases=test_cases,
                model=model,
                iterations=iterations,
                target=target,
                strategy=strategy,
                verbose=verbose,
                output_dir=output,
            )
        )

        click.echo(f"\n{result.summary()}")

    @cli.command(name="evaluate")
    @click.option("-m", "--model", default="ollama/mistral:7b", help="Evaluation model")
    @click.option("--verbose/--quiet", default=True)
    def evaluate_cmd(model: str, verbose: bool) -> None:
        """Evaluate agent without optimising."""
        from agentomatic.optimize.agent_detect import Evaluator

        if test_cases_fn is None:
            click.echo("❌ No test_cases_fn provided.")
            return

        test_cases = test_cases_fn()
        evaluator = Evaluator.for_agent(agent)
        click.echo(f"\n🔍 Evaluating agent: {getattr(agent, 'agent_name', 'unknown')}")
        click.echo(f"   Type: {evaluator.agent_type.value}")
        click.echo(f"   Metrics: {evaluator.metrics}")
        click.echo(f"   Test cases: {len(test_cases)}")
        click.echo()

        summary = asyncio.run(evaluator.evaluate(agent, test_cases, model=model))
        mean = float(summary.get("mean_score", 0.0))
        passed = bool(summary.get("passed", False))
        if verbose:
            dims = summary.get("dimensions") or {}
            for name, value in dims.items():
                click.echo(f"   {name}: {float(value):.4f}")
        click.echo(f"\n   Mean score: {mean:.4f} (threshold={evaluator.threshold})")
        click.echo("✅ Evaluation passed" if passed else "❌ Evaluation below threshold")

    @cli.command(name="experiments")
    @click.option("--agent-name", default=None, help="Filter by agent")
    @click.option("-n", "--limit", default=10, help="Max rows")
    def experiments_cmd(agent_name: str | None, limit: int) -> None:
        """View past optimisation experiments."""
        from agentomatic.optimize.experiment_tracker import get_tracker

        tracker = get_tracker()
        tracker.display_experiments(agent_name=agent_name, limit=limit)

    @cli.command(name="list-models")
    def list_models_cmd() -> None:
        """Show available models and presets."""
        click.echo("\n📋 Available models:\n")
        click.echo("  Local (Ollama):")
        click.echo("    ollama/mistral:7b")
        click.echo("    ollama/llama3:8b")
        click.echo("    ollama/qwen2.5:7b")
        click.echo("    ollama/llama3:70b")
        click.echo()
        click.echo("  Cloud:")
        click.echo("    openai/gpt-4o")
        click.echo("    openai/gpt-4o-mini")
        click.echo("    gemini/gemini-1.5-flash")
        click.echo("    gemini/gemini-1.5-pro")
        click.echo("    anthropic/claude-3-5-sonnet")
        click.echo()
        click.echo("  Presets:")
        click.echo("    --quick         2-round bootstrap (fast)")
        click.echo("    --model X -i N   custom model + iterations")

    @cli.command(name="generate")
    @click.option("-n", "--num", default=10, help="Number of cases to generate")
    @click.option("-m", "--model", default="ollama/mistral:7b", help="Generation model")
    def generate_cmd(num: int, model: str) -> None:
        """Generate synthetic test cases."""
        click.echo(f"\n📝 Generating {num} test cases with {model}...")
        _generate_cases(agent, num, model)
        click.echo(f"✅ Generated {num} cases")

    @cli.command(name="settings")
    def settings_cmd() -> None:
        """Show current optimisation settings."""
        from agentomatic.optimize.presets import Presets
        from agentomatic.optimize.settings import show_available_options

        click.echo("\n⚙️  Available options:\n")
        show_available_options()
        click.echo("\n📦 Presets:\n")
        Presets.display()

    return cli


# =====================================================================
# Internal helpers
# =====================================================================


def _case_query(case: Any) -> str:
    return str(getattr(case, "input", None) or getattr(case, "query", None) or "")


def _case_expected(case: Any) -> str:
    return str(
        getattr(case, "expected_output", None) or getattr(case, "expected_answer", None) or ""
    )


async def _do_train(
    agent: Any,
    test_cases: list[Any],
    model: str,
    iterations: int,
    target: float,
    strategy: str,
    verbose: bool,
    output_dir: str,
) -> Any:
    """Run the full training pipeline."""
    from agentomatic.optimize.callbacks import ScoreThreshold, default_callbacks
    from agentomatic.optimize.dataset import DataPoint, Dataset
    from agentomatic.optimize.fitter import PromptFitter
    from agentomatic.optimize.metrics import LLMJudgeMetric
    from agentomatic.optimize.optimizer_mixin import FitResult
    from agentomatic.optimize.presets import Preset, to_fitter_kwargs

    del verbose  # reserved for future CLI verbosity; fitter uses loguru
    agent_name = getattr(agent, "agent_name", "unknown")

    points = [
        DataPoint(
            query=_case_query(case),
            expected_answer=_case_expected(case),
            context=list(getattr(case, "context", []) or []),
        )
        for case in test_cases
    ]
    dataset = Dataset(points=points)

    metric = LLMJudgeMetric(
        name="composite",
        criteria="Evaluate whether the response is accurate, helpful, and complete.",
        model=model,
    )

    preset = Preset(
        name="train_cli",
        description="create_train_cli ad-hoc preset",
        model=model,
        max_iterations=iterations,
        strategy=strategy,
        target_score=target,
        parallel_evals=1,
        temperature=0.7,
        verbose=True,
    )
    fitter_kwargs = to_fitter_kwargs(preset)
    cbs = default_callbacks(patience=max(1, iterations // 2), target_score=target)
    if not any(isinstance(c, ScoreThreshold) for c in cbs):
        cbs.append(ScoreThreshold(threshold=target))
    fitter_kwargs["callbacks"] = cbs
    fitter_kwargs["experiment_dir"] = output_dir

    fitter = PromptFitter(
        agent=agent_name,
        local_agent=agent,
        **fitter_kwargs,
    )

    result = await fitter.fit(dataset, dataset, metric)
    wrapped = FitResult.from_prompt_fit_result(result)
    wrapped.strategy = strategy
    wrapped.model = model
    return wrapped


def _generate_cases(
    agent: Any,
    num: int,
    model: str = "ollama/mistral:7b",
    existing: list[Any] | None = None,
) -> list[Any]:
    """Generate synthetic test cases for an agent."""
    try:
        return asyncio.run(_generate_cases_async(agent, num, model, existing))
    except Exception as exc:
        from loguru import logger

        logger.warning("Synthetic case generation failed: {}", exc)
        return list(existing or [])


async def _generate_cases_async(
    agent: Any,
    num: int,
    model: str,
    existing: list[Any] | None = None,
) -> list[Any]:
    from agentomatic.optimize.synthesizer import DataSynthesizer

    agent_name = getattr(agent, "agent_name", "agent")
    desc = getattr(agent, "agent_description", "") or f"An AI agent named {agent_name}"

    synth = DataSynthesizer(model=model)
    dataset = await synth.generate(description=desc, n_samples=num)
    return list(existing or []) + list(dataset.points)


def _augment_cases(cases: list[Any], model: str = "ollama/mistral:7b") -> list[Any]:
    """Augment existing test cases."""
    try:
        return asyncio.run(_augment_cases_async(cases, model))
    except Exception as exc:
        from loguru import logger

        logger.warning("Dataset augmentation failed: {}", exc)
        return cases


async def _augment_cases_async(cases: list[Any], model: str) -> list[Any]:
    from agentomatic.optimize.dataset import DataPoint, Dataset
    from agentomatic.optimize.synthesizer import DataSynthesizer

    points = [
        DataPoint(query=_case_query(case), expected_answer=_case_expected(case)) for case in cases
    ]
    dataset = Dataset(points=points)

    synth = DataSynthesizer(model=model)
    aug_dataset = await synth.augment(
        dataset=dataset,
        strategies=["paraphrase", "edge_case"],
        multiplier=2,
    )
    return list(aug_dataset.points)
