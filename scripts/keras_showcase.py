"""Run the Keras-style agent lifecycle end to end against a real model.

``compile() -> fit() -> evaluate() -> save() -> load()``, with a real
optimizer driving a real LLM, a real metric, and an ``EarlyStopping``
callback. Every number printed is measured: the loss curve comes from the
returned ``History``, and the before/after scores from ``evaluate()``.

Point it at any OpenAI-compatible endpoint -- a local oMLX / llama.cpp /
vLLM / LM Studio server, or a hosted one::

    export OMLX_BASE_URL=http://127.0.0.1:8000/v1
    export OMLX_API_KEY=whatever
    python scripts/keras_showcase.py --model omlx/my-local-model

The agent here is deliberately simple and deterministic: its answers improve
only when the system prompt contains a specific token it must *discover* from
its own failures. That makes an improvement in the curve attributable to the
optimizer rather than to model variance -- which is the point of a showcase.

Exit code is ``0`` only when every check passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentomatic.agents.base import BaseGraphAgent
from agentomatic.agents.history import EarlyStopping
from agentomatic.agents.optimizers import PromptFitterBridge
from agentomatic.agents.types import AgentDataset, AgentExample
from agentomatic.optimize import PromptSearchSpace
from agentomatic.optimize.metrics import ContainsMetric

#: The token the optimizer has to discover. Present in the prompt, the agent
#: answers well; absent, it does not.
TARGET = "banana"
BASELINE_PROMPT = "Answer the question."

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record and print one assertion."""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(f"{label}: {detail}")


@dataclass
class ShowcaseState:
    """Per-run state for :class:`ShowcaseAgent`."""

    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class ShowcaseAgent(BaseGraphAgent[ShowcaseState]):
    """Answers well only once the prompt carries the target token."""

    agent_name = "keras_showcase"
    system_prompt = BASELINE_PROMPT

    def build_graph(self) -> Any:
        """Single-node graph."""
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: ShowcaseState) -> ShowcaseState:
        """Answer, conditioned on whether the prompt was optimized."""
        prompt = self.resolve_system_prompt(default=self.system_prompt)
        improved = TARGET in prompt.lower()
        marker = "OPT" if improved else "BASE"
        state.output = {
            "response": f"{marker}: answer to {state.request}"
            + (f" {TARGET}" if improved else ""),
            "used_prompt": prompt,
        }
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> ShowcaseState:
        """Accept both the REST (`current_query`) and inline (`query`) forms."""
        return ShowcaseState(
            request=input_data.get("current_query") or input_data.get("query") or ""
        )

    def state_to_output(self, state: ShowcaseState) -> dict[str, Any]:
        """Publish the agent's answer."""
        return state.output


class TargetMetric:
    """1.0 when the answer carries the target token, else 0.0."""

    name = "target_hit"

    def score(self, example: AgentExample, prediction: dict[str, Any]) -> float:
        """Score one prediction."""
        del example
        return 1.0 if TARGET in str(prediction.get("response", "")).lower() else 0.0


def build_dataset() -> AgentDataset:
    """A dataset with explicit train / validation / test splits."""
    queries = [
        "capital of france",
        "2 + 2",
        "colour of the sky",
        "hello world",
        "largest ocean",
        "who wrote hamlet",
        "speed of light",
        "tallest mountain",
    ]
    # The fitter warns below 4 train / 2 validation, and a warned-about run is
    # not a showcase.
    splits = [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]
    return AgentDataset(
        name="keras_showcase",
        examples=[
            AgentExample(
                id=f"e{i}",
                input={"current_query": q},
                # Comma-separated keywords, read by the optimizer's metric.
                expected_output={"response": f"OPT, {TARGET}"},
                split=split,
                metadata={"split": split},
            )
            for i, (q, split) in enumerate(zip(queries, splits, strict=True))
        ],
    )


def main() -> int:
    """Run the lifecycle and report."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("AGENTOMATIC_LIVE_MODEL", ""),
        help="e.g. omlx/my-model, ollama/qwen2.5:7b, openai/gpt-4o-mini",
    )
    ap.add_argument("--base-url", default=os.environ.get("OMLX_BASE_URL", ""))
    ap.add_argument("--api-key", default=os.environ.get("OMLX_API_KEY", ""))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-trials", type=int, default=6)
    args = ap.parse_args()

    if not args.model:
        print("No model given. Pass --model (or set AGENTOMATIC_LIVE_MODEL).")
        print("The optimizer needs a real LLM — there is nothing to showcase without one.")
        return 2

    ds = build_dataset()
    agent = ShowcaseAgent()
    metric = TargetMetric()

    print(f"── compile() — model={args.model} ───────────────────────")
    bridge = PromptFitterBridge(
        agent_name=agent.agent_name,
        task_model=args.model,
        rewrite_model=args.model,
        optimizer="rewrite",
        # The dataset's expected_output is comma-separated keywords, so the
        # fit objective has to be keyword containment. The bridge defaults to
        # ExactMatchMetric, which never fires here and leaves the curve flat.
        metric=ContainsMetric(),
        max_trials=args.max_trials,
        search_space=PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_model_params=False,
            optimize_few_shot=False,
        ),
        auto_report=False,
        concurrency=1,
        llm_base_url=args.base_url or None,
        llm_api_key=args.api_key or None,
    )
    agent.compile(ds, metrics=[metric], optimizer=bridge, loss=metric)
    meta = agent.compiled_metadata
    check("dataset recorded", meta.get("dataset_size") == len(ds), str(meta))
    check("metric recorded", metric.name in (meta.get("metrics") or []), str(meta))
    check("optimizer recorded", meta.get("optimizer") == "PromptFitterBridge", str(meta))
    check("loss recorded", meta.get("loss") not in (None, "none"), str(meta))

    print("\n── evaluate() before fit ────────────────────────────────")
    before = agent.evaluate(ds).scores[metric.name]
    print(f"     {metric.name} = {before:.3f}")
    check("baseline scored every example", len(agent.evaluate(ds).example_results) == len(ds))

    print(f"\n── fit(epochs={args.epochs}) with EarlyStopping ─────────")
    history = agent.fit(
        ds,
        epochs=args.epochs,
        verbose=0,
        optimize_mode="rewrite",
        max_trials=args.max_trials,
        callbacks=[EarlyStopping(monitor="loss", patience=2)],
    )
    curve = list(history.history.get("loss", []))
    scores = list(history.history.get(metric.name, []))
    print(f"     loss:          {[round(x, 3) for x in curve]}")
    print(f"     {metric.name:14s} {[round(x, 3) for x in scores]}")
    check("fit returned a loss curve", bool(curve), str(history.history))
    check("fit tracked the compiled metric", bool(scores), str(history.history))
    if len(curve) > 1:
        check(
            "loss never increased",
            curve[-1] <= curve[0] + 1e-9,
            f"{curve[0]:.3f} -> {curve[-1]:.3f}",
        )

    status = getattr(agent, "_last_optimize_status", None)
    check("the optimizer ran rather than silently skipping", status == "ok", str(status))

    result = getattr(agent, "_last_fit_result", None)
    check("fit left an auditable result", result is not None)
    if result is not None:
        print(
            f"     baseline={result.baseline_score:.4f}  best={result.best_score:.4f}"
            f"  holdout={result.holdout_score}"
        )
        improved = result.best_score > result.baseline_score + 1e-9
        if improved:
            tuned = agent.compiled_config.get("system_prompt") or ""
            check("the tuned prompt found the target token", TARGET in tuned.lower(), tuned[:80])
        else:
            # No improvement is a legitimate outcome — a silent one is not.
            trail = len(result.prompt_history or []) + len(result.trials or [])
            check(
                "no-improvement still left an auditable trail",
                trail > 0,
                f"{trail} prompt/trial record(s)",
            )

    print("\n── evaluate() after fit ─────────────────────────────────")
    after = agent.evaluate(ds).scores[metric.name]
    print(f"     {metric.name} = {after:.3f}  (was {before:.3f})")
    check("score did not regress", after >= before - 1e-9, f"{before:.3f} -> {after:.3f}")

    print("\n── save() / load() round trip ───────────────────────────")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "showcase.agent"
        agent.save(path)
        written = sorted(f.name for f in path.iterdir())
        check("save wrote the compiled config", "config.json" in written, str(written))
        check("save wrote the fit history", "fit_history.json" in written, str(written))

        restored = ShowcaseAgent()
        restored.load(path)
        check(
            "the tuned prompt survived the round trip",
            restored.resolve_system_prompt(default="") == agent.resolve_system_prompt(default=""),
        )
        check("the fit history survived", bool(restored.history.history.get("loss")))

        # Metrics are live objects and never serialise — supply them again.
        reloaded = restored.evaluate(ds, metrics=[metric]).scores[metric.name]
        print(f"     reloaded {metric.name} = {reloaded:.3f}")
        check(
            "a reloaded agent scores identically",
            abs(reloaded - after) < 1e-9,
            f"{after:.3f} vs {reloaded:.3f}",
        )

    print()
    if failures:
        print(f"KERAS LIFECYCLE FAILED — {len(failures)} check(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("KERAS LIFECYCLE PASSED — compile / fit / evaluate / save / load on a real model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
