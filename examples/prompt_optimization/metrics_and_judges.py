#!/usr/bin/env python3
"""Metrics, LLM judges and losses — what each returns and where it plugs in.

No optimization here. One question, three candidate answers (good, partial,
vague), scored by every kind of metric so you can see the numbers, the
per-dimension breakdown and the judge's rationale before choosing what to
optimize for.

The two metric families (every optimization API accepts either — they are
adapted to each other automatically):

=====================================  =======================================
``agentomatic.agents`` (class agents)  ``agentomatic.optimize`` (the fitter)
=====================================  =======================================
sync ``score(example, prediction)``    async ``evaluate(query, response,
→ ``float`` in [0, 1]                  expected, context)`` → ``EvalResult``
sees the whole ``AgentExample``        (``score``, ``reason``, ``metadata``,
(metadata!) and the output dict        ``failed``)
=====================================  =======================================

Where metrics go:

* ``agent.compile(metrics=[...])``   — REPORTED every epoch (``val_*`` too)
* ``agent.compile(loss=...)``        — the number ``fit()``/EarlyStopping minimise
* ``PromptFitterBridge(metric=...)`` / ``PromptFitter.fit(..., metric)``
                                     — the objective candidates are SELECTED on

Run::

    python examples/prompt_optimization/metrics_and_judges.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    build_parser,
    grounded_score,
    must_include_score,
    settings_from,
    setup_logging,
)


async def run(argv: list[str] | None = None) -> int:
    """Score three answers with every metric family."""
    args = build_parser(__doc__.splitlines()[0]).parse_args(argv)
    settings = settings_from(args)
    setup_logging(settings.log_level)

    # ------------------------------------------------------------------
    # 0. One example, three answers
    # ------------------------------------------------------------------
    # A row carries more than a question and an answer — every field below
    # reaches the metrics, and the LLM judges and the optimizer see it too:
    #   input.current_query       the question
    #   input.context.documents   what the answer must come from (RAG)
    #   input.customer_plan       any other agent input, exactly as written
    #   metadata                  labels (here the facts the answer must contain)
    #   tags                      grouping labels
    #   rubric                    per-dimension criteria for the judge
    from agentomatic.agents import AgentExample

    example = AgentExample(
        id="refund_1",
        input={
            "current_query": "Can I get a refund on my monthly plan?",
            "context": {
                "documents": [
                    {
                        "source": "refunds.md",
                        "content": "Refunds are available within 30 days of purchase for "
                        "annual plans only.",
                    }
                ]
            },
            "customer_plan": "monthly",
        },
        expected_output={
            "response": "No. Refunds are only available for annual plans, within 30 days."
        },
        metadata={"must_include": ["30 days", "annual"], "topic": "refund"},
        tags=["billing", "refund"],
        rubric={"policy": "Applies the refund rule to the customer's plan."},
    )
    query = example.input["current_query"]
    expected = example.expected_output["response"]
    # What a judge gets as context documents (``DataPoint.context``):
    # "Refunds are available within 30 days … (source: refunds.md)".
    documents = example.to_datapoint().context
    answers = {
        "good": "No — refunds are only available for annual plans, within 30 days of purchase.",
        "partial": "Refunds are possible within 30 days.",
        "vague": "Thanks for reaching out! Our team is happy to help with that.",
    }

    def show(title: str, scores: dict[str, float | str]) -> None:
        cells = "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in scores.items()
        )
        print(f"  {title:<8} {cells}")

    # ------------------------------------------------------------------
    # 1. Class-agent metrics — score(example, prediction) -> float
    # ------------------------------------------------------------------
    # ContainsTermsMetric(terms, case_sensitive=False)  fraction of terms present
    # ExactKeyMatchMetric(required_keys)                structured-output keys present
    # ResponseSimilarityMetric(fuzzy=True)              fuzzy match vs expected_output
    # CallableMetric(name, fn(example, prediction))     anything you can compute
    # agents.WeightedMetric([(name, metric, weight)])   weighted blend (any family)
    from agentomatic.agents import (
        CallableMetric,
        ContainsTermsMetric,
        ExactKeyMatchMetric,
        ResponseSimilarityMetric,
    )
    from agentomatic.agents import WeightedMetric as AgentWeightedMetric

    print("\n── 1. class-agent metrics (sync, see the whole example) ──")
    terms = ContainsTermsMetric(["30 days", "annual"], name="terms")
    keys = ExactKeyMatchMetric(["response"], name="keys")
    similarity = ResponseSimilarityMetric(name="similarity")
    facts = CallableMetric("facts", must_include_score)  # reads example.metadata
    grounded = CallableMetric("grounded", grounded_score)  # reads input.context.documents

    def plan_rule(ex: AgentExample, prediction: dict) -> float:
        """A monthly-plan refund question must be answered with a no."""
        if ex.input.get("customer_plan") != "monthly":  # any input, as written
            return 1.0
        return 1.0 if re.search(r"\b(no|not)\b", prediction["response"].lower()) else 0.0

    plan = CallableMetric("plan", plan_rule)
    blend = AgentWeightedMetric([("facts", facts, 0.7), ("similarity", similarity, 0.3)])
    for label, text in answers.items():
        prediction = {"response": text}
        show(
            f"{label}",
            {
                m.name: m.score(example, prediction)
                for m in (terms, keys, similarity, facts, grounded, plan, blend)
            },
        )

    # ------------------------------------------------------------------
    # 2. Optimize metrics — await evaluate(query, response, expected, context)
    # ------------------------------------------------------------------
    # ExactMatchMetric(fuzzy=True, threshold=0.8)       fuzzy exact match
    # ContainsMetric()                                   comma-separated keywords in expected
    # DeterministicMetric(checks=[{"type": ..., "value": ...}])
    #     types: contains | not_contains | regex | min_length | max_length |
    #            json_valid | starts_with            (score = fraction passed)
    # CustomMetric(fn(query, response, expected, context) -> float, name=...)
    # LatencyMetric(target_seconds, max_seconds)        uses the measured run time
    #                                                   (inside fit/evaluate only)
    # CostMetric(...)                                    token cost term
    # GEvalMetric / DeepEvalMetric / RedTeamMetric       need `pip install deepeval`
    from agentomatic.optimize import (
        ContainsMetric,
        CustomMetric,
        DeterministicMetric,
        ExactMatchMetric,
    )

    print("\n── 2. optimize metrics (async, text in → EvalResult out) ──")
    exact = ExactMatchMetric()
    contains = ContainsMetric()  # expected is read as "kw1, kw2, …"
    fmt = DeterministicMetric(
        name="format",
        checks=[
            {"type": "max_length", "value": 200},
            {"type": "not_contains", "value": "happy to help"},
        ],
    )
    brevity = CustomMetric(lambda q, r, e, c: 1.0 if len(r.split()) <= 25 else 0.5, name="brevity")
    for label, text in answers.items():
        results = {
            m.name: await m.evaluate(
                query, text, expected if m is not contains else "30 days, annual"
            )
            for m in (exact, contains, fmt, brevity)
        }
        show(label, {k: r.score for k, r in results.items()})

    # ------------------------------------------------------------------
    # 3. LLM-as-judge
    # ------------------------------------------------------------------
    # LLMJudgeMetric(criteria, model, temperature=0.0, base_url=, api_key=)
    #     one overall score + reason
    # LocalJudgeMetric(name, model, criteria, dimensions=[...], temperature=0.0,
    #                  base_url=, api_key=)
    #     overall score + per-dimension scores + rationale (reported per example)
    # MultiJudgePanel(judges=[LocalJudgeMetric, ...], aggregation=
    #                 "average"|"median"|"min"|"max", weights=[...])
    #     several judges/models → one robust score ("min" = strictest judge)
    # Judges score the EXPECTED answer as reference; keep temperature=0.0 so the
    # same answer gets the same score in every epoch. A judge that cannot be
    # reached or parsed returns ``failed=True`` and score 0 — never a made-up 0.5.
    # Pass the documents as ``context`` (fit()/evaluate() do it for you). Inside
    # fit()/evaluate() a judge is ALSO shown the example's other inputs
    # (customer_plan), metadata and tags, and the rubric is part of the
    # expected reference — see rag_context.py.
    from agentomatic.optimize import LLMJudgeMetric, LocalJudgeMetric, MultiJudgePanel

    print(f"\n── 3. LLM judges ({settings.judge_model}) ──")
    criteria = "Is the answer correct per the reference, complete and concise? Score 0-1."
    single = LLMJudgeMetric(criteria=criteria, model=settings.judge_spec)
    judge = LocalJudgeMetric(
        name="judge",
        model=settings.judge_spec,
        criteria=criteria,
        dimensions=["correctness", "completeness", "concision"],
    )
    strict = LocalJudgeMetric(name="strict", model=settings.judge_spec, criteria=criteria)
    panel = MultiJudgePanel(judges=[judge, strict], aggregation="median", name="panel")
    for label, text in answers.items():
        overall = await single.evaluate(query, text, expected, documents)
        detailed = await judge.evaluate(query, text, expected, documents)
        panelled = await panel.evaluate(query, text, expected, documents)
        show(
            label,
            {
                "llm_judge": overall.score,
                "judge": detailed.score,
                **{
                    f"judge.{k}": float(v)
                    for k, v in (detailed.metadata.get("dimensions") or {}).items()
                },
                "panel": panelled.score,
            },
        )
        print(f"  {'':<8} why: {' '.join(str(detailed.reason).split())[:100]}")

    # ------------------------------------------------------------------
    # 4. Composite objective — CompositeMetric([WeightedMetric(...)])
    # ------------------------------------------------------------------
    # Weights are normalised; each component is reported as a dimension; a
    # component that fails scores 0 and ≥ half the weight failing marks the
    # whole result failed (so a dead judge cannot look like a bad prompt).
    # Components may be of EITHER family. Score it like any metric:
    #   await composite.evaluate(query, response, expected)   text only
    #   composite.score(example, prediction)                  whole example —
    #       metadata-reading components (``facts``) need this form, which is
    #       also how fit() and agent.evaluate() call it.
    from agentomatic.agents import OptimizeMetricAdapter
    from agentomatic.optimize import CompositeMetric, WeightedMetric

    print("\n── 4. composite objective (judge 0.6 + facts 0.4) ──")
    objective = CompositeMetric(
        [
            WeightedMetric(name="judge", metric=judge, weight=0.6),
            WeightedMetric(name="facts", metric=facts, weight=0.4),  # class-agent metric
        ],
        name="objective",
    )
    scorer = OptimizeMetricAdapter(objective)  # keeps the full result on .last_result
    for label, text in answers.items():
        value = scorer.score(example, {"response": text})
        dims = (scorer.last_result.metadata or {}).get("dimensions", {})
        show(label, {"objective": value, **dims})

    # ------------------------------------------------------------------
    # 5. Crossing the families + losses
    # ------------------------------------------------------------------
    # as_optimize_metric(m)        class-agent metric → evaluate(...)
    # OptimizeMetricAdapter(m)     optimize metric → score(example, prediction)
    #                              (also exposes evaluate(); .failures counts errors)
    # MetricLoss(metric)           loss = 1 − score   (any family)
    # CallableLoss(fn, name=...)   custom loss(example, prediction)
    from agentomatic.agents import MetricLoss
    from agentomatic.optimize import as_optimize_metric

    print("\n── 5. adapters and losses ──")
    as_eval = await as_optimize_metric(terms).evaluate(query, answers["partial"], expected)
    judge_as_score = OptimizeMetricAdapter(judge, name="judge")
    loss = MetricLoss(judge_as_score)
    show(
        "partial",
        {
            "terms via evaluate()": as_eval.score,
            "judge via score()": judge_as_score.score(example, {"response": answers["partial"]}),
            "loss (1 − judge)": loss.compute(example, {"response": answers["partial"]}),
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Synchronous entry point."""
    return asyncio.run(run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
