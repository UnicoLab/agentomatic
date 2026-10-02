# Metrics, judges & loss

A metric turns one answer into a number in `[0, 1]`. Optimization is only as
good as that number: if it rewards the wrong thing, the optimizer will find
prompts that do the wrong thing well. This page covers every metric, how LLM
judges work, how to combine them, and where each one plugs in.

Runnable companion: [`metrics_and_judges.py`](examples.md) scores three fixed
answers with every metric below.

## Three slots, one number each

| Slot | Set with | Used for |
| --- | --- | --- |
| **Reported metrics** | `agent.compile(metrics=[...])` | Computed on train and validation every epoch (`judge`, `val_judge`, …), shown in `History` and the report. They do not steer anything. |
| **Loss** | `agent.compile(loss=...)` | The number `fit()` records as `loss` / `val_loss` and `EarlyStopping` monitors. `MetricLoss(metric)` = `1 − score`. |
| **Objective** | `PromptFitterBridge(metric=...)` or `PromptFitter.fit(..., metric)` | What candidate prompts are **selected** on. |

Use the same underlying measure for the loss and the objective (for example
`quality` for both, as in the [overview](index.md#a-complete-run-in-40-lines)),
so the epoch curve and the selection agree. Without `metric=`, the bridge uses
your compiled loss's metric, else the first compiled metric.

## Two metric families

Every API accepts either family; they are adapted to each other
automatically.

| | `agentomatic.agents` (class agents) | `agentomatic.optimize` (the fitter) |
| --- | --- | --- |
| Signature | `score(example, prediction) -> float` (sync) | `await evaluate(query, response, expected, context) -> EvalResult` |
| Sees | The whole `AgentExample` (input incl. context documents and other inputs, expected output, **metadata**, **tags**, rubric) and the output dict, the same in `fit()` and `evaluate()` | The question, the answer text, the reference text, the context documents (or what the agent retrieved). Inside `fit()` / `evaluate()`, `current_scoring_run()` also exposes the example |
| Returns | A float | `EvalResult(score, reason, metadata)` — judges add `dimensions`, rationale; failures set `metadata["evaluation_failed"]` |
| Best for | Checks that need structured output or per-example metadata | Judges and text checks |

Adapters, if you need to cross explicitly:

```python
from agentomatic.agents import OptimizeMetricAdapter, as_agent_metric
from agentomatic.optimize import as_optimize_metric

as_optimize_metric(ContainsTermsMetric(["30 days"]))    # class metric → evaluate(...)
OptimizeMetricAdapter(judge, name="judge")               # optimize metric → score(...), evaluate(...)
as_agent_metric(judge)                                   # same, idempotent
```

Inside `fit()`, `evaluate()` and composite metrics, a class-agent metric
always receives the example's `metadata` and the agent's structured output,
whichever family it is wrapped in.

## Deterministic metrics

Fast, free, reproducible. Prefer them wherever the answer has checkable facts
or format.

| Metric | Family | Options | Scores |
| --- | --- | --- | --- |
| `ContainsTermsMetric(required_terms, case_sensitive=False, name=)` | agents | fixed term list | fraction of terms present in `response` |
| `ExactKeyMatchMetric(required_keys, name=)` | agents | output keys | fraction of keys present in the output dict |
| `ResponseSimilarityMetric(name=, fuzzy=True)` | agents | | fuzzy similarity to `expected_output["response"]` |
| `CallableMetric(name, fn)` | agents | `fn(example, prediction) -> float` | anything — e.g. `example.metadata["must_include"]` coverage |
| `ExactMatchMetric(fuzzy=True, threshold=0.8)` | optimize | | fuzzy exact match vs the reference |
| `ContainsMetric()` | optimize | reference = `"kw1, kw2"` | fraction of comma-separated keywords present |
| `DeterministicMetric(name=, checks=[...])` | optimize | check `type`: `contains`, `not_contains`, `regex`, `min_length`, `max_length`, `json_valid`, `starts_with` | fraction of checks passed |
| `CustomMetric(fn, name=)` | optimize | `fn(query, response, expected, context) -> float` | clamped to `[0, 1]`; `NaN` counts as a failure |
| `LatencyMetric(target_seconds=2.0, max_seconds=10.0)` | optimize | | 1.0 under target, linear to 0 at max; uses the measured run time (fails outside a fit / evaluate run rather than guessing) |
| `CostMetric(target_tokens=500, max_tokens=3000)` | optimize | | token-usage penalty |

```python
from agentomatic.agents import CallableMetric

def must_include(example, prediction):
    facts = example.metadata.get("must_include") or []
    answer = prediction.get("response", "").lower()
    return sum(f.lower() in answer for f in facts) / len(facts) if facts else 0.0

facts = CallableMetric("facts", must_include)
```

## LLM-as-judge

A judge asks a model to grade the answer against the reference. Use one when
quality is open-ended (tone, completeness, grounding) and keep a deterministic
metric next to it.

| Judge | Options | Returns |
| --- | --- | --- |
| `LLMJudgeMetric(criteria, model, name="llm_judge", temperature=0.0, base_url=, api_key=)` | a rubric | one overall score + reason |
| `LocalJudgeMetric(name="local_judge", model, criteria, dimensions=[...], weight=1.0, temperature=0.0, base_url=, api_key=)` | a rubric + named dimensions | overall score, a score per dimension, feedback, motivation, what worked / failed, improvement hints |
| `MultiJudgePanel(judges=[...], aggregation="average", weights=None, name=)` | several `LocalJudgeMetric`s (different models or rubrics) | one aggregated score + per-judge scores |
| `GEvalMetric`, `DeepEvalMetric`, `RedTeamMetric` | DeepEval metrics (`pip install deepeval`) | DeepEval scores, oriented so higher is better |

```python
from agentomatic.optimize import LocalJudgeMetric, MultiJudgePanel

judge = LocalJudgeMetric(
    name="judge",
    model="omlx/Qwen3.5-9B-MLX-4bit",            # "provider/model"
    criteria="Is the answer correct per the reference, grounded in the policy, "
             "complete and concise? Score 0-1.",
    dimensions=["correctness", "completeness", "concision"],
    temperature=0.0,                             # same answer → same score every epoch
)

panel = MultiJudgePanel(
    judges=[
        judge,
        LocalJudgeMetric(name="strict", model="openai/gpt-4.1-mini",
                         criteria="Penalise any claim not in the reference."),
    ],
    aggregation="median",     # "average" (weighted) | "median" | "min" | "max"
    weights=None,             # per judge, for "average"
)
```

How judges stay honest:

* **Reference.** Judges receive the example's expected answer (and, for
  `AgentExample`s, a rich reference: expected output, rubric, required facts).
  Write expected answers that state the facts a correct answer must contain.
* **Context.** Judges receive the row's context documents. If the row has
  none, they get what the agent says it retrieved (`retrieval_context` /
  `citations` / `sources` in its output). Inside `fit()` and `evaluate()`
  they also get the row's other inputs (a plan, a locale…), its metadata and
  its tags, in an "Example inputs, metadata and tags" section, so an answer
  is judged in that light. See [what each stage sees](data.md#what-each-stage-sees).
* **Deterministic.** Keep `temperature=0.0`. A judge that scores the same
  answer differently each time makes every comparison noise; the
  [significance rule](generalization.md#the-acceptance-rules) exists because
  judges are noisy anyway.
* **Failures are failures.** An unreachable judge or an unparsable reply
  returns `score=0.0` with `metadata["evaluation_failed"] = True`, never a
  made-up 0.5. The fitter counts failed examples as 0 for every candidate,
  logs them, and flags an "evaluation blackout" when most calls fail.
* **Parsing.** Replies are parsed tolerantly (JSON in prose or code fences,
  `score`/`overall_score`/`rating`, 0–10 or 0–100 scales normalised).
* **Panels.** `"median"` is robust to one outlier judge; `"min"` is the
  strictest; failed judges are dropped, and the panel fails only if all do.
* **Endpoint.** `model="omlx/…"` routes to `base_url` (or the process
  default set by `LLMCaller.configure(base_url=...)` / `OMLX_BASE_URL`).
  `openai/…`, `ollama/…`, `gemini/…` and `litellm/…` are the other providers.

## Combining metrics

Two equivalent tools; pick by family:

```python
# optimize family — for PromptFitter
from agentomatic.optimize import CompositeMetric, WeightedMetric

objective = CompositeMetric(
    [
        WeightedMetric(name="judge", metric=judge, weight=0.6),
        WeightedMetric(name="facts", metric=facts, weight=0.4),   # either family
    ],
    name="objective",
)

# agents family — for compile(metrics=..., loss=...) and PromptFitterBridge
from agentomatic.agents import WeightedMetric as AgentWeightedMetric

quality = AgentWeightedMetric([("judge", judge, 0.6), ("facts", facts, 0.4)], name="quality")
```

* Weights are normalised; negative weights are rejected.
* Each component is reported as a **dimension**, in the report and in
  `EvalResult.metadata["dimensions"]`.
* A failed component scores 0 (the denominator stays fixed, so a dead judge
  lowers the score instead of silently re-weighting the rest); the composite
  is marked failed when at least half its weight failed.
* `CompositeMetric` also has `score(example, prediction)`, so it can sit in
  `compile(metrics=[...])` directly.

## Losses

| Loss | Meaning |
| --- | --- |
| `MetricLoss(metric, name=None)` | `1 − metric.score(...)`; accepts either family |
| `CallableLoss(fn, name="loss")` | `fn(example, prediction) -> float`, lower is better |
| any object with `compute(example, prediction) -> float` | custom |

`fit()` averages the loss over train (`loss`) and validation (`val_loss`)
after every epoch, plus a baseline row before the first epoch.
`agents.EarlyStopping(monitor="val_loss")` stops when it stops improving.

## Choosing metrics

1. Start with **one deterministic metric** that captures correctness for your
   data (required facts, keys, format). It is cheap, reproducible and cannot
   be sweet-talked.
2. Add **one judge** with a short, specific rubric and 2–4 dimensions for the
   open-ended part of quality.
3. Blend them (judge 0.5–0.7) into the **objective**, and use the same blend
   as the **loss**.
4. Report the components separately (`compile(metrics=[judge, facts, quality])`)
   so a gain in the blend that comes from only one part is visible.
5. Always evaluate the untouched **test** split before and after.
