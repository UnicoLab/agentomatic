# Generalization & overfitting

An optimizer rewarded only for its validation score will find prompts that
are good *at the validation set*: prompts that paste validation answers in,
that exploit one judge's quirks, or that win by luck on six examples. None of
that survives contact with real users. Agentomatic prevents it with two
things: **separate data roles** and **acceptance rules** every candidate must
pass.

Runnable demo: [`anti_overfitting.py`](examples.md) applies the rules to
hand-made numbers, then runs a fit in which a prompt that memorises the
validation answers is vetoed and a general prompt is kept.

## Data roles

| Split | Role | Can it change which prompt wins? |
| --- | --- | --- |
| `train` | What the optimizer **learns from**: failure clusters, judge feedback, few-shot demos. Augmentation grows only this split. | Indirectly (it shapes proposals) |
| `validation` | What candidates are **scored and selected on**. | Yes |
| `holdout` | A **veto-only gate**: a candidate that wins on validation but does not carry over here is rejected. It never picks a winner. | Can only say no |
| `test` | **Never passed to the fitter.** Evaluate it before and after `fit()`. | No |

* Label rows with `"split": "train" | "validation" | "holdout" | "test"`
  (see [Datasets](data.md)). Unknown split names trigger a warning.
* No `holdout` rows? The fitter reserves `holdout_fraction` (default 20 %, at
  least one example) of validation as the gate. If validation is too small to
  split, the slice comes from train instead and is removed from train, so the
  gate stays unseen by the optimizer.
* No `validation` rows? The class-agent bridge carves 20 % of train (seeded
  shuffle); the optimizer then still never selects on what it learned from.
* With `PromptFitter` directly, `fit(trainset, valset, metric, testset=gate)`:
  the `testset` argument **is the holdout gate**. Keep your real test split
  out of the fitter and evaluate it yourself.

Why keep the test split out? Once a split has been used to accept or reject
candidates, the winner was chosen partly *because* it did well there; its
score on that split is optimistic. Only a split that played no part in the
choice gives an unbiased estimate.

## The acceptance rules

A candidate replaces the incumbent (the best config so far) only if **all**
of these hold. Every rejected candidate records which rule failed in its
`reason`.

| Rule | Knob (default) | Rejects |
| --- | --- | --- |
| **Minimum lift**: validation score − incumbent ≥ threshold | `min_absolute_improvement` (`0.001`; the examples use `0.01`) | Changes too small to matter |
| **Significance**: paired bootstrap over the validation examples, P(candidate > incumbent) ≥ threshold | `min_confidence` (`0.8`) | Wins that are likely judge noise or luck on a few examples |
| **No holdout regression**: holdout − incumbent's holdout ≥ −tolerance | `holdout_tolerance` (`0.02`) | Prompts that get better on validation and worse elsewhere |
| **Bounded gap growth**: (validation − holdout) − (incumbent's validation − holdout) ≤ max(gap, ½ × validation lift) | `max_generalization_gap` (`0.15`) | Prompts whose gain is concentrated on the validation examples |
| **Transfer**: holdout lift ≥ ratio × validation lift | `min_transfer_ratio` (`0.25`) | Memorisation: large validation gain, (almost) none held out. Skipped when the holdout is already ≥ 0.99. |

Before a candidate reaches these rules it must also beat the incumbent on a
validation **minibatch of the same examples** (promotion), and be a config
that was not already scored (duplicates are skipped).

The rules are relative to the **incumbent**, not to absolute values: a
dataset whose baseline already scores differently on validation and holdout is
not penalised for that difference, only for widening it.

### Worked examples

Incumbent: validation 0.50, holdout 0.50. Defaults as above.

| Candidate | Validation | Holdout | Decision | Why |
| --- | --- | --- | --- | --- |
| General improvement | 0.80 | 0.75 | accept | lift 0.30, gap grows 0.05 |
| Memorised validation answers | 1.00 | 0.50 | reject | gap grows 0.50 > 0.25; holdout lift 0 < 25 % of 0.50 |
| Gain does not transfer | 0.62 | 0.52 | reject | holdout +0.02 < 25 % of +0.12 |
| Holdout regression | 0.52 | 0.45 | reject | holdout −0.05 < −0.02 |
| Gap blows up | 0.95 | 0.60 | reject | gap grows 0.35 > 0.225 |

Significance, six validation examples, incumbent per-example scores
`[0, 0, 1, 1, 0, 1]`:

| Candidate | Per-example | P(better) | Decision |
| --- | --- | --- | --- |
| One example flipped | `[1, 0, 1, 1, 0, 1]` | 0.67 | reject (not significant) |
| Every failure fixed | `[1, 1, 1, 1, 1, 1]` | 0.98 | accept |

Both checks are public functions if you want to apply them yourself:

```python
from agentomatic.optimize import check_generalization, paired_improvement_confidence

check = check_generalization(
    fit_score=0.62, holdout_score=0.52,          # candidate
    baseline_fit=0.50, baseline_holdout=0.50,    # incumbent
    max_gap=0.15, holdout_tolerance=0.02, min_transfer_ratio=0.25,
)
check.ok, check.reason          # False, "Does not transfer: validation improved +0.1200 but …"

paired_improvement_confidence([1, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1])   # 0.67
```

## Learning only from train

The optimizer's inputs — the examples it analyses, the failure clusters, the
judge feedback, the few-shot demos it may add — come from **train**
(`reflect_on="train"`, `reflection_size=8`). An optimizer that read the
validation failures would write prompts that fix exactly those examples, and
validation would stop measuring anything. `reflect_on="validation"` restores
that behaviour for comparison only.

## After the fit: `apply()` and the test split

```python
before = agent.evaluate(data.test)      # or evaluate before fit()
history = agent.fit(data, epochs=2)
after = agent.evaluate(data.test)
result = merge_fit_results(history.fit_results)

result.apply(version="v2_fit", agent_dir="agents/support")
```

`apply()` writes `prompts.json` only when the result improved
(`min_improvement`) and, when a holdout score exists, its generalization gap
is within `max_generalization_gap`. `force=True` overrides both — do it only
after reading the report.

## Practical guidance

* **Size.** Six validation examples can only distinguish large improvements
  (one flipped example is not significant at 0.8). Aim for 20+ validation and
  10+ holdout examples; augment **train**, never validation or holdout.
* **Diversity.** Validation and holdout should cover the same kinds of
  questions, phrased differently. A holdout made of one topic cannot tell
  memorisation from a topic-specific gain.
* **Deterministic judges.** `temperature=0.0` for every judge; add a
  deterministic metric to the objective.
* **Read the reasons.** A run where every candidate is `not significant`
  means the validation set is too small or the judge too noisy; a run full of
  `does not transfer` means the optimizer is fitting the validation set — add
  data before loosening the rules.
* **Loosening the rules** (`min_confidence=0.6`, `min_transfer_ratio=0`,
  larger `max_generalization_gap`) makes more candidates pass and more of
  them overfit. Measure the test split before and after if you do.
