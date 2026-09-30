# Reports

Every optimization run can produce an interactive HTML report (HolySheet, with
a self-contained fallback page when HolySheet is not installed) and a JSON
file with the same data. The report answers three questions, in this order:
*did it help*, *what changed*, and *why*.

## Generating one

```python
from agentomatic.optimize import generate_fit_report

before = agent.evaluate(data.test)
history = agent.fit(data, epochs=2)
after = agent.evaluate(data.test)

path = generate_fit_report(
    history,                                   # History, PromptFitResult, a list of them, or the agent
    output_path="reports/support.html",        # + reports/support.json
    baseline_eval=before,                      # EvaluationReports on the SAME examples
    final_eval=after,
    eval_dataset=data.test,                    # questions + expected answers for the table
    dataset_stats=data.metadata.get("augment_stats"),
    model_name="Qwen3.5-9B-MLX-4bit",
    optimizer_name="gepa_like",
)
```

| Argument | Meaning |
| --- | --- |
| `result` | The `History` returned by `fit()` (recommended — all epochs plus loss curves), a `PromptFitResult`, a list of per-epoch results, or the fitted agent. Multiple epochs are merged so the report spans the **original** prompt to the **final** one. |
| `output_path` | HTML path (default `.optimize/<agent>/fit_report_<id>.html`); the JSON sidecar is written next to it. |
| `baseline_eval`, `final_eval` | `EvaluationReport`s before / after, for the per-example table. |
| `baseline_eval_scores`, `eval_scores` | Scores-only alternative to the reports. |
| `eval_dataset` | The evaluated examples. |
| `keras_history` | `History.history`; taken from `result` when it is a `History`. |
| `dataset_stats`, `dataset_sizes` | Augmentation stats and split sizes. |
| `run_config`, `stack_name`, `model_name`, `optimizer_name` | Header and settings details. |

`train_and_report` / scaffolded `train.py` scripts produce this report
automatically (`reports/train_<agent>.html`). `PromptFitter(auto_report=True)`
additionally writes one per fit under `experiment_dir`.

## What is in it

| Section | Shows |
| --- | --- |
| **Verdict** | Validation initial → final, holdout gate, test metrics before → after, a one-line conclusion, and the stop reason. |
| **Test scoreboard** | Every compiled metric on the untouched test split, before / after / Δ. |
| **What changed in the prompt** | The initial → final diff (long lines wrapped so changed sentences stand out), both full prompts side by side, and every accepted change in order with why it was accepted. |
| **Epochs** | Start / best / held-out / candidates / stop reason per epoch. |
| **All candidates** | Every proposed candidate: epoch, round, minibatch / validation / held-out scores, confidence, **decision** and **reason**; the diff of each distinct candidate prompt. |
| **Examples — before vs after** | Per example: question, expected answer, answer before and after, scores before and after, per-metric Δ, and the judge's rationale. Uses the test split when `baseline_eval`/`final_eval` are given, otherwise the validation examples the fitter scored. |
| **Data & settings** | Split sizes, augmentation stats, and every fitter knob used. |
| **Run configuration**, **Recommendations** | Model, optimizer, suggestions, parameter changes, metric deltas, a rollout recommendation. |
| **Curves** | Best score / loss per round, Keras `loss` / `val_loss` and metrics per epoch, the held-out evaluation. |
| **Prompt evolution** | Per round: score, whether accepted, the diff vs the previous version, and the epoch learnings (what worked, what failed, next focus, judge insights). |
| **Failure analysis** | Failure clusters from the train split and APO critiques, when present. |

## Reading it

* **Start at the verdict.** A validation gain with no test gain is the
  signature of overfitting; a test gain that matches the validation gain is
  the result you want.
* **Check the reasons.** Many `not significant` rejections → validation is
  too small or the judge too noisy. Many `does not transfer` → the optimizer
  is fitting the validation examples. Many `duplicate` → the rewrite model
  keeps proposing the current prompt (try a stronger `rewrite_model` or a
  different `optimizer`). See [Generalization](generalization.md#practical-guidance).
* **Read the examples table.** It shows *how* answers changed, which a mean
  score hides.

## The JSON sidecar

`<report>.json` contains `initial_prompt`, `final_prompt`, `prompt_changed`,
`validation` and `holdout_gate` (initial / final), `test` (before / after
scores), `test_examples` (the per-example rows), `epochs` (every
`PromptFitResult.to_dict()`), `keras_history` and `dataset_stats` — for CI
checks, dashboards or diffing runs.

## Evaluation-only reports

```python
from agentomatic.optimize import generate_eval_report

report = agent.evaluate(data.test)
generate_eval_report(report, output_path="reports/eval.html")
```

Scaffolded projects ship `agents/NAME/eval.py`, which does this from the
command line.
