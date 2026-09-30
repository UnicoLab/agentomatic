# Class agents: compile, fit, evaluate

`BaseGraphAgent` has a Keras-style optimization API. You describe the data,
what "good" means and how to search; `fit()` runs the
[optimization loop](how-it-works.md) once per epoch and keeps the best prompt.

Runnable: [`examples/prompt_optimization/train.py`](examples.md) — every step
below, with every option listed in comments.

## Requirements on the agent

* Nodes read the prompt with `self.resolve_system_prompt(default=...)` (see
  [How it works](how-it-works.md#the-agent-must-read-its-prompt-through-the-framework)).
* `input_to_state(data)` builds the state from the example's `input`
  (the question arrives as `data["current_query"]`).
* `state_to_output(state)` returns a dict with the answer under `response`
  (or your structured keys — metrics read them).

## Step by step

```python
from agentomatic.agents import (
    CallableMetric, EarlyStopping, EpochDiffCallback, MetricLoss,
    PromptFitterBridge, WeightedMetric,
)
from agentomatic.optimize import (
    LocalJudgeMetric, PromptSearchSpace, generate_fit_report, load_data,
    merge_fit_results, prepare_dataset,
)

# 1. data — train / validation / holdout / test (see Datasets)
data = load_data("agents/support/datasets/all.jsonl")
data, _ = prepare_dataset(data, augment=True, n_examples=60,
                          model="omlx/Qwen3.5-9B-MLX-4bit",
                          llm_base_url="http://127.0.0.1:8000/v1")

# 2. metrics — reported every epoch (see Metrics, judges & loss)
judge = LocalJudgeMetric(name="judge", model="omlx/Qwen3.5-9B-MLX-4bit",
                         criteria="Correct, grounded, complete and concise.",
                         dimensions=["correctness", "completeness"])
facts = CallableMetric("facts", must_include_score)
quality = WeightedMetric([("judge", judge, 0.6), ("facts", facts, 0.4)], name="quality")

# 3. loss — what fit() records as loss / val_loss
loss = MetricLoss(quality)

# 4. optimizer — how candidates are proposed and selected
optimizer = PromptFitterBridge(
    agent_name=agent.agent_name,
    task_model="omlx/Qwen3.5-9B-MLX-4bit",
    rewrite_model="omlx/Qwen3.5-9B-MLX-4bit",
    llm_base_url="http://127.0.0.1:8000/v1",
    llm_api_key="local",
    optimizer="gepa_like",
    metric=quality,
    max_trials=8,
    search_space=PromptSearchSpace(optimize_system_prompt=True, optimize_few_shot=False,
                                   optimize_model_params=False),
    min_confidence=0.8,           # any PromptFitter keyword is passed through
)

# 5. compile
agent.compile(data, metrics=[judge, facts, quality], optimizer=optimizer, loss=loss)

# 6. fit — the test split is never used; measure it before and after
before = agent.evaluate(data.test)
history = agent.fit(
    data,
    epochs=2,
    callbacks=[EpochDiffCallback(epochs=2),
               EarlyStopping(monitor="val_loss", mode="min", patience=1)],
)
after = agent.evaluate(data.test)

# 7. report, and keep the prompt if it earned it
generate_fit_report(history, output_path="reports/support.html",
                    baseline_eval=before, final_eval=after, eval_dataset=data.test)
merge_fit_results(history.fit_results).apply(version="v2_fit", agent_dir="agents/support")
```

## `compile(dataset=None, metrics=None, optimizer=None, loss=None)`

| Argument | Meaning |
| --- | --- |
| `dataset` | Default data for `fit()`/`evaluate()`. |
| `metrics` | Reported metrics — either [family](metrics.md#two-metric-families). Names must be unique and not `loss`. |
| `optimizer` | A `PromptFitterBridge` (or any object with `optimize(agent, dataset, metrics) -> dict`). |
| `loss` | A loss, or a metric (wrapped as `MetricLoss`). |

## `PromptFitterBridge(...)`

The bridge runs one `PromptFitter.fit` per epoch on the agent instance, starting
from the agent's current prompt.

| Option | Default | Meaning |
| --- | --- | --- |
| `agent_name` | `""` | Agent name (reports, `prompts.json`). |
| `task_model` | `"ollama/qwen2.5:7b"` | Model for failure clustering and the optimizer's analysis calls. (The agent itself runs on its own LLM.) |
| `rewrite_model` | `"openai/gpt-4.1"` | Model that writes the candidate prompts. |
| `llm_base_url`, `llm_api_key` | — | OpenAI-compatible endpoint for optimizer and judge calls (also set as the process default at construction, so epoch-0 judge scoring uses it). |
| `optimizer` | `"rewrite"` | See [optimizers](how-it-works.md#optimizers-how-candidates-are-proposed). |
| `metric` | compiled loss's metric, else first compiled metric | The objective candidates are selected on. |
| `max_trials` | `8` | Distinct candidates per epoch. |
| `**kwargs` | | Passed to `PromptFitter`: `search_space`, `patience`, `min_absolute_improvement`, `min_confidence`, `max_generalization_gap`, `holdout_tolerance`, `min_transfer_ratio`, `holdout_fraction`, `reflect_on`, `reflection_size`, `concurrency`, `callbacks`, `auto_report`, `experiment_dir`, … (see [PromptFitter](prompt-fitter.md#constructor-options)). |

How the bridge splits your `AgentDataset`: `train` → what the optimizer
learns from; `validation` → selection; `holdout` → the veto gate (or a slice of
validation); `test` → dropped. Without a `validation` split, 20 % of train is
carved off with a seeded shuffle.

After each epoch: `agent._last_fit_result` is that epoch's `PromptFitResult`,
`agent._last_optimize_status` is `"ok"` or `"skipped: <reason>"` (a failed
optimizer never crashes `fit()`; check this when nothing changed), and the
prompt is kept only if the epoch improved it.

## `fit(dataset=None, *, epochs=1, verbose=1, callbacks=None, validation_data=None, ...)`

| Argument | Meaning |
| --- | --- |
| `epochs` | Optimization passes; each starts from the previous best. |
| `verbose` | `0` silent, `1` one line per epoch. |
| `callbacks` | `agentomatic.agents` callbacks (below). |
| `validation_data` | Examples for `val_*` logs (default: the validation split). |
| `search_space`, `optimize_mode`, `optimize_prompt`, `optimize_params`, `optimize_few_shot`, `model_param_space`, `max_trials`, `**optimize_kwargs` | Per-call overrides for the bridge. Given without a compiled optimizer, `fit()` builds a `PromptFitterBridge` from them. |

Before the first epoch, `fit()` records a baseline row (epoch −1), so the
curve starts at the true starting loss.

Returns a `History`:

| Attribute | Contents |
| --- | --- |
| `history.history` | `{"loss": [...], "val_loss": [...], "judge": [...], "val_judge": [...], ...}` — baseline first |
| `history.epoch` | Epoch indices (`-1` = baseline) |
| `history.fit_results` | One `PromptFitResult` per optimized epoch |
| `history.params` | Epochs, optimizer, metric names, split sizes |
| `history.best("val_loss", mode="min")`, `history.final("loss")`, `history.summary()` | Helpers |

### Callbacks for `fit()`

| Callback | Options | Does |
| --- | --- | --- |
| `EarlyStopping` | `monitor="loss"`, `mode="auto"` (min for `*loss`), `patience=0`, `min_delta=0.0`, `restore_best=False` | Stops when `monitor` stops improving; `restore_best` restores the best epoch's prompt/config |
| `EpochDiffCallback` | `epochs=1`, `prompt_key="system_prompt"` | Prints loss and a unified prompt diff after each epoch; `per_epoch` keeps the records |
| your own `Callback` subclass | `on_train_begin/end`, `on_epoch_begin/end(epoch, logs)` | Anything; set `self.agent.stop_training = True` to stop |

These are different from the `agentomatic.optimize` callbacks
(`EarlyStopping`, `ModelCheckpoint`, …), which control the fitter's inner
trial loop — pass those to the bridge as `callbacks=[...]`. Passing one to
`fit()` raises a `TypeError` that says so.

## `evaluate(dataset, metrics=None)`

Runs the agent on each example and scores it with `metrics` (default: the
compiled ones). Returns an `EvaluationReport`: `scores` (mean per metric),
`example_results` (per example: `prediction`, `scores`, `error`,
`duration_ms`, judge rationale in `metadata`) and `metadata`
(`metric_failures`, `transform_errors`). Pass two reports (before and after
`fit()`) to [`generate_fit_report`](reports.md) for a per-example comparison.

## Saving

`agent.save(path)` writes the compiled config (best prompt, params, few-shot
examples) and `fit_history.json`; `Agent.load(path)` / `agent.load_compiled(path)`
restore them. `PromptFitResult.apply(version=..., agent_dir=...)` instead writes a
new version to the agent's `prompts.json`.

## One call: `train_and_report`

Scaffolded projects (`agentomatic init NAME --template basic|full|…`) ship
`agents/NAME/train.py`, which wraps the steps above in `train_and_report` and
reads the model endpoint from the project's stack:

```bash
python agents/NAME/train.py --epochs 2 --trials 12 --augment --n-examples 40
AGENTOMATIC_STACK=gemini python agents/NAME/train.py --optimizer gepa_like --apply
```

| Flag / env (`AGENTOMATIC_…`) | Default | Meaning |
| --- | --- | --- |
| `--stack` | `local` | Stack whose `default` / `rewrite` LLM profiles are used |
| `--epochs` | `2` | Epochs |
| `--trials` | `12` | Candidates per epoch |
| `--patience` | `2` | Fitter patience |
| `--optimizer` | `rewrite` | `rewrite`, `gepa_like`, `mipro_like`, `few_shot_bootstrap`, `param_search` |
| `--dataset` | `datasets/all.jsonl` | Dataset path |
| `--augment`, `--n-examples`, `--persist` | off | Augmentation (see [Datasets](data.md#augmentation)) |
| `--apply`, `--apply-as` | off, `v2_fit` | Write the prompt to `prompts.json` if it improved |
| `--min-improvement` | `0.001` | Minimum validation lift |
| `--persist-fit-store` | off | Also store retrain artefacts in `AGENTOMATIC_FIT_STORE_URL` / `DATABASE_URL` |

In Python, `train_and_report(agent, config=TrainConfig(...))` accepts the same
settings plus `required_keys`, `judge_criteria`, `judge_dimensions`,
`judge_weight`, `sequential`, `concurrency`, `optimize_model_params`,
`augment_strategies`, `augment_model` and `evaluate_baseline` (default on:
score the test split before *and* after, for the report). It returns a
`TrainResult` with `history`, `fit_result` (merged across epochs),
`fit_results`, `baseline_eval_scores`, `eval_scores`, `report_path`,
`optimize_status`, `applied_version` and `dataset_sizes`.

The staged primitives it uses are public too: `build_default_metrics`,
`compile_agent`, `fit_agent`, `evaluate_agent`.
