# Low level: `PromptFitter`

`PromptFitter` is the engine behind every entry point. Use it directly when
you want to build the datasets, the metric and the search space yourself,
attach inner-loop callbacks, or optimize an agent running behind the API.

Runnable: [`low_level_prompt_fitter.py`](examples.md) — every option below,
commented, plus a before/after test evaluation and a report.

```python
from agentomatic.optimize import (
    CompositeMetric, Dataset, LocalJudgeMetric, PromptFitter, PromptSearchSpace,
    WeightedMetric, generate_fit_report, load_data,
)

data = load_data("datasets/support.jsonl")
points = lambda examples: Dataset(points=[e.to_datapoint() for e in examples])

metric = CompositeMetric([
    WeightedMetric(name="judge", metric=LocalJudgeMetric(model="omlx/Qwen3.5-9B-MLX-4bit",
                                                          criteria="Correct and grounded."), weight=0.6),
    WeightedMetric(name="facts", metric=facts, weight=0.4),
])

fitter = PromptFitter(
    agent="support_agent",
    local_agent=agent,                                  # in-process, no server
    task_model="omlx/Qwen3.5-9B-MLX-4bit",
    rewrite_model="omlx/Qwen3.5-9B-MLX-4bit",
    llm_base_url="http://127.0.0.1:8000/v1",
    optimizer="rewrite",
    search_space=PromptSearchSpace(optimize_system_prompt=True, optimize_few_shot=False,
                                   optimize_model_params=False),
    max_trials=8,
    baseline_system_prompt=agent.system_prompt,
)

before = agent.evaluate(data.test, [facts])
result = await fitter.fit(points(data.train), points(data.validation), metric,
                          testset=points(data.holdout))       # the holdout GATE
agent.compiled_config["system_prompt"] = result.best_prompt
after = agent.evaluate(data.test, [facts])

generate_fit_report(result, output_path="reports/fit.html",
                    baseline_eval=before, final_eval=after, eval_dataset=data.test)
```

## `fit(trainset, valset, metric, testset=None)`

| Argument | Role |
| --- | --- |
| `trainset` | What the optimizer learns from (reflection, few-shot pool). |
| `valset` | What candidates are selected on. |
| `metric` | The objective — any metric of [either family](metrics.md#two-metric-families). |
| `testset` | The **holdout gate** (veto only). Omit to carve `holdout_fraction` of `valset`. Do not pass your real test split. |

## Constructor options

**Target and models**

| Option | Default | Meaning |
| --- | --- | --- |
| `agent` | — | Agent name (reports, `prompts.json`, API routes). |
| `local_agent` | `None` | A live agent instance, called in-process. |
| `api_base`, `api_prefix` | `http://localhost:8000`, `/api/v1` | The running platform to call when `local_agent` is not set. |
| `task_model` | `"ollama/qwen2.5:7b"` | Model for failure clustering and optimizer analysis calls. |
| `rewrite_model` | `task_model` | Model that writes candidate prompts. |
| `llm_base_url`, `llm_api_key` | — | OpenAI-compatible endpoint for those calls (also the process default). |
| `local_judges` | `None` | Extra judge model specs appended to the objective as a panel. |

**Baseline**

| Option | Default | Meaning |
| --- | --- | --- |
| `baseline_system_prompt` | `None` | Start from this prompt instead of `prompts.json`. |
| `base_prompt_version` | `"v1"` | `prompts.json` version used when no baseline prompt is given. |
| `baseline_model_params`, `baseline_few_shot_examples` | `None` | Start from these params / demos (the class-agent bridge uses them to compound across epochs). |

**Search**

| Option | Default | Meaning |
| --- | --- | --- |
| `optimizer` | `"gepa_like"` | `rewrite`, `gepa_like`, `mipro_like`, `few_shot`, `param_search`, `apo`, or an object with `propose(...)` ([table](how-it-works.md#optimizers-how-candidates-are-proposed)). |
| `search_space` | prompt + few-shot + params | What may change ([below](#search-space)). |
| `max_trials` | `30` | Distinct candidates evaluated; 4 per round. |
| `patience` | `3` | Rounds without an accepted candidate before stopping. |
| `rewrite_passes`, `multipass`, `slm_multipass`, `llm_multipass`, `slm_default_passes`, `llm_default_passes` | auto: 3 passes for local models, 2 for frontier | Draft → critique → refine passes for rewrites. |

**Acceptance (anti-overfitting)** — see [Generalization](generalization.md#the-acceptance-rules)

| Option | Default |
| --- | --- |
| `min_absolute_improvement` | `0.001` |
| `min_confidence` | `0.8` |
| `max_generalization_gap` | `0.15` |
| `holdout_tolerance` | `0.02` |
| `min_transfer_ratio` | `0.25` |
| `holdout_fraction` | `0.2` |
| `reflect_on` | `"train"` |
| `reflection_size` | `8` |

**Execution and output**

| Option | Default | Meaning |
| --- | --- | --- |
| `concurrency` (alias `n_runners`) | `1` | Parallel agent / judge calls. `>1` disables `sequential` unless forced. |
| `sequential` | auto | Force one call at a time (kind to small local servers). |
| `callbacks` | `None` | Inner-loop callbacks ([below](#callbacks)). |
| `auto_report`, `experiment_dir` | `True`, `.optimize` | Write an HTML trial report per fit. |
| `dashboard` | `False` | Live terminal dashboard. |
| `drain_seconds` | `1.5` | Pause after the loop so in-flight calls finish. |
| `trace_store_path` | `None` | Persist execution traces (used by `apo`). |

## Search space

`PromptSearchSpace` declares what an optimizer may change; anything not
enabled stays as in the baseline.

| Field | Default | Meaning |
| --- | --- | --- |
| `optimize_system_prompt` | `True` | Rewrite the system prompt |
| `optimize_user_template` | `False` | Rewrite the user-message template |
| `optimize_few_shot` | `True` | Choose demonstrations from train |
| `max_few_shot_examples`, `few_shot_selection_strategy` | `5`, `"diversity_weighted"` | Demo count and selection |
| `optimize_model_params` + `model_param_space` | `True`, a temperature / top_p / max_tokens grid | e.g. `{"temperature": [0.0, 0.2, 0.5]}` |
| `optimize_rag_params` + `rag_param_space` | `False`, `{}` | e.g. `{"top_k": [3, 5, 8]}` |
| `optimize_tool_params` + `tool_param_space` | `False`, `{}` | Tool policy knobs |
| `optimize_model_choice` + `model_choices`, `fallback_models` | `False` | Choose among models |
| `search_method` | `"random"` | `grid`, `random` or `tpe` for parameter search |
| `optimize_nodes`, `node_match` | all | Scope trace-aware optimization to graph nodes |

!!! tip
    Turn off what you do not intend to tune. With the defaults, a run may
    also change few-shot demos and sampling parameters.

## Callbacks

`agentomatic.optimize` callbacks observe and steer the inner trial loop:

| Callback | Options (defaults) | Does |
| --- | --- | --- |
| `EarlyStopping` (also exported as `OptimizeEarlyStopping`) | `monitor="score"`, `patience=3`, `min_delta=0.005`, `mode="max"`, `restore_best_weights=True` | Stops when the best score stalls |
| `ScoreThreshold` | `threshold=0.85`, `mode="max"` | Stops once good enough |
| `ModelCheckpoint` | `save_dir="optimization_results/checkpoints"`, `save_best_only=True`, `save_freq=1`, `max_checkpoints=5` | Saves accepted configs |
| `PlateauStopping` | `patience=2`, `factor=0.5`, `min_temperature=0.1` | Cools the rewrite temperature on plateaus |
| `TemperatureScheduler` | `initial_temperature=0.7`, `min_temperature=0.1`, `decay_rate=0.9`, `decay_type="exponential"`, `step_size=3` | Schedules the rewrite temperature |
| `NaNStopping` | `max_consecutive_nan=2`, `validate_output=True`, `nan_rollback=True` | Aborts on broken outputs |
| `ProgressLogger` | `show_prompt_diff=False`, `show_delta_chars=0` | Logs each trial |

## The result: `PromptFitResult`

| Field | Contents |
| --- | --- |
| `baseline_config`, `best_config` | `PromptRuntimeConfig` (system prompt, user template, few-shot examples, model / RAG / tool params) |
| `baseline_score`, `best_score`, `absolute_improvement`, `improved` | Validation scores |
| `baseline_holdout_score`, `holdout_score`, `generalization_gap` | Holdout gate scores |
| `trials` | Every candidate and phase: `name`, `phase` (`skipped`, `minibatch`, `full_val`), `score`, `decision`, `reason`, `confidence`, `holdout_score`, `system_prompt`, … |
| `baseline_examples`, `best_examples` | Per-example validation results (question, expected, answer, score, feedback) |
| `metric_deltas`, `param_suggestions`, `failure_clusters`, `suggestions` | What changed and why |
| `score_history`, `prompt_history` | Per-round curves and epoch learnings |
| `settings`, `dataset_sizes`, `early_stop_reason`, `duration_seconds` | How the run was configured and ended |
| `best_prompt`, `best_params`, `best_few_shot_examples` | Shortcuts |

Methods: `summary()` (text), `to_dict()` (JSON), and
`apply(version="v2_fit", agent_dir=None, *, force=False, min_improvement=0.0,
require_generalization=True, max_generalization_gap=0.15)`, which writes
`prompts.json`, `runtime_config.json` and an audit line to
`fit_history.jsonl` — or refuses a result that did not improve or does not
generalise.
