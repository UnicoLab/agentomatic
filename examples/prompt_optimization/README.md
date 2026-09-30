# Prompt optimization examples

Runnable, commented examples of Agentomatic's prompt optimization, from "score
some answers" to a full Keras-style training run with an HTML report. They all
optimize the same small `SupportAgent` (`agent.py`), which answers Acme Cloud
customer questions from a policy knowledge base, on `datasets/support.jsonl`.

| Script | Level | What you learn |
| --- | --- | --- |
| [`metrics_and_judges.py`](metrics_and_judges.py) | building blocks | Every metric family, LLM judges (single, per-dimension, panel), composite objectives, adapters between the two metric protocols, losses. Scores three fixed answers; no optimization. |
| [`data_augmentation.py`](data_augmentation.py) | building blocks | `prepare_dataset(augment=True)`: every strategy and option, leak protection, `augment_stats`. |
| [`anti_overfitting.py`](anti_overfitting.py) | mechanics | The acceptance rules on hand-made numbers, then a fit where a prompt that memorizes the validation answers is vetoed and a general one is kept. |
| [`low_level_prompt_fitter.py`](low_level_prompt_fitter.py) | low level | `PromptFitter` with every knob explicit: datasets, composite metric, search space, inner-loop callbacks, acceptance rules, before/after test evaluation, report. |
| [`train.py`](train.py) | high level | The class-agent workflow: `load_data` → `prepare_dataset` → metrics → loss → `PromptFitterBridge` → `compile` → `fit` → test evaluation → `generate_fit_report`. |

Start with `metrics_and_judges.py` if you are choosing metrics, `train.py` if
you want the whole loop, and `low_level_prompt_fitter.py` when you need control
over every step. In a scaffolded project (`agentomatic init NAME --template
basic`), `agents/NAME/train.py` calls `train_and_report(...)`, which runs this
same loop from the project's stack configuration.

## Running them

Each example talks to an OpenAI-compatible model server. oMLX is the default;
llama.cpp, vLLM, LM Studio or Ollama's `/v1` work the same way.

```bash
omlx serve --model Qwen3.5-9B-MLX-4bit        # http://127.0.0.1:8000/v1

python examples/prompt_optimization/metrics_and_judges.py
python examples/prompt_optimization/data_augmentation.py --n-examples 40
python examples/prompt_optimization/anti_overfitting.py
python examples/prompt_optimization/low_level_prompt_fitter.py --trials 8
python examples/prompt_optimization/train.py --epochs 2 --trials 6 [--augment]
```

Shared flags (or environment variables), defined in `common.py`:

| Flag | Environment variable | Default |
| --- | --- | --- |
| `--base-url` | `OMLX_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `--api-key` | `OMLX_API_KEY` | `local` |
| `--model` | `AGENTOMATIC_LOCAL_MODEL` | `Qwen3.5-9B-MLX-4bit` |
| `--judge-model` | `EXAMPLE_JUDGE_MODEL` | same as `--model` |
| `--rewrite-model` | `EXAMPLE_REWRITE_MODEL` | same as `--model` |
| `--out-dir` | `EXAMPLE_OUT_DIR` | `examples/prompt_optimization/out` |
| `--log-level` | `AGENTOMATIC_LOG_LEVEL` | `INFO` |

`train.py` writes `train_report.html` (+ `train_report.json`) and
`train_summary.json` to `--out-dir`; `low_level_prompt_fitter.py` writes
`low_level_report.html` and `low_level_result.json`.

## The data

`datasets/support.jsonl` has one JSON object per line:

```json
{"id": "refund_2", "split": "validation",
 "input": {"current_query": "I bought an annual plan last week. Can I still get my money back?"},
 "expected_output": {"response": "Yes. Annual plans can be refunded within 30 days of purchase."},
 "metadata": {"must_include": ["30 days"], "topic": "refund"}}
```

The four splits have four jobs, and keeping them apart is what stops the
optimizer from overfitting:

| Split | Rows | Role |
| --- | --- | --- |
| `train` | 10 | What the optimizer **learns from**: failure analysis, judge feedback, few-shot demos. Augmentation only grows this split. |
| `validation` | 6 | What candidate prompts are **scored and selected on**. |
| `holdout` | 4 | A **veto-only gate**: a candidate that wins on validation but does not carry over here is rejected. Optional; without it a slice of validation is reserved. |
| `test` | 6 | **Never used by optimization.** Evaluate it before and after `fit()` for an unbiased number. |

`metadata.must_include` feeds the deterministic `facts` metric
(`common.must_include_score`); any metadata you put there reaches class-agent
metrics during `fit()`, `evaluate()` and inside composite metrics.

## Testing without a model server

`tests/test_prompt_optimization_examples.py` runs every script end to end
against `tests/fake_openai_server.py`, a deterministic stand-in that plays the
agent, the judge, the prompt rewriter and the augmenter.
