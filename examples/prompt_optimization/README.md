# Prompt optimization examples

Runnable, commented examples of Agentomatic's prompt optimization, from "score
some answers" to a full Keras-style training run with an HTML report. They all
optimize the same small RAG `SupportAgent` (`agent.py`), which answers Acme
Cloud customer questions from policy documents. These are the documents sent
with the request (`context.documents`) when there are any, otherwise the ones
it retrieves from its own knowledge base. It also takes the customer's plan
into account.

| Script | Level | What you learn |
| --- | --- | --- |
| [`metrics_and_judges.py`](metrics_and_judges.py) | building blocks | Every metric family, LLM judges (single, per-dimension, panel), composite objectives, adapters between the two metric protocols, losses. Scores three fixed answers; no optimization. |
| [`data_augmentation.py`](data_augmentation.py) | building blocks | `prepare_dataset(augment=True)`: every strategy and option, leak protection, `augment_stats`. |
| [`anti_overfitting.py`](anti_overfitting.py) | mechanics | The acceptance rules on hand-made numbers, then a fit where a prompt that memorizes the validation answers is vetoed and a general one is kept. |
| [`low_level_prompt_fitter.py`](low_level_prompt_fitter.py) | low level | `PromptFitter` with every knob explicit: datasets, composite metric, search space, inner-loop callbacks, acceptance rules, before/after test evaluation, report. |
| [`train.py`](train.py) | high level | The class-agent workflow: `load_data` → `prepare_dataset` → metrics → loss → `PromptFitterBridge` → `compile` → `fit` → test evaluation (overall and per tag) → `generate_fit_report`. |
| [`rag_context.py`](rag_context.py) | RAG | Rows that bring their own documents, plan, metadata, tags and rubric (`datasets/policies.jsonl`). Prints what the agent, the judge and the rewrite model each see, fits with context-aware metrics, and reports test scores per tag. |

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
python examples/prompt_optimization/rag_context.py --epochs 1 --trials 4 [--augment]
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
`low_level_report.html` and `low_level_result.json`; `rag_context.py` writes
`rag_report.html` (+ `rag_report.json`) and `rag_summary.json`.

## The data

Both datasets have one JSON object per line. A row carries the whole case,
not just a question and an answer. This is `datasets/policies.jsonl`:

```json
{"id": "policy_05", "split": "train",
 "input": {"current_query": "How long are audit logs kept?",
           "context": {"documents": [{"source": "audit.md",
               "content": "Audit logs are kept for 400 days on Enterprise and 90 days on Team."}]},
           "customer_plan": "team"},
 "expected_output": {"response": "On the Team plan, audit logs are kept for 90 days (400 days on Enterprise)."},
 "metadata": {"topic": "security", "difficulty": "medium", "must_include": ["90 days"]},
 "tags": ["security", "compliance", "plan-specific"],
 "rubric": {"groundedness": "Every claim comes from the row's context documents.",
            "plan": "Applies the policy to the customer's plan when one is given.",
            "refusal": "Says so plainly when the documents do not answer the question."}}
```

`datasets/support.jsonl` uses the same shape. Its agent retrieves documents
itself, so rows carry no documents, and `customer_plan` appears only where the
plan changes the answer.

Every stage of the optimization gets the whole row:

| Stage | Sees |
| --- | --- |
| Agent | `input` exactly as written: the question, `context.documents`, `customer_plan` |
| Metrics | The whole row: `example.input`, `.metadata`, `.tags` and `.rubric`, in `fit()` and in `evaluate()` |
| LLM judge | Question, answer, expected answer plus rubric, the documents (or what the agent retrieved), and the other inputs, metadata and tags |
| Rewrite model | Each failure, success and sample with its documents, inputs, metadata and tags, plus what the agent retrieved |
| Augmenter | The seed's documents and inputs; variations inherit them |
| Report | A `context` column in the per-example table |

The four splits have four jobs, and keeping them apart is what stops the
optimizer from overfitting:

| Split | Rows (support / policies) | Role |
| --- | --- | --- |
| `train` | 10 / 8 | What the optimizer **learns from**: failure analysis, judge feedback, few-shot demos. Augmentation only grows this split. |
| `validation` | 6 / 5 | What candidate prompts are **scored and selected on**. |
| `holdout` | 4 / 3 | A **veto-only gate**: a candidate that wins on validation but does not carry over here is rejected. Optional; without it a slice of validation is reserved. |
| `test` | 6 / 4 | **Never used by optimization.** Evaluate it before and after `fit()` for an unbiased number. |

`metadata.must_include` feeds the deterministic `facts` metric
(`common.must_include_score`). `input.context.documents` feeds the `grounded`
metric (`common.grounded_score`). `tags` break test scores down by slice
(`common.scores_by_tag`), so you can see whether the new prompt helped billing
questions and hurt security ones.

## Testing without a model server

`tests/test_prompt_optimization_examples.py` runs every script end to end
against `tests/fake_openai_server.py`, a deterministic stand-in that plays the
agent, the judge, the prompt rewriter and the augmenter.
