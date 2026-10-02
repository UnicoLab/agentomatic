# Examples

[`examples/prompt_optimization`](https://github.com/UnicoLab/agentomatic/tree/main/examples/prompt_optimization)
contains runnable scripts, from scoring a few answers to a full training run.
Each one is commented step by step and lists every option of the methods it
uses. All of them optimize the same small RAG `SupportAgent`, which answers
Acme Cloud customer questions from policy documents. Those are the documents a
row sends (`context.documents`), or else the ones it retrieves from its own
knowledge base, and it also takes the customer's plan into account. Two
datasets are used, each split into train / validation / holdout / test:

* `support.jsonl` has 26 rows, with tags and a `customer_plan` input where the
  plan matters.
* `policies.jsonl` has 20 rows. Each brings its own documents, plus a plan,
  metadata, tags and a rubric.

| Script | Level | Covers | Docs |
| --- | --- | --- | --- |
| `metrics_and_judges.py` | building blocks | Every metric family, LLM judges (single, per-dimension, panel), composite objectives, adapters, losses — scored on three fixed answers | [Metrics](metrics.md) |
| `data_augmentation.py` | building blocks | `prepare_dataset` strategies and options, leak protection, `augment_stats` | [Datasets](data.md) |
| `anti_overfitting.py` | mechanics | The acceptance rules on hand-made numbers; a fit in which a prompt memorising the validation answers is vetoed and a general one kept | [Generalization](generalization.md) |
| `low_level_prompt_fitter.py` | low level | `PromptFitter` with every knob, inner-loop callbacks, before/after test evaluation, report | [PromptFitter](prompt-fitter.md) |
| `train.py` | high level | `load_data` → `prepare_dataset` → metrics → loss → `PromptFitterBridge` → `compile` → `fit` → test evaluation (overall and per tag) → `generate_fit_report` | [Class agents](class-agents.md) |
| `rag_context.py` | RAG | Rows with their own documents, plan, metadata, tags and rubric. Prints what the agent, the judge and the rewrite model see; context-aware metrics (`facts`, `grounded`); per-tag test scores; `--augment` keeps each seed's documents | [Datasets](data.md#rag-datasets) |

## Running them

Start an OpenAI-compatible model server (oMLX shown; llama.cpp, vLLM,
LM Studio or Ollama's `/v1` work the same way), then:

```bash
omlx serve --model Qwen3.5-9B-MLX-4bit        # http://127.0.0.1:8000/v1

python examples/prompt_optimization/metrics_and_judges.py
python examples/prompt_optimization/data_augmentation.py --n-examples 40
python examples/prompt_optimization/anti_overfitting.py
python examples/prompt_optimization/low_level_prompt_fitter.py --trials 8
python examples/prompt_optimization/train.py --epochs 2 --trials 6 --augment
python examples/prompt_optimization/rag_context.py --epochs 1 --trials 4
```

Every script takes `--base-url`, `--api-key`, `--model`, `--judge-model`,
`--rewrite-model`, `--out-dir` and `--log-level` (or `OMLX_BASE_URL`,
`OMLX_API_KEY`, `AGENTOMATIC_LOCAL_MODEL`, …). Outputs land in
`examples/prompt_optimization/out/`.

What `train.py` prints at the end (with a real model the numbers differ):

```text
Step 10/10 — results
  validation (selection): 0.000 → 0.875
  test judge    0.000 → 0.840
  test facts    0.000 → 1.000
  test quality  0.000 → 0.904
  prompt diff:
--- initial
+++ final
@@ -1 +1 @@
-You are a support assistant for Acme Cloud. Help the customer.
+You are a precise Acme Cloud support assistant. Quote the relevant policy snippet verbatim, then answer in one short sentence. Never invent policy.
  report: examples/prompt_optimization/out/train_report.html
```

What `rag_context.py` prints first, for one row (abridged):

```text
── What each stage sees — row policy_05
agent (input_to_state receives exactly this):
  {"current_query": "How long are audit logs kept?", "context": {"documents": [{"source": "audit.md", "content": "Audit logs are kept for 400 days on Enterprise and 90 days on Team."}]}, "customer_plan": "team"}
LLM judge — context documents:
  - Audit logs are kept for 400 days on Enterprise and 90 days on Team. (source: audit.md)
LLM judge — other inputs, metadata and tags:
  {"inputs": {"customer_plan": "team"}, "metadata": {"topic": "security", "difficulty": "medium", "must_include": ["90 days"]}, "tags": ["security", "compliance", "plan-specific"]}
rewrite model (dataset sample in its briefing):
1. Q: How long are audit logs kept?
   - Context documents: Audit logs are kept for 400 days on Enterprise and 90 days on Team. (source: audit.md)
   - Other inputs: {"customer_plan": "team"}
   - Example metadata: {"topic": "security", "difficulty": "medium", "must_include": ["90 days"]}
   - Tags: security, compliance, plan-specific
   Expected:
     ## Expected answer
     On the Team plan, audit logs are kept for 90 days (400 days on Enterprise).
```

…and last, the test split per tag:

```text
── Test quality by tag (before → after)
  backups         0.00 → 0.85
  plan-specific   0.00 → 0.90
  pricing         0.03 → 0.83
  refusal         0.03 → 0.20
```

## Tested

`tests/test_prompt_optimization_examples.py` runs every script as a
subprocess against a deterministic fake OpenAI-compatible server
(`tests/fake_openai_server.py`) and checks the behaviour each one teaches:
the prompt improves on validation *and* test, augmentation grows only train,
the memorising prompt is rejected, and the report contains the prompts,
candidates and per-example results. For `rag_context.py` it also checks the
requests the fake server received:

* the agent answered from each row's documents and plan;
* judges saw the documents, the plan, the metadata and the tags;
* the rewrite model's briefing contained all of them;
* augmented rows kept their seed's documents.
