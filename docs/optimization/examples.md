# Examples

[`examples/prompt_optimization`](https://github.com/UnicoLab/agentomatic/tree/main/examples/prompt_optimization)
contains runnable scripts, from scoring a few answers to a full training run.
Each one is commented step by step and lists every option of the methods it
uses. All of them optimize the same small `SupportAgent` (it answers Acme
Cloud customer questions from a policy knowledge base) on a 26-row dataset
split into train / validation / holdout / test.

| Script | Level | Covers | Docs |
| --- | --- | --- | --- |
| `metrics_and_judges.py` | building blocks | Every metric family, LLM judges (single, per-dimension, panel), composite objectives, adapters, losses — scored on three fixed answers | [Metrics](metrics.md) |
| `data_augmentation.py` | building blocks | `prepare_dataset` strategies and options, leak protection, `augment_stats` | [Datasets](data.md) |
| `anti_overfitting.py` | mechanics | The acceptance rules on hand-made numbers; a fit in which a prompt memorising the validation answers is vetoed and a general one kept | [Generalization](generalization.md) |
| `low_level_prompt_fitter.py` | low level | `PromptFitter` with every knob, inner-loop callbacks, before/after test evaluation, report | [PromptFitter](prompt-fitter.md) |
| `train.py` | high level | `load_data` → `prepare_dataset` → metrics → loss → `PromptFitterBridge` → `compile` → `fit` → test evaluation → `generate_fit_report` | [Class agents](class-agents.md) |

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

## Tested

`tests/test_prompt_optimization_examples.py` runs every script as a
subprocess against a deterministic fake OpenAI-compatible server
(`tests/fake_openai_server.py`) and checks the behaviour each one teaches:
the prompt improves on validation *and* test, augmentation grows only train,
the memorising prompt is rejected, and the report contains the prompts,
candidates and per-example results.
