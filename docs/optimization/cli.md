# CLI & legacy API

`agentomatic optimize` optimizes an agent that is **already running** behind
the Agentomatic API, from a terminal. It uses the same `PromptFitter` engine,
datasets and acceptance rules as the Python APIs ([How it works](how-it-works.md)).
For in-process agents and full control, prefer
[class agents](class-agents.md) or [`PromptFitter`](prompt-fitter.md).

## Before you start

Install the optimization dependencies and create a dataset with a stable,
measurable expectation for each request:

```bash
pip install "agentomatic[optimize]"
```

```jsonl
{"query":"What is the refund period?","expected_answer":"30 days"}
{"query":"How do I reset my password?","expected_answer":"reset, password"}
```

`query` is required. `expected_answer`, `context` (a list of strings), and
`metadata` are optional. Use `exact_match` when the expected answer should
match exactly; use `contains` with comma-separated required phrases when a
response can vary in wording. For generative quality, use a calibrated judge
metric and a held-out test dataset.

Start the target platform before running the API-backed paths:

```bash
agentomatic run --port 8000
```

## Recommended: CLI fitter modes

The CLI has two families. `prompt_only` is the legacy default; every other
mode uses the modern `PromptFitter` path.

```bash
# Prompt + configuration rewrite against the running agent.
agentomatic optimize support_bot \
  --dataset eval.jsonl \
  --val-dataset validation.jsonl \
  --test-dataset holdout.jsonl \
  --mode rewrite \
  --metrics contains \
  --llm omlx/Qwen3.5-9B-MLX-4bit \
  --rewrite-llm omlx/Qwen3.5-9B-MLX-4bit \
  --host http://127.0.0.1:8000 \
  --max-trials 20 \
  --apply
```

`--dataset` is what the optimizer learns from, `--val-dataset` what candidates
are selected on, and `--test-dataset` the fitter's **holdout gate** — it can
only veto candidates ([Generalization](generalization.md#data-roles)). Keep a
further test set outside the command and evaluate it afterwards.

`--apply` writes a prompt version only when the result passes the fitter's
improvement and generalization guards. Omit it to inspect the report first.
The command writes reports and trial artifacts under `.optimize/` by default.

### Mode reference

| CLI mode | Fitter optimizer | Best for |
|---|---|---|
| `rewrite` | `RewriteOptimizer` | Improving system instructions from failure patterns |
| `gepa_like` | `GEPALikeOptimizer` | Targeted edits once evaluator feedback is useful |
| `mipro_like` | `MIPROLikeOptimizer` | Exploring prompt and few-shot combinations |
| `few_shot` | `FewShotBootstrapOptimizer` | Selecting strong demonstrations from labelled data |
| `param_search` | `ParamSearchOptimizer` | Searching model, RAG, or tool parameter grids |
| `apo` | `APOOptimizer` | Trace-aware critique and edit search |

`--mode` accepts exactly the values in the first column plus `prompt_only`.
The fitter API also accepts the aliases `mipro`, `gepa`, and
`few_shot_bootstrap`; prefer the CLI spellings above in new scripts.

### Parameter-only search

Use a search space to make the allowed changes explicit. This avoids an
optimization run silently tuning parameters you did not intend to change.

```yaml
# search-space.yaml
optimize_system_prompt: false
optimize_few_shot: false
optimize_model_params: true
search_method: tpe
model_param_space:
  temperature: [0.0, 0.2, 0.5]
  top_p: [0.8, 0.95]
rag_param_space:
  top_k: [3, 5, 8]
```

```bash
agentomatic optimize support_bot \
  --dataset eval.jsonl \
  --mode param_search \
  --search-space search-space.yaml \
  --search-method tpe \
  --no-optimize-prompt \
  --param temperature=0.0,0.2,0.5 \
  --max-trials 18
```

`--param` adds or replaces a model parameter grid entry. `--search-method`
accepts `grid`, `random`, or `tpe`. For `apo`, `--node-match` scopes trace
critique to matching graph-node or subagent names; `--n-runners` sets fitter
evaluation concurrency.

Run `agentomatic optimize --help` for the complete, version-specific CLI
contract. Do not use undocumented commands such as `agentomatic eval`,
`agentomatic route`, or `agentomatic promote`: they are not Agentomatic CLI
commands.

## Compatibility API: `PromptOptimizer`

`PromptOptimizer` and CLI `--mode prompt_only` remain available for existing
API-backed integrations. It supports three legacy strategies only:
`iterative_rewrite`, `few_shot`, and `chain_of_thought`.

```python
from agentomatic.optimize import Dataset, PromptOptimizer

optimizer = PromptOptimizer(
    agent="support_bot",
    metrics=["contains"],
    strategy="iterative_rewrite",
    llm="omlx/Qwen3.5-9B-MLX-4bit",
    api_base="http://127.0.0.1:8000",
)

dataset = Dataset.from_jsonl("eval.jsonl")
result = await optimizer.optimize(
    dataset=dataset,
    max_iterations=10,
    target_score=0.9,
)
print(result.report())
```

Equivalent CLI:

```bash
agentomatic optimize support_bot \
  --dataset eval.jsonl \
  --mode prompt_only \
  --strategy iterative_rewrite \
  --host http://127.0.0.1:8000
```

Do not use old names such as `mipro`, `ensemble`, or
`bootstrap_randomsearch` with `--strategy`; they are not legacy CLI strategy
values. Use the fitter modes above instead.

## Local oMLX verification

The live optimization tests exercise the actual OpenAI-compatible provider
path. Point them at a real local model rather than demo mode:

```bash
export OMLX_BASE_URL=http://127.0.0.1:8000/v1
export OMLX_API_KEY=local
export AGENTOMATIC_LIVE_MODEL=omlx/Qwen3.5-9B-MLX-4bit

uv run pytest tests/test_live_omlx_optimize.py \
  tests/test_live_omlx_keras_optimize.py \
  -q --override-ini='addopts='
```

Those tests are intentionally skipped when no reachable model endpoint is
configured. The standard test suite still verifies deterministic optimizer
logic, but a production rollout should run the live suite with a real model and
your own representative evaluation dataset.

## Production checklist

1. Keep a labelled validation set and a separate held-out test set.
2. Use a deterministic metric or a calibrated judge before trusting a score.
3. Run without `--apply` first and review the report, prompt diff, and failure
   clusters.
4. Verify the saved version in a staging deployment with the same auth,
   connections, tools, and model configuration as production.
5. Run the [deployment verifier](../guide/verifying-a-deployment.md) before promoting
   the application through your normal release process.
