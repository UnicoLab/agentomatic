# Datasets & augmentation

## The dataset file

Class-agent datasets are JSONL, one `AgentExample` per line (the format every
template's `datasets/all.jsonl` uses):

```json
{"id": "refund_2",
 "split": "validation",
 "input": {"current_query": "I bought an annual plan last week. Can I still get my money back?"},
 "expected_output": {"response": "Yes. Annual plans can be refunded within 30 days of purchase."},
 "metadata": {"must_include": ["30 days"], "topic": "refund"},
 "rubric": {"correctness": "States the 30-day window for annual plans."},
 "tags": ["billing"]}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | recommended | Stable id; generated (`example_0003`) when missing. Reports join before/after results on it. |
| `split` | recommended | `train`, `validation`, `holdout` or `test` (default `train`). See [data roles](generalization.md#data-roles). |
| `input` | yes | What the agent receives — the same dict `input_to_state()` gets. The question is read from `question`, `query`, `current_query` or `request`. `input.context` becomes the judge's retrieval context. |
| `expected_output` | for scoring | The reference answer (`{"response": "..."}`, or your structured output). Judges receive it as the reference. |
| `metadata` | no | Anything metrics need: required facts, topic, difficulty. Class-agent metrics read it (`example.metadata`). |
| `rubric` | no | Per-dimension criteria, appended to the judge's reference. |
| `tags` | no | Free-form labels. |

```python
from agentomatic.optimize import load_data

data = load_data("agents/support/datasets/all.jsonl")    # AgentDataset
data.train, data.validation, data.holdout, data.test    # lists of AgentExample
```

`PromptFitter` works on the simpler `agentomatic.optimize.Dataset` of
`DataPoint(query, expected_answer, context, metadata)`. Convert, or build one
directly:

```python
from agentomatic.optimize import Dataset

valset = Dataset(points=[e.to_datapoint() for e in data.validation])   # keeps metadata
Dataset.from_list([{"query": "Refund window?", "expected_answer": "30 days"}])
Dataset.from_jsonl("eval.jsonl")        # {"query", "expected_answer", "context", "metadata"}
trainset, valset = dataset.split(ratio=0.8)
```

## Writing good examples

* **One fact per expectation you care about.** An expected answer is what the
  judge compares against; state the facts a correct answer must contain, not
  a vague description.
* **Put checkable facts in `metadata`** (`must_include`, allowed values) so a
  deterministic metric can score them.
* **Cover the question space in every split.** Validation and holdout should
  ask about the same topics as train, in different words.
* **Size.** Validation decides; it needs enough examples for a real
  improvement to be [significant](generalization.md#the-acceptance-rules) —
  20+ if you can. Seed rows in scaffolded templates are placeholders: replace
  them before trusting a score.

## Augmentation

`prepare_dataset(augment=True)` asks an LLM for variations of your **train**
rows and adds the ones that are safe to learn from.

```python
from agentomatic.optimize import prepare_dataset

data, written = prepare_dataset(
    data,
    augment=True,
    n_examples=60,                                    # target TOTAL size
    strategies=["paraphrase", "formality_shift", "expansion"],
    model="omlx/Qwen3.5-9B-MLX-4bit",
    llm_base_url="http://127.0.0.1:8000/v1",
    llm_api_key="local",
    per_call=4,
    persist=True,
    persist_path="agents/support/datasets/all.augmented.jsonl",
)
print(data.metadata["augment_stats"])
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `augment` | `False` | Run augmentation (`False`: only persist / pass through). |
| `n_examples` | 3 × seed size | Target **total** size; only train grows. |
| `strategies` | `["paraphrase"]` | See the table below. Strategies rotate call by call, so a short run still mixes them. |
| `model` | `"openai/gpt-4o-mini"` | Augmentation LLM spec (`omlx/…`, `openai/…`, `ollama/…`, …). |
| `llm_base_url`, `llm_api_key` | process default | Endpoint for these calls only — no process-wide side effect. |
| `per_call` | `4` | Variations requested per call. Small batches finish within small local models' output limits; a truncated reply is salvaged, not dropped. |
| `max_tokens` | `4096` | Reply budget per call. |
| `strict` | `False` | Raise `RuntimeError` when fewer rows than requested were added (otherwise a warning). |
| `persist`, `persist_path`, `seed_path` | | Write the result as JSONL for review. Default path `<seed>.augmented.jsonl` (or `<seed>.prepared.jsonl` without augmentation); the seed file is never overwritten. |

### Strategies

| Strategy | Kind | Produces |
| --- | --- | --- |
| `paraphrase` | label-preserving | Same question, different words |
| `perturbation` | label-preserving | Small wording changes, typos, word order |
| `add_noise` | label-preserving | Irrelevant detail around the question |
| `simplify` | label-preserving | Shorter, plainer phrasing |
| `formality_shift` | label-preserving | More or less formal register |
| `expansion` | new question | Related questions on the same topic |
| `complicate` | new question | Multi-part or harder versions |
| `adversarial` | new question | Tricky or misleading phrasings |
| `edge_case` | new question | Boundary conditions |

**Label-preserving** rows keep the seed's `expected_output`, `metadata` and
input keys, so every metric scores them exactly like the seed. **New
question** rows get an LLM-written answer in the seed's schema — review them
before trusting scores computed on them.

### What keeps augmented data safe

* Only **train** grows; validation, holdout and test are untouched, and
  without a train split nothing is generated.
* Generated questions that duplicate an existing one, or are ≥ 90 % similar
  to any validation / holdout / test question, are dropped (a paraphrase of a
  test question in train would inflate every score measured on it).
* Each new row records `metadata.source = "augment"`, its `strategy` and its
  `parent_id`.
* `data.metadata["augment_stats"]` records `requested`, `calls`, `parsed`,
  `duplicates`, `near_duplicates`, `empty_calls`, `schema_mismatch`, `added`
  and `by_strategy`; the [report](reports.md) shows them.

Runnable: [`data_augmentation.py`](examples.md), or
`python examples/prompt_optimization/train.py --augment`.

## Generating a seed dataset from scratch

`DataSynthesizer` can draft a first dataset from a description. Always review
it and keep a human-written validation set.

```python
from agentomatic.optimize import DataSynthesizer

synth = DataSynthesizer(model="omlx/Qwen3.5-9B-MLX-4bit", base_url="http://127.0.0.1:8000/v1")
dataset = await synth.generate(
    description="A support assistant that explains refunds and account settings.",
    n_samples=40,
    categories=["refunds", "security"],
)
dataset.to_jsonl("seed.jsonl")
```
