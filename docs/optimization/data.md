# Datasets & augmentation

## The dataset file

Class-agent datasets are JSONL, one `AgentExample` per line. Every
template's `datasets/all.jsonl` uses this format. A row carries the whole
case: the question, the documents the answer must come from, any other agent
input, labels, tags and a rubric. All of it reaches the optimization, not
just the question and the answer.

```json
{"id": "policy_05",
 "split": "train",
 "input": {"current_query": "How long are audit logs kept?",
           "context": {"documents": [{"source": "audit.md",
               "content": "Audit logs are kept for 400 days on Enterprise and 90 days on Team."}]},
           "customer_plan": "team"},
 "expected_output": {"response": "On the Team plan, audit logs are kept for 90 days (400 days on Enterprise)."},
 "metadata": {"topic": "security", "difficulty": "medium", "must_include": ["90 days"]},
 "tags": ["security", "compliance", "plan-specific"],
 "rubric": {"groundedness": "Every claim comes from the row's context documents.",
            "plan": "Applies the policy to the customer's plan when one is given."}}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | recommended | Stable id; generated (`example_0003`) when missing. Reports join before/after results on it. |
| `split` | recommended | `train`, `validation`, `holdout` or `test` (default `train`). See [data roles](generalization.md#data-roles). |
| `input` | yes | What the agent receives, unchanged: the same dict `input_to_state()` gets. The question is read from `question`, `query`, `current_query` or `request`. |
| `input.context` | no | Documents the answer must come from (RAG). The value can be `{"documents": [...]}`, a list, or one string. A document is a string or a dict with `content` (or `text` / `page_content`) and an optional `source`. |
| other `input.*` keys | no | Any other agent input: `messages` (earlier turns), a plan, a locale, a user profile, and so on. |
| `expected_output` | for scoring | The reference answer (`{"response": "..."}`, or your structured output). Judges receive it as the reference. |
| `metadata` | no | Labels metrics need: required facts (`must_include`), topic, difficulty. |
| `tags` | no | Grouping labels. Filter with `data.filter_by_tags(...)`, and break results down by tag. |
| `rubric` | no | Per-dimension criteria, added to the judge's reference. |

```python
from agentomatic.optimize import load_data

data = load_data("agents/support/datasets/all.jsonl")    # AgentDataset
data.train, data.validation, data.holdout, data.test    # lists of AgentExample
billing = data.filter_by_tags("billing")                # any-match
```

`PromptFitter` works on the simpler `agentomatic.optimize.Dataset` of
`DataPoint(query, expected_answer, context, metadata, tags)`. Convert, or
build one directly:

```python
from agentomatic.optimize import Dataset

valset = Dataset(points=[e.to_datapoint() for e in data.validation])   # keeps everything
Dataset.from_list([{"query": "Refund window?", "expected_answer": "30 days",
                    "context": ["Refunds: 30 days."], "tags": ["billing"],
                    "metadata": {"invoke": {"customer_plan": "team"}}}])
Dataset.from_jsonl("eval.jsonl")   # {"query", "expected_answer", "context", "metadata", "tags"}
trainset, valset = dataset.split(ratio=0.8)
```

`to_datapoint()` maps a row as follows:

| Row field | `DataPoint` field |
| --- | --- |
| `input` (except the question) | `metadata.invoke`, sent to the agent exactly as written |
| `input.context` | `context`, as document texts such as `"Audit logs are kept … (source: audit.md)"` |
| `metadata` | `metadata` |
| `tags` | `tags` |
| `rubric` | the expected reference, ahead of the answer, for the judge |

## What each stage sees

| Stage | Receives |
| --- | --- |
| **Agent** | `input` exactly as written: the question, `context`, `customer_plan` and the rest. Evaluation hands it a copy, so an agent that edits its input never changes the dataset. |
| **Metrics** | The whole row (`example.input`, `.metadata`, `.tags`, `.rubric`), the same inside `fit()`, in `evaluate()` and nested in a composite. |
| **LLM judges** | Question, answer, expected answer plus rubric, and the context documents. If the row has no documents, they get what the agent retrieved instead. Inside `fit()` / `evaluate()` they also get the other inputs, the metadata and the tags. |
| **Rewrite model** | Every failure, success and dataset sample in its [briefing](how-it-works.md) shows `Context documents`, `Other inputs`, `Example metadata`, `Tags` and what the agent retrieved. It is told to write instructions that *use* such context and never to copy one row's facts into the prompt. |
| **Augmenter** | The seed's documents and other inputs (a "Seed context" section). Variations inherit a copy of them, plus the seed's tags (and `augmented`) and rubric. |
| **Report** | A `context` column in the per-example table: tags, documents, inputs and metadata. |

## RAG datasets

There are two ways for a RAG agent to get its documents. Both work with every
stage above.

* **The row brings its documents** (`input.context.documents`). The agent
  answers from exactly those. The judge checks groundedness against them, and
  the rewrite model reads them next to each answer. Use this to optimize the
  *answering* prompt independently of retrieval quality, or when the gold
  passages are known. The `rag` template reads them in `input_to_state`
  (`context.documents`, flattened to `documents` over REST) and falls back to
  its own retrieval when there are none.
* **The agent retrieves for itself.** List what it used in the output under
  `retrieval_context`, `citations`, `sources` or `documents`. These can be
  strings or dicts with `content` and `source`. Judges then score
  groundedness against those, and the rewrite model sees them as
  `Retrieved by the agent`.

```python
def input_to_state(self, data):
    ctx = data.get("context") or {}
    docs = ctx.get("documents") if isinstance(ctx, dict) else ctx
    return MyState(question=data.get("current_query", ""), documents=list(docs or []))

def state_to_output(self, state):
    return {"response": state.answer, "citations": state.citations}   # what it used
```

Metrics can read the documents too. For example, a groundedness check that
counts how many of the answer's words come from them:
[`grounded_score`](examples.md).

`examples/prompt_optimization/rag_context.py` runs the whole loop on rows
that bring their own documents, plan, metadata, tags and rubric. It prints
what each stage sees and reports test scores per tag.

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
* The generator sees each seed's documents and other inputs (a "Seed
  context" section) and is told to ask only what those documents answer.
  Every variation inherits a copy of its seed's inputs, rubric and tags (plus
  `augmented`), so a RAG row stays answerable.
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
