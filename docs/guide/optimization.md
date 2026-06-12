# Prompt Optimization

Agentomatic includes a built-in prompt optimization engine that works like `model.fit()` but for prompts. Inspired by [DSPy](https://github.com/stanfordnlp/dspy), it provides 7 optimization strategies, 8+ metric types, dataset synthesis, and interactive HTML reports.

```bash
pip install agentomatic[optimize]
```

---

## Quick Start

```python
from agentomatic.optimize import PromptOptimizer, Dataset

optimizer = PromptOptimizer(
    agent="my_agent",
    metrics=["exact_match", "contains"],
    strategy="iterative_rewrite",
)

result = await optimizer.optimize(
    dataset=Dataset.from_jsonl("qa.jsonl"),
    max_iterations=10,
    target_score=0.9,
)

print(result.report())   # Rich terminal report
result.apply()           # Save to prompts.json as new version
```

!!! tip "One command from the CLI"
    ```bash
    agentomatic optimize my_agent \
      --dataset qa.jsonl \
      --metrics exact_match,contains \
      --strategy iterative_rewrite \
      --max-iterations 10 \
      --apply
    ```

---

## Separate Models for Rewriting vs Evaluation

Use a powerful model for prompt rewriting and a fast model for evaluation:

```python
optimizer = PromptOptimizer(
    agent="my_agent",
    rewrite_llm="ollama/llama3:70b",   # powerful — rewrites prompts
    eval_llm="ollama/mistral:7b",      # fast — evaluates responses
    metrics=["answer_relevancy"],
)
```

---

## Dataset Format

### JSONL

```jsonl
{"query": "What is our PTO policy?", "expected_answer": "20 days per year"}
{"query": "How do I file expenses?", "expected_answer": "Use the HR portal", "context": ["Employee handbook"]}
```

### CSV

```csv
query,expected_answer
What is our PTO?,20 days
How to file expenses?,HR portal
```

### Python

```python
from agentomatic.optimize import Dataset

dataset = Dataset.from_list([
    {"query": "Q1", "expected_answer": "A1"},
    {"query": "Q2", "expected_answer": "A2", "context": ["doc1"]},
])
```

---

## Data Synthesis & Augmentation

### Generate from Description

```python
from agentomatic.optimize import DataSynthesizer

synth = DataSynthesizer(model="ollama/mistral:7b")

dataset = await synth.generate(
    description="HR assistant that answers policy questions",
    n_samples=50,
    categories=["leave", "benefits", "expenses"],
    difficulty_levels=["easy", "medium", "hard"],
)
dataset.to_jsonl("eval_data.jsonl")
```

### Generate from Documents (DeepEval)

```python
# Uses DeepEval's native Synthesizer when available
dataset = await synth.generate_from_docs(
    document_paths=["handbook.pdf", "policy.txt"],
    n_samples=50,
)
```

### Augment Existing Data

5 augmentation strategies:

```python
augmented = await synth.augment(
    dataset=existing_dataset,
    strategies=["paraphrase", "perturbation", "adversarial"],
    multiplier=5,
)
```

| Strategy | What It Does |
|---|---|
| `paraphrase` | Rephrase queries preserving intent |
| `perturbation` | Typos, informal language, abbreviations |
| `expansion` | Related follow-up questions |
| `adversarial` | Edge cases, ambiguous queries |
| `formality_shift` | Casual ↔ professional tone |

### Red Team Testing

```python
attacks = await synth.red_team(
    agent_description="HR assistant with employee data access",
    n_samples=30,
    vulnerabilities=["pii", "bias", "prompt_injection"],
)
```

---

## Metrics

### Built-in (No LLM Required)

| Metric | Description |
|---|---|
| `exact_match` | Fuzzy or exact string matching |
| `contains` | Checks if keywords appear in response |

### DeepEval Metrics (LLM Required)

| Metric | Description |
|---|---|
| `answer_relevancy` | Is the answer relevant to the question? |
| `faithfulness` | Is the answer grounded in context? |
| `hallucination` | Does the answer contain hallucinations? |
| `contextual_relevancy` | Is retrieved context relevant? |
| `bias` | Does the response contain bias? |
| `toxicity` | Is the response toxic? |

### GEval (Free-form Criteria)

```python
optimizer = PromptOptimizer(
    agent="hr_bot",
    metrics=["geval:Is the answer accurate and professional?"],
)
```

### LLM-as-Judge

```python
from agentomatic.optimize import LLMJudgeMetric

metric = LLMJudgeMetric(
    criteria="Score the response on professional tone and accuracy",
    model="ollama/mistral:7b",
    name="professionalism",
)
```

### Custom Metrics

```python
from agentomatic.optimize import CustomMetric

def tone_check(query, response, expected, context) -> float:
    return 1.0 if "please" in response.lower() else 0.0

metric = CustomMetric(tone_check, name="politeness")
optimizer = PromptOptimizer(agent="bot", metrics=[metric, "exact_match"])
```

### Wrap Any DeepEval Metric

```python
from deepeval.metrics import FaithfulnessMetric
from agentomatic.optimize import DeepEvalMetric

metric = DeepEvalMetric(FaithfulnessMetric())
optimizer = PromptOptimizer(agent="rag_bot", metrics=[metric])
```

---

## All 7 Optimization Strategies

| Strategy | Alias | Inspired By | Approach |
|---|---|---|---|
| `iterative_rewrite` | — | DSPy COPRO | LLM failure analysis → rewrite |
| `few_shot_bootstrap` | `few_shot` | DSPy BootstrapFewShot | Auto-select best examples |
| `chain_of_thought` | `cot` | DSPy CoT | Add reasoning instructions |
| `mipro` | — | DSPy MIPROv2 | N parallel candidates → fuse best |
| `bootstrap_random_search` | `random_search` | DSPy RandomSearch | Weighted random example subsets |
| `ensemble` | — | Multi-path | Run ALL strategies → fuse results |

### Iterative Rewrite (Default)

The LLM analyzes failures and rewrites the prompt to address them:

1. Run dataset through agent
2. Evaluate with metrics
3. Identify low-scoring responses
4. LLM rewrites prompt to fix failures
5. Repeat until `target_score` reached or `max_iterations` exhausted

```python
optimizer = PromptOptimizer(
    agent="my_agent",
    strategy="iterative_rewrite",
    metrics=["answer_relevancy"],
)
```

### Few-Shot Bootstrap

Automatically selects the best examples as few-shot demonstrations:

```python
optimizer = PromptOptimizer(
    agent="my_agent",
    strategy="few_shot",
    metrics=["exact_match"],
)
```

### Chain-of-Thought

Adds step-by-step reasoning instructions:

```python
optimizer = PromptOptimizer(
    agent="my_agent",
    strategy="chain_of_thought",
    metrics=["answer_relevancy"],
)
```

### MIPRO

Generates N parallel prompt candidates and fuses the best:

```python
optimizer = PromptOptimizer(
    agent="my_agent",
    strategy="mipro",
    metrics=["faithfulness", "answer_relevancy"],
)
```

### Bootstrap Random Search

Samples weighted random subsets of examples:

```python
optimizer = PromptOptimizer(
    agent="my_agent",
    strategy="random_search",
    metrics=["exact_match"],
)
```

### Ensemble

Runs ALL strategies and fuses the best results:

```python
optimizer = PromptOptimizer(
    agent="my_agent",
    strategy="ensemble",
    metrics=["answer_relevancy", "faithfulness"],
)
```

---

## Per-Agent Optimization Pattern

Optimize each agent in your platform independently:

```python
from agentomatic import AgentPlatform
from agentomatic.optimize import PromptOptimizer, Dataset

platform = AgentPlatform.from_folder("agents/")
app = platform.build()

# Optimize each agent with its own dataset
for agent_name in ["hr_bot", "rag_agent", "classifier"]:
    optimizer = PromptOptimizer(
        agent=agent_name,
        metrics=["answer_relevancy", "faithfulness"],
        strategy="iterative_rewrite",
        rewrite_llm="ollama/llama3:70b",
        eval_llm="ollama/mistral:7b",
    )
    result = await optimizer.optimize(
        dataset=Dataset.from_jsonl(f"data/{agent_name}_eval.jsonl"),
        max_iterations=10,
        target_score=0.9,
    )
    if result.improved:
        result.apply()
        print(f"✅ {agent_name}: {result.baseline_score:.0%} → {result.best_score:.0%}")
```

---

## Prompt Versioning

Optimized prompts are saved as new versions in `prompts.json`:

```json
{
    "v1": {"system": "Original prompt", "user_template": "{query}"},
    "v1_optimized": {
        "system": "Improved prompt from optimization",
        "user_template": "{query}",
        "_metadata": {
            "optimization": {
                "score": 0.92,
                "baseline_score": 0.65,
                "improvement_pct": 41.5,
                "iterations": 7
            }
        }
    }
}
```

---

## Comparing Prompt Versions

```python
results = await optimizer.compare_prompts(
    dataset=dataset,
    prompts={
        "v1_original": "You are a helpful assistant.",
        "v2_detailed": "You are a precise, detailed assistant...",
        "v3_concise": "You are a concise assistant...",
    },
)
# Prints a Rich comparison table with per-metric scores
```

---

## HTML Reports (HolySheet)

After optimization, an interactive HTML report is auto-generated using [HolySheet](https://github.com/UnicoLab/holysheet):

- 📊 Interactive score vs iteration chart (ECharts)
- 📝 Prompt diffs between versions
- 📋 Full iteration history with KPI cards
- 🏆 Best prompt highlighted

Falls back to inline SVG reports if HolySheet is not installed.

```python
# Disable auto-report
optimizer = PromptOptimizer(agent="bot", auto_report=False)

# Generate manually
from agentomatic.optimize import generate_html_report
generate_html_report(result, output_path="my_report.html")
```

---

## CLI Usage

Full optimization from the command line:

```bash
# Basic optimization
agentomatic optimize my_agent \
  --dataset qa.jsonl \
  --metrics exact_match,contains \
  --strategy iterative_rewrite \
  --max-iterations 10 \
  --apply

# With separate rewrite/eval models
agentomatic optimize my_agent \
  --dataset qa.jsonl \
  --metrics answer_relevancy,faithfulness \
  --strategy mipro \
  --rewrite-llm ollama/llama3:70b \
  --eval-llm ollama/mistral:7b \
  --target-score 0.9 \
  --apply

# Red team testing
agentomatic optimize my_agent \
  --red-team \
  --vulnerabilities pii,bias,prompt_injection \
  --n-samples 30
```

---

## DeepEval Native Integration

The module uses DeepEval natively when installed:

- **Metrics**: `LLMTestCase`, `GEval`, `AnswerRelevancyMetric`, etc.
- **Synthesizer**: `generate_goldens_from_docs()` for document-based dataset creation
- **Red Teaming**: `RedTeamer` for 40+ vulnerability scans
- **Dataset bridge**: Convert between agentomatic `Dataset` ↔ DeepEval `EvaluationDataset`

---

## Experiment Tracking

All runs are logged to `.optimize/{agent}/experiments.json`:

```json
[{
    "experiment_id": "abc123",
    "agent": "my_agent",
    "best_score": 0.92,
    "best_iteration": 7,
    "rewrite_llm": "ollama/llama3:70b",
    "eval_llm": "ollama/mistral:7b",
    "iterations": [...]
}]
```
