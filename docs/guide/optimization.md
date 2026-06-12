# Prompt Optimization

Agentomatic includes a built-in prompt optimization engine that works like `model.fit()` but for prompts.

## Installation

```bash
pip install agentomatic[optimize]
```

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

## CLI

```bash
agentomatic optimize my_agent \
  --dataset qa.jsonl \
  --metrics exact_match,contains \
  --strategy iterative_rewrite \
  --max-iterations 10 \
  --rewrite-llm ollama/llama3:70b \
  --eval-llm ollama/mistral:7b \
  --apply
```

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

## Metrics

### Built-in (No LLM Required)

| Metric | Description |
|---|---|
| `exact_match` | Fuzzy or exact string matching |
| `contains` | Checks if keywords appear in response |

### DeepEval (LLM Required)

| Metric | Description |
|---|---|
| `answer_relevancy` | Is the answer relevant to the question? |
| `faithfulness` | Is the answer grounded in context? |
| `hallucination` | Does the answer contain hallucinations? |
| `contextual_relevancy` | Is retrieved context relevant? |
| `bias` | Does the response contain bias? |
| `toxicity` | Is the response toxic? |

### Custom Metrics

```python
from agentomatic.optimize import CustomMetric

def tone_check(query, response, expected, context) -> float:
    return 1.0 if "please" in response.lower() else 0.0

metric = CustomMetric(tone_check, name="politeness")
optimizer = PromptOptimizer(agent="bot", metrics=[metric, "exact_match"])
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

## Optimization Strategies

### Iterative Rewrite (Default)

The LLM analyzes failures and rewrites the prompt to address them:

1. Run dataset through agent
2. Evaluate with metrics
3. Identify low-scoring responses
4. LLM rewrites prompt to fix failures
5. Repeat

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

## HTML Reports

After optimization, an HTML report is auto-generated with:

- 📊 Score vs iteration chart (inline SVG)
- 📝 Prompt diffs between versions (side-by-side)
- 📋 Full iteration history table
- 🏆 Best prompt highlighted

```python
# Disable auto-report
optimizer = PromptOptimizer(agent="bot", auto_report=False)

# Generate manually
from agentomatic.optimize import generate_html_report
generate_html_report(result, output_path="my_report.html")
```

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

## Experiment Tracking

All runs are logged to `.optimize/{agent}/experiments.json`:

```json
[{
    "experiment_id": "abc123",
    "agent": "my_agent",
    "best_score": 0.92,
    "best_iteration": 7,
    "iterations": [...]
}]
```
