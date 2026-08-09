🔍 Deep Dive: Old "Agents" Project → Agentomatic Improvement Opportunities

## Executive Summary

The old `Agents` project has a **remarkably polished optimizer/evaluation system** that's significantly more accessible and "ML-engineer-friendly" than what Agentomatic currently offers. The core design philosophy is _"feel like Keras/sklearn"_ — `agent.fit()`, callbacks, presets, auto-detection. Below are the specific features ranked by impact.

---

## 🏆 TIER 1: Game-Changing Additions (High Impact / Feasible)

### 1. ML-Style Callbacks System

**What the old project has:**
```python
# agents/default_agent/train.py
callbacks=[
    CallbackType.EARLY_STOPPING,   # Stop when no improvement for N iterations
    CallbackType.CHECKPOINT,       # Save best prompt to disk
    CallbackType.PROGRESS,         # Rich terminal progress
]
```

Full callback inventory:
| Callback | Purpose | Agentomatic status |
|---|---|---|
| `EarlyStopping` | Stop when score stagnates for `patience` rounds | ✅ `optimize.callbacks` (wired into PromptFitter) |
| `ModelCheckpoint` | Save best prompt to `optimization_results/checkpoints/` | ✅ `optimize.callbacks` |
| `NaNStopping` | Detect and halt on NaN/invalid outputs | ✅ `optimize.callbacks` |
| `TemperatureScheduler` | Anneal temperature over iterations | ✅ `optimize.callbacks` |
| `ProgressLogger` | Rich terminal with ETA, per-iteration scores | ✅ `optimize.callbacks` + RichProgressCallback |
| `PlateauStopping` | Reduce temperature when plateau detected | ✅ `optimize.callbacks` |
| `ScoreThreshold` | Early-exit when target score reached | ✅ `optimize.callbacks` + built-in loop |

**Recommendation:** Agentomatic's `PromptFitter` has internal convergence logic but no **pluggable callback system**. A Callback base class with `on_train_begin/end`, `on_iteration_begin/end`, `on_evaluation_begin/end` hooks would make the optimizer extensible and familiar to ML engineers. The old project's callback design (in `app/core/optimizer/callbacks.py`) is a clean template.

---

### 2. Universal Agent Type Auto-Detection for Evaluation

**What the old project has:**
```python
from app.core.optimizer import Evaluator

# Auto-detect agent type and select optimal metrics!
evaluator = Evaluator.for_agent(my_agent)
# Detects: STATELESS, RAG, TOOL_USING, CONVERSATIONAL, DEEP_AGENT

results = await evaluator.evaluate(my_agent, test_cases)
```

The `AgentType.detect()` method inspects agent attributes (`retriever`, `tools`, `memory`, `subagents`) and selects the right metric mix:
- **STATELESS** → `answer_relevancy`, `geval`
- **RAG** → `ragas_faithfulness`, `context_precision`, `context_recall`
- **TOOL_USING** → `tool_call_accuracy`, `tool_selection`, `answer_relevancy`
- **CONVERSATIONAL** → `answer_relevancy`, `toxicity`
- **DEEP_AGENT** → `task_completion`, `step_efficiency`

**Recommendation:** Agentomatic has `structured_metrics.py` and `judges.py` but requires manual configuration. Adding `Evaluator.for_agent()` with auto-detection would dramatically lower the barrier for users. The `AgentType` enum and detection logic from `app/core/optimizer/evaluators.py` (lines 55-102) is directly portable.

---

### 3. Rich Data Augmentation (DL-Inspired)

**What the old project has:**
```python
from app.core.optimizer.data_generator import TestCaseGenerator, AugmentationType

generator = TestCaseGenerator(config)
# 8 augmentation types:
augmented = await generator.augment(existing_cases)
```

Augmentation types the old project has that Agentomatic doesn't:
| Type | Description |
|---|---|
| `PARAPHRASE` | Rephrase input while preserving meaning |
| `ADD_NOISE` | Add typos, grammar issues (robustness) |
| `SIMPLIFY` | Make input shorter and clearer |
| `COMPLICATE` | Add details, make more complex |
| `EDGE_CASE` | Generate unusual but valid variants |
| `ADVERSARIAL` | Create inputs that might confuse the system |
| `MULTI_TURN` | Multi-turn conversation variants |

Agentomatic's `DataSynthesizer` has `paraphrase`, `perturbation`, `expansion`, `adversarial`, and `formality` — but is missing the fine-grained categorization and the **`DiversitySelector`** class that ensures test case coverage across categories and input lengths.

**Recommendation:** Add `ADD_NOISE`, `SIMPLIFY`, `COMPLICATE`, and `EDGE_CASE` to Agentomatic's synthesizer. Add the `DiversitySelector` as a utility.

---

### 4. Presets Framework

**What the old project has:**
```python
settings = OptimizerSettings.for_local()    # Free Ollama, 3 iterations
settings = OptimizerSettings.for_quality()  # GPT-4o, 10 iters, combined strategy
settings = OptimizerSettings.for_quick()    # Fast 2-iter bootstrap

settings.display()  # Prints a beautiful config table
```

These are **battle-tested presets** that encode real-world experience about what works:
- `for_local()`: Ollama, ITERATIVE_REFINEMENT, 3 iters, sequential evals
- `for_quality()`: GPT-4o, COMBINED strategy, 10 iters, 4 parallel evals
- `for_quick()`: Ollama, BOOTSTRAP_FEW_SHOT, 2 iters

**Recommendation:** Agentomatic should expose presets on the `OptimizationConfig` or equivalent. This is one of the highest-ROI changes — a one-liner that gets users 80% of the way.

---

## 🥈 TIER 2: Strong Improvements (Medium Impact)

### 5. SQLite Experiment Tracking

**What the old project has:**
```python
# All runs tracked automatically in SQLite
optimizer.show_experiments()       # Rich table of all runs
optimizer.get_best_experiment()    # Best run by score
```

The `ExperimentTracker` class in `app/core/optimizer/optimizer.py` (lines 307-570) creates a full SQLite database with:
- Experiment table (agent, strategy, model, timestamps, scores)
- Iteration table (per-iteration scores, prompts, metrics)
- Query methods: `get_experiments()`, `get_best_experiment()`, `display_experiments()`

Agentomatic has `fit_store.py` but the old project's design is **more user-facing** with explicit display commands.

**Recommendation:** Add `show_experiments()` and `get_best_experiment()` to Agentomatic's fit store, with a rich table display.

---

### 6. Per-Agent `evals.py` Auto-Discovery

**What the old project has:**
```
agents/
├── default_agent/
│   ├── agent.py
│   ├── evals.py          ← Auto-discovered!
│   └── train.py
├── condenser_agent/
│   ├── agent.py
│   ├── evals.py          ← Auto-discovered!
│   └── train.py
```

The `discover_agent_evals()` function in `evals/run_agent_evals.py` scans agent folders for `evals.py` files, dynamically imports them, and extracts `get_test_cases()`, `get_custom_metrics()`, and `THRESHOLDS`. This **co-locates evaluation with the agent** — a pattern Agentomatic could adopt.

The pytest integration (`evals/test_discovered_agents.py`) dynamically generates parametrized tests from discovered evals — run `pytest evals/ -v` and every agent gets tested.

**Recommendation:** Add an `evals.py` auto-discovery pattern so each agent ships with its own evaluation definitions.

---

### 7. Pydantic Settings with `.env` Support

**What the old project has:**
```python
# app/core/optimizer/settings.py
class OptimizerSettings(BaseSettings):
    trainer_model: str = "ollama/ministral-3:8b"
    eval_model: str = "ollama/ministral-3:8b"
    max_iterations: int = 5
    target_score: float = 0.85
    strategy: OptimizationStrategy = OptimizationStrategy.ITERATIVE_REFINEMENT
    agent_type: AgentTypeEnum = AgentTypeEnum.AUTO
    eval_metrics: list[EvalMetric] = [EvalMetric.ANSWER_RELEVANCY, EvalMetric.GEVAL]
    # ... 20+ fields

    class Config:
        env_prefix = "OPTIMIZER_"
        env_file = ".env"
```

This means users can configure the optimizer entirely via `.env`:
```bash
OPTIMIZER_TRAINER_MODEL=gemini/gemini-2.5-flash
OPTIMIZER_MAX_ITERATIONS=10
OPTIMIZER_STRATEGY=combined
```

The `settings.display()` method renders a beautiful table of all config values.

**Recommendation:** Agentomatic's `cli_settings.py` could benefit from Pydantic Settings with `.env` support and a `display()` method.

---

### 8. RobustLLMHandler with JSON Extraction

**What the old project has:**
```python
# app/core/optimizer/llm_handler.py
class RobustLLMHandler:
    async def invoke(self, prompt, expected_json=False):
        # 1. Try normal call
        # 2. Extract from code blocks
        # 3. Find JSON structure in text
        # 4. Repair malformed JSON
        # 5. Fall back to simplified prompt
```

The `JSONExtractor` class has multi-strategy extraction:
1. `_try_parse`: Direct JSON parse
2. `_extract_code_block`: Find ```json blocks
3. `_find_json_structure`: Regex for JSON-like structures
4. `_repair_json`: Fix common issues (trailing commas, unquoted keys, etc.)

The `ModelProfile` class detects model capabilities (tool calling, JSON mode, vision, etc.) and adapts prompts accordingly.

**Recommendation:** Agentomatic's `llm_caller.py` could adopt the multi-strategy JSON extraction from `JSONExtractor` and the model capability detection from `ModelProfile`.

---

## 🥉 TIER 3: Nice-to-Haves (Lower Priority)

### 9. Plotly HTML Reports
The old project generates interactive Plotly HTML reports showing score progression, metric breakdowns, prompt diffs, and training configuration. Agentomatic's `holysheet_reports.py` and `report.py` could be enhanced with interactive charts.

### 10. Prompt Version Control with Rollback
The `PromptVersionControl` class (`prompt_optimizer.py` lines 453-502) tracks prompt versions with scores and supports rollback to any previous version.

### 11. CLI Auto-Generation
`create_train_cli(agent, test_cases_fn)` auto-generates a full Click CLI for any agent's `train.py` with commands: `train`, `generate`, `experiments`, `list-models`, `evaluate`.

### 12. The `OptimizerMixin` Pattern
```python
class MyAgent(OptimizerMixin, BaseAgent):
    ...
# Now agent has fit() method
result = await agent.fit(test_cases)
```
This is the most discoverable API possible. Agentomatic's `PromptFitter.fit()` is similar but requires more scaffolding.

### 13. Makefile Training Commands
The old project's Makefile has 15+ training/evaluation commands that make the workflow dead simple:
```bash
make train-agent AGENT=default_agent
make eval-ollama
make train-experiments
```

---

## 📊 Quick Reference: What to Port

| Feature | Source File | Relative Path in old project | Priority |
|---|---|---|---|
| Callback classes | `callbacks.py` | `app/core/optimizer/callbacks.py` | 🔴 P0 |
| Auto-detect agent type | `evaluators.py:55-102` | `app/core/optimizer/evaluators.py` | 🔴 P0 |
| DiversitySelector | `data_generator.py:429-485` | `app/core/optimizer/data_generator.py` | 🔴 P0 |
| Presets (for_local etc.) | `settings.py` | `app/core/optimizer/settings.py` | 🔴 P0 |
| ExperimentTracker (SQLite) | `optimizer.py:307-570` | `app/core/optimizer/optimizer.py` | 🟡 P1 |
| Evals auto-discovery | `run_agent_evals.py` | `evals/run_agent_evals.py` | 🟡 P1 |
| Pydantic Settings | `settings.py` | `app/core/optimizer/settings.py` | 🟡 P1 |
| JSONExtractor | `llm_handler.py:126-267` | `app/core/optimizer/llm_handler.py` | 🟡 P1 |
| Plotly reports | `visualization.py` | `app/core/optimizer/visualization.py` | 🟢 P2 |
| PromptVersionControl | `prompt_optimizer.py:453-502` | `app/core/optimizer/prompt_optimizer.py` | 🟢 P2 |
| Augmentation types | `data_generator.py:32-42` | `app/core/optimizer/data_generator.py` | 🟢 P2 |
| create_train_cli | `cli.py` | `app/core/optimizer/cli.py` | 🟢 P2 |

---

## 🎯 Recommended Implementation Order

1. **Callbacks** → Add `EarlyStopping`, `ModelCheckpoint`, `NaNStopping` as pluggable hooks in `PromptFitter`
2. **Presets** → Add `for_local()`, `for_quality()`, `for_quick()` class methods on config
3. **Auto-detect** → Add `AgentType.detect()` and `Evaluator.for_agent()` 
4. **Experiment tracking** → Add SQLite-backed `ExperimentTracker` with `show_experiments()`
5. **DiversitySelector** → Add to the synthesizer module
6. **Evals auto-discovery** → Support `evals.py` per agent folder

The old project is a goldmine of UX improvements. Want me to start implementing any of these?


## Adapt agentomatic to be able to correctly handle agents classes with langchain abstractions
We need to be able to nicely and with good UX handle agents that use langchain abstractions:
- prompt template (adapt everything including prompt manager)
- messages abstractions (AIMessage, HumanMessage, chains)
- other ones (RunnableConfig , from loguru import logger
from langchain.schema import Document
from langchain_core.tools import tool, BaseTool, ToolException
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

with agents like exampl:
import os
from dotenv import load_dotenv
# from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.prebuilt.chat_agent_executor import AgentState

# Load environment variables from .env if present
load_dotenv()

# def read prompt from file 
def read_prompt_from_file(file_path: str) -> str:
    """Read a prompt from a file."""
    with open(file_path, 'r') as file:
        return file.read()

system_prompt = read_prompt_from_file("src/prompt.txt")
agent_history_system_prompt = read_prompt_from_file("src/prompt_agent_history.txt")

class State(TypedDict):
    """Main graph state."""
    messages: Annotated[list[AnyMessage], add_messages]


# llm = ChatVertexAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_tokens=None,
#     max_retries=6,
#     stop=None,
#     location="europe-west4",
#     project="ddp-genai-dev-frlm-1ji"
# )

# for local dev
llm = OllamaLLM(model="gemma3:1b")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    MessagesPlaceholder("messages")
])
llm_model = prompt_template | llm


agent_history_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    MessagesPlaceholder("messages")
]) 
history_llm_model = agent_history_prompt_template | llm


def chatbot(state: State) -> State:
    system_message = system_prompt + " : {context}"
    state["messages"] = llm_model.invoke({"system_message": system_message, "messages": state["messages"]})
    return state

def agent_history(state: State) -> State:
    """This node is used to make a story from the conversation history when having all information asked"""
    system_message = agent_history_system_prompt + " : {context}"
    state["messages"]  = history_llm_model.invoke({"system_message": system_message, "messages": state["messages"]})    
    return state
    #return {"summary": response.content, "messages": state["messages"]}



def should_continue(state: AgentState)-> Literal["agent_history", END]:
    messages = state["messages"]
    # If there is no function call, then we finish
    if len(messages) > 10:
        return "agent_history"
    else:
        return END
    
    
graph_builder=StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("agent_history", agent_history)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(START, should_continue)
graph_builder.add_edge("agent_history", END)

graph = graph_builder.compile()

(but when packaged into a class -> just to demonstrate typical used methods and approach)
