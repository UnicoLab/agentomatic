<div align="center">

<p align="center">
  <img src="assets/logo.png" width="300" alt="agentomatic">
</p>

# ⚡ Agentomatic

### Drop agents, not code

[![CI](https://github.com/UnicoLab/agentomatic/actions/workflows/ci.yml/badge.svg)](https://github.com/UnicoLab/agentomatic/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentomatic.svg)](https://pypi.org/project/agentomatic/)
[![Python](https://img.shields.io/pypi/pyversions/agentomatic.svg)](https://pypi.org/project/agentomatic/)
[![License](https://img.shields.io/github/license/UnicoLab/agentomatic.svg)](https://github.com/UnicoLab/agentomatic/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://unicolab.github.io/agentomatic)

**The Zero-Code Multi-Agent API & Observability Framework.**
Build, trace, optimize, and time-travel debug production-ready AI agent APIs in just 3 lines of code. Agentomatic natively provides auto-discovery, auto-routing, dynamic streaming, a built-in visual Studio, and A2A protocols right out of the box.

[Documentation](https://unicolab.github.io/agentomatic) · [Agentomatic Studio](#-agentomatic-studio) · [Quick Start](#-quick-start) · [CLI Reference](#-cli) · [Templates](#-templates) · [Contributing](CONTRIBUTING.md)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Agentomatic Studio** | Embedded visual agent debugger with graph rendering, live SSE node streaming, state mutation, and historical time-travel capabilities. |
| ⚡ **Prompt Optimizer** | Enterprise-grade prompt and configuration fitting utilizing 5 distinct optimizers with deployment recommendations. |
| 🔍 **Zero-Code Auto-Discovery** | Drop an agent folder → 26 fully-documented REST endpoints appear automatically. |
| 🚀 **Rich API Surface** | Natively handles `invoke`, `stream`, `chat`, `A2A`, `health`, `config`, `threads`, `memory`, and `feedback`. |
| 🧵 **Universal Execution Modes** | Every agent, plugin, pipeline, and endpoint can run **sync**, **async**, **batch**, **streaming**, or as a **background task** — automatically, no extra code. |
| 📮 **Task Board** | Unified `/api/v1/tasks` API: submit, poll status/progress, stream SSE events, cancel, and receive completion webhooks — with a pluggable, durable `TaskStore`. |
| 🩺 **Unified Status Dashboard** | One `/status` HTML page + `/api/v1/status` JSON covering every agent, plugin, pipeline, endpoint, ingestor, storage, and the task engine. |
| 📥 **Ingestion / RAG Packaging** | Bring any library (PDF→markdown, loaders, embedders); Agentomatic packages it as a discoverable ingestor callable sync/async/as-a-task. |
| 🧱 **Composable Pipelines** | Chain agents, plugins, endpoints, ingestors, transforms, loops, and sub-pipelines with typed data-passing, conditionals, retries, rollback/compensation, and schema enforcement. |
| 🗄️ **Pluggable Storage** | Use `MemoryStore`, `SQLAlchemy`, or plug in your own custom persistence layer. |
| 🔐 **Enterprise Middleware** | High-performance pipeline with JWT Auth, dynamic rate limiting, and Prometheus telemetry — all toggleable. |
| 📦 **Scaffolding Templates** | Jumpstart with 14 templates: `basic`/`class`, `full`, `coordinator`, `pipeline`, `rag`, `chatbot`, `deepagent`, `custom`, `legacy_dict`, `plugin`, `endpoint`, `connection`, `ingestion`, `extraction`. |
| 🧬 **Class-Based Agents** | Define agents as Python classes with a **Keras-style ML lifecycle**: `compile()` → `fit()` (epochs, `verbose`, callbacks, `validation_data`) → `evaluate()` → `transform()`, returning a real `History` object. |
| 🤖 **A2A Protocol** | True Agent-to-Agent communication flows integrated out of the box. |
| 🔌 **Framework Agnostic** | Fully supports LangGraph, LangChain, or raw Python execution logic. |
| 🩺 **Beautiful CLI** | A rich terminal experience with commands like `doctor`, `inspect`, and `test`. |
| 🧪 **Data Synthesizer** | Auto-generate and systematically augment evaluation datasets using LLMs. |
| 📊 **Observability HTML Reports** | Generate rich SVG charts, prompt diffs, and deep experiment tracking analytics. |
| 🚦 **Human-in-the-Loop** | Seamlessly suspend, intercept, and resume execution with human approval gates. |
| 🌳 **Thread Lineage** | First-class parent/child conversation tracking with recursive ancestry traversal. |
| ⏰ **HITL TTL Expiry** | Automatic garbage collection and cleanup of stale suspended states (7-day default). |
| 🛡️ **LLM Failover Chains** | Multi-provider fallback pipelines to guarantee extreme runtime resilience. |
| 🧬 **Thread Forking** | Clone conversations and branch execution at any specific message index natively. |
| 🔀 **A/B Prompt Routing** | Dynamically inject weight-based prompt version selection to test optimizations in production. |
| 🪝 **State Hooks** | Before/after node interceptors designed specifically for robust audit and telemetry logs. |
| 🧠 **Conversation Memory** | Automatic short-term session logic paired with long-term memory windowing. |
| 📝 **Auto-Summarization** | Intelligent LLM-powered compression of excessively long conversations to save token limits. |
| 📋 **Thread CRUD** | Full lifecycle management (`create`, `update`, `delete`, `clear`). |
| 💬 **Message Persistence** | Every conversational turn is automatically saved to storage — ensuring history survives system restarts perfectly. |

## 🚀 Quick Start

### Install

```bash
pip install agentomatic[all]
```

### Create an Agent

```bash
agentomatic init my_agent --template basic
```

### Build & Run

```python
# main.py
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/")
app = platform.build()
```

```bash
uvicorn main:app --reload
```

### Test

```bash
# CLI
agentomatic test my_agent

# curl
curl -X POST http://localhost:8000/api/v1/my_agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentPlatform                            │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Registry │  │ Middleware   │  │ Storage               │ │
│  │          │  │ ├─ Auth      │  │ ├─ MemoryStore        │ │
│  │ agent_a  │  │ ├─ RateLimit │  │ ├─ SQLAlchemyStore   │ │
│  │ agent_b  │  │ ├─ Metrics   │  │ └─ YourStore(ABC)    │ │
│  │ agent_c  │  │ └─ Logging   │  │                       │ │
│  └──────────┘  └──────────────┘  └───────────────────────┘ │
│                                                             │
│  Per Agent: POST /invoke, /stream, /chat, /a2a/tasks ...   │
└─────────────────────────────────────────────────────────────┘
```

## 📂 Agent Structure

Only `agent.py` is required. Everything else is optional overrides:

```
agents/my_agent/
├── __init__.py      ← Optional: Python package init
├── agent.py         ← REQUIRED: Contains your BaseGraphAgent subclass
├── config.py        ← Optional: Pydantic config
├── schemas.py       ← Optional: custom request/response models
├── tools.py         ← Optional: LangChain tools
├── api.py           ← Optional: custom router (REPLACES auto-gen)
├── prompts.json     ← Optional: versioned prompt templates
├── langgraph.json   ← Optional: LangGraph Studio config
├── .env.example     ← Optional: environment variables
└── README.md        ← Optional: agent documentation
```

## 📦 Templates

```bash
agentomatic init my_agent --template <template>
```

| Template | Description |
|----------|-------------|
| `basic` | Minimal class-based agent (recommended) — quick start |
| `full` | All override files — class agent with config, schemas, api, tools, prompts |
| `rag` | RAG class-based agent — retrieve → generate pipeline |
| `chatbot` | Conversational class-based agent with memory |
| `deepagent` | Deep Agent — planning, tools, subagents (requires deepagents package) |
| `custom` | Framework-agnostic — no LangGraph dependency |
| `legacy_dict` | Legacy functional agent — 3 files (`__init__`, graph, nodes) |
| `plugin` | ML Model Plugin — wrap classical ML models with REST endpoints |

## 🖥️ CLI

```
⚡ Agentomatic — Drop agents, not code

  init <name>      Scaffold a new agent from template
  run              Start the platform server
  run --studio     Start with Agentomatic Studio visual debugger 🎨
  run --with-ui    Start with Chainlit chat interface 💬
  demo             Launch demo platform with Studio (no setup needed)
  list             List discovered agents (Rich table)
  test <name>      Interactive terminal testing
  inspect <name>   Show agent structure + config
  doctor           Environment health check
  optimize <name>  Run prompt optimization
  ui               Launch Chainlit debug UI standalone
  pipeline         Pipeline management commands
```

## 🧬 Class-Based Agents (NEW)

Define agents as Python classes with built-in graph wiring and ML lifecycle:

```python
from dataclasses import dataclass, field
from agentomatic import BaseGraphAgent, EarlyStopping

@dataclass
class MyState:
    query: str = ""
    output: dict = field(default_factory=dict)

class MyAgent(BaseGraphAgent[MyState]):
    agent_name = "my_agent"

    def build_graph(self):
        g = self.new_graph()
        g.add_node("process", self.process)
        g.add_node("format", self.format_out)
        g.set_entry_point("process")
        g.add_edge("process", "format")
        g.set_finish_point("format")
        return g.compile()

    def process(self, state):
        state.output = {"response": f"Hello! You asked: {state.query}"}
        return state

    def format_out(self, state):
        return state

    def input_to_state(self, data):
        return MyState(query=data.get("query", ""))

    def state_to_output(self, state):
        return state.output

# Keras-style ML workflow
agent = MyAgent()
result = agent.transform({"query": "Hello!"})

agent.compile(dataset=dataset, metrics=[accuracy], loss=my_loss)
history = agent.fit(
    dataset,
    epochs=5,
    verbose=1,                       # Keras-like per-epoch log lines
    validation_data=valset,          # adds val_* metrics
    callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
)
print(history.best("val_loss", mode="min"))
report = agent.evaluate(dataset.test, metrics)
agent.save("compiled/v1")
```

`fit()` runs the prompt optimizer under the hood (via `PromptFitterBridge`),
records per-epoch metrics/loss into a `History` object (also on `agent.history`),
fires `Callback` hooks, and supports `EarlyStopping` — treating a GenAI agent
like a classical trainable model. See the
[class-based agents guide](https://unicolab.github.io/agentomatic/guide/class-agents/).

## 🎨 Agentomatic Studio

Agentomatic ships with a built-in React-based visual studio designed for time-travel debugging, real-time node streaming, and state inspection. Works with class-based agents, LangGraph, LangChain, and any custom framework via the adapter system.

To use the studio, install the optional package dependencies and run with the `--studio` flag:

```bash
pip install "agentomatic[studio]"
agentomatic run --studio
```

The unified server will bind to `http://localhost:8000` and mount the studio at `http://localhost:8000/studio/ui/`.

**Key Studio Features**:
- **Live Node Streaming**: Watch Server-Sent Events (SSE) transition node activity dynamically.
- **Conditional Breakpoints**: Right-click graph nodes to intercept flow before execution triggers.
- **Time-Travel History**: Rewind to any state checkpoint and replay from historical forks.
- **Live State Editing**: Mutate graph state payloads on the fly during a breakpoint pause.

## 🧠 ML Model Plugins (NEW)

Agentomatic isn't just for LLMs. Wrap classical ML models (Scikit-Learn, PyTorch, PyMC) securely with auto-generated REST endpoints:

```python
from agentomatic.plugins import BaseMLPlugin
from pydantic import BaseModel

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPlugin(BaseMLPlugin[IrisInput, dict]):
    async def load_model(self):
        # Load sklearn model from disk
        import joblib
        self.model = joblib.load("iris_model.pkl")

    async def predict(self, inputs: IrisInput) -> dict:
        prediction = self.model.predict([[
            inputs.sepal_length, inputs.sepal_width,
            inputs.petal_length, inputs.petal_width
        ]])
        return {"species": prediction[0]}
```

Place it in `plugins/` and Agentomatic auto-discovers it alongside your AI agents!

## 🧵 Execution Modes & Task Board

Every resource — **agents, plugins, pipelines, endpoints, and ingestors** — is
automatically callable in **every** execution mode. No extra code:

| Mode | How | Use case |
|------|-----|----------|
| **Sync** | `POST /api/v1/{agent}/invoke` | Immediate request/response |
| **Streaming** | `POST /api/v1/{agent}/invoke/stream` (SSE) | Token/node streaming |
| **Async task** | `POST /api/v1/{agent}/invoke/async` | Fire-and-forget, poll later |
| **Batch** | `POST /api/v1/{agent}/invoke/batch` | Many inputs, bounded concurrency |
| **A2A** | `POST /api/v1/{agent}/a2a/tasks` | Agent-to-agent task protocol |

Async work is tracked by a unified **task board** — ideal for long-running jobs
like document ingestion where the frontend polls for progress:

```bash
# Submit an async task (returns immediately with a task id)
curl -X POST http://localhost:8000/api/v1/my_agent/invoke/async \
  -H "Content-Type: application/json" -d '{"query": "long job"}'
# → {"id": "task_ab12...", "status": "queued", ...}

# Poll status + progress
curl http://localhost:8000/api/v1/tasks/task_ab12...
# → {"status": "running", "progress": {"percent": 42, "message": "chunking"}}

# Stream live progress events (SSE)
curl -N http://localhost:8000/api/v1/tasks/task_ab12.../events

# Cancel
curl -X POST http://localhost:8000/api/v1/tasks/task_ab12.../cancel
```

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/tasks` | Submit a task for any target |
| `GET` | `/api/v1/tasks` | List/filter tasks (status, target, …) |
| `GET` | `/api/v1/tasks/{id}` | Task status, progress, result |
| `GET` | `/api/v1/tasks/{id}/result` | Terminal result payload |
| `GET` | `/api/v1/tasks/{id}/events` | SSE progress events |
| `POST` | `/api/v1/tasks/{id}/cancel` | Request cancellation |
| `DELETE` | `/api/v1/tasks/{id}` | Delete a task record |

Tasks report `progress` (`percent`, `message`, `stage`), support **completion
webhooks** (`callback_url`), and persist through a pluggable `TaskStore`
(in-memory by default, or the durable `SQLAlchemyTaskStore`). See the
[Tasks guide](https://unicolab.github.io/agentomatic/guide/tasks/).

## 🩺 Unified Status Dashboard

A single control-plane view of the whole platform's health:

```bash
open http://localhost:8000/status          # HTML dashboard
curl http://localhost:8000/api/v1/status   # JSON API
```

Aggregates the health of every **agent, plugin, pipeline, endpoint, ingestor**,
the **storage backend**, and the **task engine** (queue depth, running/terminal
counts) into one page — with per-resource drill-down and an overall
`healthy` / `degraded` / `unhealthy` roll-up.

## 📥 Ingestion & RAG

Agentomatic is about **ops, not implementation**: bring your favourite libraries
(PDF→markdown, loaders, splitters, embedders, vector stores) and Agentomatic
*packages* them as a discoverable **ingestor** that is callable sync, async, or
as a tracked task — and usable as a pipeline step.

```python
from agentomatic.ingestion import BaseIngestor, IngestionRequest, IngestionResult

class DocsIngestor(BaseIngestor):
    ingestor_name = "docs"

    async def ingest(self, request: IngestionRequest, ctx) -> IngestionResult:
        # Reuse ANY library you like:
        text = my_pdf_lib.to_markdown(request.source)     # extract
        chunks = my_splitter.split(text)                  # chunk
        vectors = my_embedder.embed(chunks)               # embed
        await my_store.upsert(vectors)                    # persist
        await ctx.report(percent=100, message="done")     # progress → task board
        return IngestionResult(documents=1, chunks=len(chunks), upserted=len(vectors))
```

Drop it in `ingestion/`, and it's auto-discovered with its own endpoints,
task support, and pipeline step. See the
[Ingestion guide](https://unicolab.github.io/agentomatic/guide/ingestion/).

## 🧱 Pipelines

Compose agents, plugins, endpoints, ingestors, transforms, loops, and
sub-pipelines into a single graph with **full control over data-passing**:

```yaml
# pipelines/rag_ingest.yaml
name: rag_ingest
strict_schema: true
on_error: rollback
steps:
  - ingestion: docs            # reuse your ingestor
    input: { source: "{{ input.path }}" }
    output: ingested
  - agent: summarizer
    input: { text: "{{ ingested.summary }}" }
    output: summary
    retry: { max_attempts: 3 }
    rollback: "await store.delete(ingested.id)"   # compensation
  - plugin: classifier         # call an ML plugin mid-pipeline
    input: { features: "{{ summary }}" }
```

Supports input/output mapping, shared context, conditionals, retries, timeouts,
`on_error` policies (**including rollback/compensation**), optional input/output
**schema enforcement**, and per-step async execution — all runnable via the same
sync/async/streaming/task modes. See the
[Pipelines guide](https://unicolab.github.io/agentomatic/guide/pipelines/).

## ⚙️ Configuration

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore  # or SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    # Storage
    store=MemoryStore(),
    # Auth
    enable_auth=True,
    auth_api_key="your-secret-key",
    # Rate limiting
    enable_rate_limit=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    # Prometheus metrics
    enable_metrics=True,
    # Custom middleware
    middleware=[(MyMiddleware, {"arg": "value"})],
)
app = platform.build()
```

## 🗄️ Storage Backends

```python
# Development
from agentomatic.storage import MemoryStore
store = MemoryStore()

# Production (PostgreSQL)
from agentomatic.storage import SQLAlchemyStore
store = SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db")

# Custom
from agentomatic.storage import BaseStore
class RedisStore(BaseStore):
    async def create_thread(self, ...): ...
    async def get_thread(self, ...): ...
```

**Durable task storage** (for background tasks that must survive restarts or be
shared across workers) uses a separate, pluggable `TaskStore`:

```python
from agentomatic.tasks import SQLAlchemyTaskStore

platform = AgentPlatform.from_folder(
    "agents/",
    task_store=SQLAlchemyTaskStore("postgresql+asyncpg://user:pass@localhost/db"),
)
```

Defaults to an in-memory store; install with `agentomatic[db]` (SQLite) or
`agentomatic[db-postgres]` (PostgreSQL). See
[Tasks → Persistence](https://unicolab.github.io/agentomatic/guide/tasks/#persistence-durability).

## 🎨 Debug UI

Built-in ChatGPT-like interface powered by Chainlit:

```bash
pip install agentomatic[ui]
agentomatic run --with-ui
# → http://localhost:8000/chat
```

Features: agent selector, streaming, tool call visualization, chain-of-thought, feedback collection.

## 🎨 Agentomatic Studio

Visual debugging environment for **any agent framework** — graph visualization, real-time execution tracing, state inspection, and time-travel debugging.

```bash
# Quick demo (no setup required)
agentomatic demo

# With your agents
agentomatic run --studio
# → Studio at http://localhost:8000/studio/ui/
```

**Universal Framework Support:**

| Feature | LangGraph | LangChain | Custom / Raw Python |
|---|:---:|:---:|:---:|
| Graph Visualization | ✅ Real graph | ✅ LCEL / synthetic | ✅ Synthetic or `@studio_graph` |
| SSE Node Streaming | ✅ Full | ✅ `astream_events` | ✅ Trace-based |
| State Inspection | ✅ Checkpointer | ✅ I/O capture | ✅ Custom or in-memory |
| Time-Travel History | ✅ Checkpoints | ✅ Traces | ✅ Traces |
| Breakpoints / HITL | ✅ | ❌ | ❌ |

**Studio API Endpoints** (mounted at `/studio/`):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/studio/info` | Server info + capabilities |
| `GET` | `/studio/agents` | List agents with debugging capabilities |
| `GET` | `/studio/agents/{name}/graph` | Graph topology (nodes, edges) |
| `GET` | `/studio/agents/{name}/schemas` | Input/output JSON schemas |
| `POST` | `/studio/agents/{name}/runs/stream` | Execute with SSE event streaming |
| `GET` | `/studio/agents/{name}/threads/{tid}/state` | Thread state snapshot |
| `GET` | `/studio/agents/{name}/threads/{tid}/history` | Checkpoint history |

**Studio Decorators** — incrementally upgrade any agent's Studio experience:

```python
from agentomatic.studio import studio_graph, studio_state

@studio_graph
def my_topology():
    return {"nodes": [...], "edges": [...]}

@studio_state
async def get_state(thread_id: str) -> dict:
    return await my_db.get_state(thread_id)
```

## 📊 Auto-Generated Endpoints

Every agent gets 12+ endpoints automatically:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/{agent}/invoke` | Synchronous invocation |
| `POST` | `/api/v1/{agent}/invoke/stream` | SSE streaming |
| `POST` | `/api/v1/{agent}/chat` | Session-aware chat |
| `GET` | `/api/v1/{agent}/health` | Per-agent health |
| `GET` | `/api/v1/{agent}/card` | A2A agent card |
| `POST` | `/api/v1/{agent}/a2a/tasks` | A2A task submission |
| `GET` | `/api/v1/{agent}/threads` | List threads |
| `POST` | `/api/v1/{agent}/threads/{id}/approve` | HITL: approve suspended state |
| `POST` | `/api/v1/{agent}/threads/{id}/reject` | HITL: reject suspended state |
| `GET` | `/api/v1/{agent}/threads/{id}/pending` | HITL: list pending approvals |
| `POST` | `/api/v1/{agent}/threads/{id}/fork` | Fork thread at message index |
| `GET` | `/api/v1/{agent}/threads/{id}/lineage` | Thread ancestry/descendant tree |
| ... | ... | + config, prompts, thread messages |

### 🔧 Prompt Fitting (ML-like API)

Agentomatic Optimize treats your deployed agent configuration as a **parameter surface**
to fit against real evaluation data. The output is never a compiled program — it's a
**better deployment configuration**: an improved prompt, tuned model parameters,
optimized RAG settings, and a rollout recommendation you can ship with confidence.

**Recommended entrypoints** for class agents (scaffolded as `train.py` / `eval.py`):

```python
from agentomatic.optimize import (
    TrainConfig, train_and_report, print_train_result,
    EvalConfig, evaluate_and_report,
)

result = train_and_report(agent, config=TrainConfig(
    agent_name="assistant", agent_dir=HERE, stacks_dir=ROOT / "stacks",
    epochs=2, max_trials=12, optimizer="rewrite",
    augment=True, n_examples=40, persist=True,
))
print_train_result(result)  # HolySheet HTML at result.report_path

ev = evaluate_and_report(agent, config=EvalConfig(
    agent_name="assistant", agent_dir=HERE, stacks_dir=ROOT / "stacks",
    split="test", prefer_augmented=True,
))
```

See the [Prompt Optimization guide](https://unicolab.github.io/agentomatic/guide/optimization/).

> **Philosophy:** Your agent is already deployed. Optimization produces a *better version*
> of that deployment, not a new artifact. Every result includes a `DeploymentRecommendation`
> with canary weights and confidence scores so you can roll out safely.

#### EvalContract — Structural Quality Gate

Define what a valid agent response looks like *before* you optimize:

```python
from agentomatic.optimize import EvalContract

contract = EvalContract(
    name="scoping_response",
    input_fields=["query", "context"],
    output_format="json",
    required_output_fields=["answer", "confidence", "risks", "next_questions"],
    constraints=["confidence must be between 0.0 and 1.0"],
)

score = contract.validate(response_text)       # 0.0 – 1.0
metric = contract.as_metric(weight=0.10)       # use inside CompositeMetric
criteria = contract.as_judge_criteria()         # feed to LLM judge
```

#### CompositeMetric — Multi-Dimensional Scoring

Combine quality judges with **negative-weight** cost/latency penalties so the optimizer
balances accuracy against operational cost:

```python
from agentomatic.optimize import (
    CompositeMetric, WeightedMetric,
    LocalJudgeMetric, LatencyMetric, CostMetric,
)

metric = CompositeMetric(metrics=[
    WeightedMetric("completeness",   LocalJudgeMetric("completeness"),      weight=0.30),
    WeightedMetric("relevance",      LocalJudgeMetric("business_relevance"),weight=0.25),
    WeightedMetric("risk_detection", LocalJudgeMetric("risk_detection"),    weight=0.20),
    WeightedMetric("format",         contract.as_metric(),                  weight=0.10),
    WeightedMetric("latency",        LatencyMetric(),                       weight=-0.10),
    WeightedMetric("cost",           CostMetric(),                          weight=-0.05),
])
```

> Negative weights penalize candidates that are slower or more expensive, steering the
> fitter toward cost-effective configurations.

#### PromptSearchSpace — Full Configuration Surface

Tell the fitter *what* it's allowed to change:

```python
from agentomatic.optimize import PromptSearchSpace

space = PromptSearchSpace(
    optimize_system_prompt=True,
    optimize_few_shot=True,
    optimize_model_params=True,
    optimize_model_choice=True,
    model_choices=["ollama/qwen2.5:7b", "openai/gpt-4.1"],
    fallback_models=["openai/gpt-4.1-mini"],
    model_param_space={
        "temperature": [0.0, 0.1, 0.2, 0.4, 0.7],
        "top_p": [0.7, 0.9, 1.0],
    },
    rag_param_space={"top_k": [3, 5, 8, 12], "rerank": [True, False]},
    optimize_rag_params=True,
)
```

#### PromptFitter — The scikit-learn-like API

```python
from agentomatic.optimize import PromptFitter

fitter = PromptFitter(
    agent="scope_agent",
    task_model="ollama/qwen2.5:7b",
    rewrite_model="openai/gpt-4.1",
    optimizer="gepa_like",
    search_space=space,
    max_trials=30,
    min_absolute_improvement=0.05,
    concurrency=5,
)
result = await fitter.fit(trainset, valset, metric, testset=testset)
```

Access the full result surface:

```python
result.best_prompt              # optimized system prompt
result.best_params              # {"temperature": 0.2, "top_p": 0.9}
result.best_few_shot_examples   # selected few-shot examples
result.metric_deltas            # per-dimension improvement
result.suggestions              # actionable recommendations
result.deployment_recommendation # canary rollout config
result.summary()                # human-readable summary
result.apply(version="v2_optimized")
```

**Five optimisation strategies:**

| Strategy | What it does |
|---|---|
| `rewrite` | LLM-driven prompt rewrite based on failure analysis |
| `few_shot_bootstrap` | Score²-weighted example selection with diversity scoring |
| `mipro_like` | Multi-perspective instruction generation + cross-product search |
| `gepa_like` | Feedback-guided targeted prompt mutations |
| `param_search` | Grid search over model/RAG/tool parameters |

#### DeploymentRecommendation — Ship With Confidence

Every `PromptFitResult` includes a deployment recommendation based on the observed
improvement magnitude and variance:

```python
rec = result.deployment_recommendation
print(rec.confidence)              # "high" / "medium" / "low"
print(rec.rollout.strategy)        # "canary"
print(rec.rollout.initial_weight)  # 0.40
print(rec.summary())               # human-readable deployment plan
```

#### Failure Clusters — Targeted Diagnostics

The fitter groups validation failures into actionable clusters, each with the parameters
most likely to resolve the issue and the expected metric gain:

```
Failure cluster 1:
  Agent answered without using retrieval context.
  → Suggested fix: force context-first behavior.
  → Affected params: rag.top_k, tool_policy.force_retrieval
  → Expected metric gain: faithfulness +0.18

Failure cluster 2:
  Agent produced unstructured answers.
  → Suggested fix: stronger output format block.
  → Affected params: prompt.output_contract
  → Expected metric gain: format_compliance +0.12
```

#### Ideal CLI Flow

```bash
# 1. Run your agents
agentomatic run

# 2. Generate a synthetic evaluation dataset from your docs
agentomatic dataset synth scope_agent --from-docs docs/scoping.md --n 100

# 3. Evaluate the current version
agentomatic eval scope_agent --dataset scope_eval.jsonl --metrics scoping_quality

# 4. Fit a better configuration
agentomatic optimize scope_agent --optimize prompt,params,rag,tools

# 5. Canary release — send 20 % traffic to the new version
agentomatic route scope_agent --version v2_optimized --weight 20

# 6. Promote when satisfied
agentomatic promote scope_agent --version v2_optimized
```

#### Vocabulary

| ❌ Avoid | ✅ Use instead |
|----------|----------------|
| Program | Agent endpoint |
| Compile | Fit / optimize / tune |
| Signature | EvalContract |
| Module | Deployment component |
| Predictor | Agent version |
| Compiled artifact | Optimized config version |

## 🛠️ Development

```bash
# Install
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic
make dev  # Installs all deps + pre-commit hooks

# Quality
make lint          # Ruff linter
make format        # Auto-format
make typecheck     # Mypy
make test          # All tests
make test-cov      # With coverage
make check-all     # lint + typecheck + test

# Docs
make docs-serve    # Local docs server
make docs-build    # Build static site

# Build
make build         # Package
make publish       # PyPI
```

## 📜 License

MIT — see [LICENSE](LICENSE).

## 👥 Authors

**[UnicoLab](https://github.com/UnicoLab)** — Building the future of AI agent platforms.

---

<div align="center">

**[⭐ Star us on GitHub](https://github.com/UnicoLab/agentomatic)** — it helps!

Made with ❤️ by UnicoLab

</div>
