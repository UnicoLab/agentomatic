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
| 🔍 **Zero-Code Auto-Discovery** | Drop an agent folder → 25+ fully-documented REST endpoints appear automatically. |
| 🚀 **Rich API Surface** | Natively handles `invoke`, `stream`, `chat`, `A2A`, `health`, `config`, `threads`, `memory`, and `feedback`. |
| 🗄️ **Pluggable Storage** | Use `MemoryStore`, `SQLAlchemy`, or plug in your own custom persistence layer. |
| 🔐 **Enterprise Middleware** | High-performance pipeline with JWT Auth, dynamic rate limiting, and Prometheus telemetry — all toggleable. |
| 📦 **Scaffolding Templates** | Jumpstart development with 9 templates: `basic`, `full`, `rag`, `chatbot`, `deepagent`, `custom`, `swarm`, `pipeline`, `class`. |
| 🧬 **Class-Based Agents** | Define agents as Python classes with ML lifecycle: `compile()` → `fit()` → `evaluate()` → `transform()`. |
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

| Template | Files | Description |
|----------|-------|-------------|
| `basic` | 7 | Minimal agent — quick start |
| `full` | 11 | All override files — config, schemas, api, tools |
| `rag` | 9 | Retrieve → Generate pipeline |
| `chatbot` | 8 | Conversational with memory |
| `deepagent` | 6 | Autonomous planning with sub-agents |
| `custom` | 4 | Framework-agnostic — no LangGraph |
| `swarm` | 10+ | Multi-agent delegation and handoffs |
| `pipeline` | 2 | Multi-agent workflow composition (YAML) |
| `class` | 4 | **NEW** Python class with ML lifecycle |

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

Define agents as Python classes with LangGraph-style graph wiring and ML lifecycle:

```python
from dataclasses import dataclass, field
from agentomatic import BaseGraphAgent

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

# ML-like workflow
agent = MyAgent()
result = agent.transform({"query": "Hello!"})
agent.compile(dataset, metrics)
agent.fit(dataset)
report = agent.evaluate(dataset.test, metrics)
agent.save("compiled/v1")
```

## 🎨 Agentomatic Studio

Agentomatic ships with a built-in React-based visual studio designed for time-travel debugging, real-time node streaming, and state inspection for all underlying LangGraph agents.

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
