# Quick Start

Get from zero to a running multi-agent API with visual debugging in under 60 seconds.

---

## Prerequisites

Before you begin, make sure you have the following:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| **Python** | 3.11+ | `python --version` |
| **pip** or **uv** | Latest | `pip --version` / `uv --version` |
| **LLM API key** | — | `echo $OPENAI_API_KEY` or local Ollama |

!!! tip "Local LLMs with Ollama"
    You don't need an OpenAI API key to get started. Agentomatic works with [Ollama](https://ollama.com) out of the box:
    ```bash
    # Install Ollama, then pull a model
    ollama pull mistral:7b
    ```

---

## 1. Install Agentomatic

=== "pip (Recommended)"

    Install all features including Studio, Chat UI, database support, and optimization:

    ```bash
    pip install agentomatic[all]
    ```

=== "uv (Fast)"

    ```bash
    uv add agentomatic --extra all
    ```

=== "poetry"

    ```bash
    poetry add agentomatic -E all
    ```

=== "Minimal + Studio"

    Install only the core platform and visual debugger:

    ```bash
    pip install "agentomatic[studio,cli]"
    ```

??? info "Available Installation Extras"

    Agentomatic is modular — install only what you need:

    | Extra | What It Includes | When You Need It |
    |-------|-----------------|------------------|
    | `all` | Everything below | Production / full development |
    | `langgraph` | LangGraph + LangChain Core | LangGraph-based agents |
    | `langchain` | LangChain + LangChain Community | LangChain LCEL agents |
    | `ollama` | LangChain-Ollama bindings | Local LLM development with Ollama |
    | `openai` | LangChain-OpenAI bindings | OpenAI API-based agents |
    | `azure` | LangChain-OpenAI (Azure) | Azure OpenAI deployments |
    | `vertex` | LangChain-Google VertexAI | Google Cloud Vertex AI |
    | `cli` | Rich terminal output, questionary prompts | Better CLI experience |
    | `ui` | Chainlit chat interface | Conversational testing at `/chat` |
    | `studio` | React-based visual debugger | Graph debugging at `/studio/ui/` |
    | `db` | SQLAlchemy + aiosqlite | SQLite thread persistence |
    | `db-postgres` | SQLAlchemy + asyncpg + psycopg | PostgreSQL thread persistence |
    | `optimize` | DeepEval, HolySheet | Automatic prompt tuning |
    | `metrics` | Prometheus client | `/metrics` endpoint for monitoring |
    | `telemetry` | OpenTelemetry SDK | Distributed tracing |

!!! success "Verify your installation"
    ```bash
    agentomatic doctor
    ```
    This checks your Python version, installed packages, optional extras, and external service connections.

---

## 2. Scaffold Your First Agent

Generate a fully functional agent using the CLI:

```bash
agentomatic init my_chatbot --template basic
```

This creates a self-contained agent package under `agents/`:

```text
agents/my_chatbot/
├── __init__.py      # AgentManifest card
├── agent.py         # REQUIRED: BaseGraphAgent subclass
├── llm.py           # Stack-aware LLM helpers
├── prompts.json     # Versioned system and user prompt templates
├── langgraph.json   # LangGraph Studio local settings
├── .env.example     # Environment variables blueprint
└── README.md        # Agent documentation
```

!!! note "Available Templates"
    | Template | Description |
    |----------|-------------|
    | `basic` / `class` | Minimal class-based agent (recommended) — quick start |
    | `chatbot` | Conversational class-based agent with memory |
    | `rag` | RAG class-based agent — retrieve → generate pipeline |
    | `full` | All features: class agent with config, schemas, train/eval |
    | `coordinator` | Orchestrator that routes to specialists |
    | `pipeline` | Multi-step YAML workflow |
    | `deepagent` | Deep Agent with planning, tools, subagents |
    | `custom` | Framework-agnostic — no LangGraph dependency |
    | `legacy_dict` | Legacy functional agent — `manifest` + `node_fn` |
    | `plugin` | ML Model Plugin with REST endpoints |
    | `endpoint` / `connection` / `ingestion` / `extraction` | Ops / RAG building blocks |

    Select interactively by omitting the `--template` flag:
    ```bash
    agentomatic init my_agent  # Shows interactive picker
    ```

---

## 3. Choose Your Framework

Agentomatic supports multiple agent frameworks. Here's a quickstart for each:

=== "Class-Based Agent (Recommended)"

    The default and most feature-rich option. Subclass `BaseGraphAgent`:

    ```python title="agents/my_chatbot/agent.py"
    from dataclasses import dataclass, field
    from typing import Any
    from agentomatic.agents import BaseGraphAgent

    @dataclass
    class ChatbotState:
        request: str = ""
        output: dict[str, Any] = field(default_factory=dict)

    class ChatbotAgent(BaseGraphAgent[ChatbotState]):
        agent_name = "my_chatbot"
        agent_description = "A conversational chatbot."
        agent_framework = "graph_agent"

        def build_graph(self):
            g = self.new_graph()
            g.add_node("process", self.process)
            g.set_entry_point("process")
            g.set_finish_point("process")
            return g.compile()

        def process(self, state: ChatbotState) -> ChatbotState:
            state.output = {
                "response": f"Hello! You said: {state.request}",
                "agent_type": "agent-my_chatbot",
            }
            return state

        def input_to_state(self, input_data: dict[str, Any]) -> ChatbotState:
            return ChatbotState(request=input_data.get("current_query", ""))

        def state_to_output(self, state: ChatbotState) -> dict[str, Any]:
            return state.output
    ```


=== "LangChain LCEL Agent"

    Use LangChain Expression Language (LCEL) chains. Expose your logic via `node_fn`:

    ```python title="agents/lcel_agent/__init__.py"
    from agentomatic import AgentManifest
    from typing import Any

    manifest = AgentManifest(
        name="lcel_agent",
        slug="lcel-agent",
        description="A LangChain LCEL-based summarizer agent.",
        intent_keywords=["summarize", "tldr", "summary"],
        version="1.0.0",
        framework="langchain",  # (1)!
    )

    async def node_fn(state: dict[str, Any]) -> dict[str, Any]:  # (2)!
        """Execute the LCEL chain directly."""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a concise summarizer. Summarize the input in 2-3 sentences."),
            ("human", "{query}"),
        ])
        llm = ChatOllama(model="mistral:7b", temperature=0.3)
        chain = prompt | llm | StrOutputParser()  # (3)!

        query = state.get("current_query", "")
        result = await chain.ainvoke({"query": query})

        return {
            "response": result,
            "agent_type": "agent-lcel_agent",
            "steps_taken": ["lcel_chain"],
        }
    ```

    1. Framework hint for adapters — Studio will show a linear chain view
    2. `node_fn` is the single-function alternative to `graph_fn` — simpler but no graph visualization
    3. LCEL pipe syntax: prompt → LLM → output parser

=== "Deep Agent"

    Use the Deep Agent framework for hierarchical, goal-oriented agents:

    ```python title="agents/deep_planner/__init__.py"
    from agentomatic import AgentManifest
    from .agent import build_deep_agent

    manifest = AgentManifest(
        name="deep_planner",
        slug="deep-planner",
        description="A planning agent using Deep Agent framework.",
        intent_keywords=["plan", "organize", "schedule"],
        version="1.0.0",
        framework="langgraph",  # (1)!
    )

    def graph_fn():  # (2)!
        """Return the compiled Deep Agent as a LangGraph-compatible graph."""
        return build_deep_agent()
    ```

    1. Deep Agents compile to LangGraph graphs, so use `framework="langgraph"` for full Studio support
    2. The `graph_fn` returns the compiled graph — Studio can inspect nodes, edges, and state

    ```python title="agents/deep_planner/agent.py"
    from deepagents import create_deep_agent

    def build_deep_agent():
        """Build and return a Deep Agent workflow."""
        agent = create_deep_agent(
            name="planner",
            model="ollama/mistral:7b",
            goal="Break down user requests into actionable steps",
            tools=["web_search", "calculator"],
        )
        return agent.compile()  # (3)!
    ```

    3. `.compile()` returns a LangGraph `CompiledStateGraph` — compatible with all Agentomatic features

    !!! info "Learn more about Deep Agent integration"
        See the [Deep Agent Integration Guide](../guide/deep-agents.md) for advanced configuration, custom tools, and multi-agent hierarchies.

=== "Custom Python Agent"

    No framework needed — use plain Python with `node_fn`:

    ```python title="agents/simple_bot/__init__.py"
    from agentomatic import AgentManifest
    from typing import Any

    manifest = AgentManifest(
        name="simple_bot",
        slug="simple-bot",
        description="A zero-dependency echo agent in pure Python.",
        intent_keywords=["echo", "test", "ping"],
        version="1.0.0",
        framework="custom",  # (1)!
    )

    async def node_fn(state: dict[str, Any]) -> dict[str, Any]:  # (2)!
        """Pure Python agent — no LLM, no framework."""
        query = state.get("current_query", "")
        return {
            "response": f"You said: {query}",
            "agent_type": "agent-simple_bot",
            "steps_taken": ["echo"],
            "suggestions": ["Say something else", "Try another agent"],
        }
    ```

    1. `framework="custom"` tells Agentomatic there's no graph to inspect — Studio shows basic metadata only
    2. `node_fn` receives the full state dict and must return a dict with at least `"response"`

    !!! tip "When to use `node_fn` vs `graph_fn`"
        | Use `node_fn` when... | Use `graph_fn` when... |
        |---|---|
        | Your agent is a single function | Your agent has multiple steps/nodes |
        | No graph visualization needed | You want Studio graph debugging |
        | Using LCEL chains or plain Python | Using LangGraph `StateGraph` |
        | Simpler is better | Complex orchestration required |

---

## 4. Run the Platform

Choose your preferred development mode:

=== "With Studio (Recommended)"

    Launch the platform with the visual debugging studio:

    ```bash
    agentomatic run --studio --reload
    ```

    | Service | URL |
    |---------|-----|
    | :material-api: FastAPI Application | `http://localhost:8000` |
    | :material-file-document: OpenAPI Swagger Docs | `http://localhost:8000/docs` |
    | :material-palette: **Agentomatic Studio** | `http://localhost:8000/studio/ui/` |

    !!! tip "Studio is the primary debug tool"
        Studio provides graph visualization, SSE node streaming, time-travel debugging, state inspection, and live state editing. See the [Studio Guide](../guide/studio.md) for details.

=== "With Chat UI"

    Launch with the Chainlit conversational interface:

    ```bash
    agentomatic run --with-ui --reload
    ```

    | Service | URL |
    |---------|-----|
    | :material-api: FastAPI Application | `http://localhost:8000` |
    | :material-file-document: OpenAPI Swagger Docs | `http://localhost:8000/docs` |
    | :material-message-text: **Chainlit Chat UI** | `http://localhost:8000/chat` |

=== "With Both"

    Run Studio and Chat UI together:

    ```bash
    agentomatic run --studio --with-ui --reload
    ```

=== "Python Entry Point"

    Create a `main.py` file for programmatic control:

    ```python title="main.py"
    from agentomatic import AgentPlatform

    platform = AgentPlatform.from_folder(
        "agents/",
        title="My Agent Platform",
        enable_studio=True,   # (1)!
        enable_metrics=True,  # (2)!
    )
    app = platform.build()
    ```

    1. Enables Agentomatic Studio at `/studio/ui/`
    2. Enables Prometheus metrics at `/metrics`

    ```bash
    uvicorn main:app --reload
    ```

---

## 5. Query Your Agent

Interact with the running agent using any HTTP client:

=== "curl — Invoke"

    ```bash
    curl -X POST http://localhost:8000/api/v1/my_chatbot/invoke \
      -H "Content-Type: application/json" \
      -d '{"query": "Hello! What can you do?"}'
    ```

=== "curl — Stream (SSE)"

    ```bash
    curl -N -X POST http://localhost:8000/api/v1/my_chatbot/invoke/stream \
      -H "Content-Type: application/json" \
      -d '{"query": "Tell me a story about a robot"}'
    ```

    The `-N` flag disables output buffering so you see SSE events in real-time:
    ```text
    data: {"node": "greet", "output": {"agent_type": "agent-my_chatbot"}}
    data: {"node": "respond", "output": {"response": "Once upon a time..."}}
    data: [DONE]
    ```

=== "curl — Multi-Turn Chat"

    ```bash
    # First message — starts a conversation thread
    curl -X POST http://localhost:8000/api/v1/my_chatbot/chat \
      -H "Content-Type: application/json" \
      -d '{
        "query": "My name is Alice",
        "thread_id": "thread_abc123",
        "user_id": "user-1"
      }'

    # Follow-up — Agentomatic remembers the thread history
    curl -X POST http://localhost:8000/api/v1/my_chatbot/chat \
      -H "Content-Type: application/json" \
      -d '{
        "query": "What is my name?",
        "thread_id": "thread_abc123",
        "user_id": "user-1"
      }'
    ```

=== "Python — httpx"

    ```python title="test_invoke.py"
    import httpx

    response = httpx.post(
        "http://localhost:8000/api/v1/my_chatbot/invoke",
        json={"query": "Hello! What can you do?"},
    )
    print(response.json())
    ```

=== "Python — SSE Streaming"

    ```python title="test_stream.py"
    import httpx

    with httpx.stream(
        "POST",
        "http://localhost:8000/api/v1/my_chatbot/invoke/stream",
        json={"query": "Tell me a story"},
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                print(line[6:])
    ```

=== "Interactive CLI"

    ```bash
    # Opens a chat-like session in your terminal
    agentomatic test my_chatbot
    ```

    ```text
    ⚡ agentomatic
    🧪 Testing agent: my_chatbot
       API: http://localhost:8000/api/v1/my_chatbot/invoke
       Type 'quit' or 'exit' to stop

    🗣️  You: Hello!
    🤖 my_chatbot: Hello! How can I assist you today?
       Steps: greet → respond
       ⏱ 114ms
    ```

### Expected JSON Response

```json
{
  "response": "Hello! How can I assist you today?",
  "agent_type": "agent-my_chatbot",
  "thread_id": "thread_abc123",
  "suggestions": [],
  "citations": [],
  "steps_taken": ["greet", "respond"],
  "metadata": {},
  "duration_ms": 114.2
}
```

---

## 6. Auto-Generated Endpoints

Every registered agent gets a full suite of REST endpoints automatically:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/{agent}/invoke` | Synchronous invocation |
| `POST` | `/api/v1/{agent}/invoke/stream` | SSE streaming invocation |
| `POST` | `/api/v1/{agent}/chat` | Multi-turn conversation with thread persistence |
| `GET` | `/api/v1/{agent}/health` | Agent health check |
| `GET` | `/api/v1/{agent}/config` | Agent configuration |
| `GET` | `/api/v1/{agent}/prompts` | Available prompt versions |
| `GET` | `/api/v1/{agent}/card` | A2A agent card |
| `POST` | `/api/v1/{agent}/a2a/tasks` | Submit A2A task |
| `GET` | `/api/v1/{agent}/a2a/tasks/{id}` | Get A2A task status |
| `POST` | `/api/v1/{agent}/threads` | Create thread |
| `GET` | `/api/v1/{agent}/threads` | List conversation threads |
| `GET` | `/api/v1/{agent}/threads/{id}` | Get thread history |
| `PATCH` | `/api/v1/{agent}/threads/{id}` | Update thread |
| `DELETE` | `/api/v1/{agent}/threads/{id}` | Delete thread |
| `GET` | `/api/v1/{agent}/threads/{id}/messages` | Get messages |
| `DELETE` | `/api/v1/{agent}/threads/{id}/messages` | Delete messages |
| `GET` | `/api/v1/{agent}/threads/{id}/summary` | Get thread summary |
| `POST` | `/api/v1/{agent}/feedback` | Submit user feedback |
| `GET` | `/api/v1/{agent}/feedback` | Retrieve feedback entries |
| `GET` | `/api/v1/{agent}/feedback/export` | Export feedback data |
| `GET` | `/api/v1/{agent}/threads/{id}/pending` | Get pending HITL actions |
| `POST` | `/api/v1/{agent}/threads/{id}/approve` | HITL approval |
| `POST` | `/api/v1/{agent}/threads/{id}/reject` | HITL rejection |
| `POST` | `/api/v1/{agent}/threads/{id}/fork` | Fork thread |
| `GET` | `/api/v1/{agent}/threads/{id}/lineage` | Get thread lineage |
| `POST` | `/api/v1/{agent}/optimize/invoke` | Optimization invocation |

!!! info "Full endpoint documentation"
    Visit `http://localhost:8000/docs` for the interactive Swagger UI with all endpoints, schemas, and try-it-out functionality.

---

## 7. Configuration

### Environment Variables

Set these in your `.env` file or export them in your shell:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTOMATIC_AGENTS_DIR` | `agents` | Directory to scan for agent packages |
| `AGENTOMATIC_HOST` | `0.0.0.0` | Server bind address |
| `AGENTOMATIC_PORT` | `8000` | Server bind port |
| `AGENTOMATIC_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `AGENTOMATIC_DB_URL` | `sqlite:///data/threads.db` | Database URL for thread persistence |
| `OPENAI_API_KEY` | — | Required for OpenAI-based agents |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |

### Per-Agent Configuration

Each agent can define its own `config.py` with a Pydantic model:

```python title="agents/my_chatbot/config.py"
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    prompt_version: str = Field("v1", description="Active prompt template version")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048)
    llm_model: str = Field("gpt-4o-mini")
```

!!! info "Runtime Access"
    Configuration is accessible via the API at `GET /api/v1/my_chatbot/config` and can be modified at runtime via `POST /api/v1/my_chatbot/config`.

---

## :material-help-circle: Understanding the State Dictionary

When Agentomatic receives a REST request and invokes your agent, it maps the request body fields into a **state dictionary**. Understanding this mapping is crucial:

| REST Request Field | State Dict Key | Description |
|--------------------|----------------|-------------|
| `query` | `current_query` | The user's input text |
| `user_id` | `user_id` | User identifier (default: `"default-user"`) |
| `thread_id` | `thread_id` | Conversation thread (for multi-turn chat) |
| `context` | `context` | Arbitrary context dict |
| `metadata` | `metadata` | Request metadata dict |
| `prompt_version` | `prompt_version` | Active prompt version (default: `"v1"`) |

!!! tip "Class-based agents vs functional agents"
    - **Class-based agents**: You control mapping via `input_to_state()` — use `input_data.get("current_query")` or any key you prefer
    - **Functional agents** (`node_fn`): The state dict is passed directly with keys above — use `state.get("current_query")`

---

## :material-bug: Troubleshooting

??? question "My agent isn't being discovered"
    1. **Check the folder**: Your agent must be in the `agents/` directory (or wherever `--agents-dir` points)
    2. **Check for errors**: Look at the console output for `❌ Failed to discover` messages
    3. **Class agents**: Need `agent.py` with a `BaseGraphAgent` subclass
    4. **Functional agents**: Need `__init__.py` with a `manifest` variable

??? question "`ModuleNotFoundError` when starting"
    This usually means a dependency is missing. Common fixes:
    ```bash
    pip install agentomatic[langgraph]  # For LangGraph-based agents
    pip install agentomatic[ollama]     # For Ollama LLM provider
    pip install agentomatic[all]        # Install everything
    ```

??? question "Agent returns empty response"
    - **Class agents**: Make sure `state_to_output()` returns a dict with a `"response"` key
    - **Functional agents**: Make sure `node_fn` returns a dict with a `"response"` key
    - Check that your node methods actually **return** the updated state (`return state`)

??? question "Port already in use"
    ```bash
    # Kill the existing process
    lsof -ti :8000 | xargs kill -9
    # Or use a different port
    agentomatic run --port 8001
    ```

??? question "CORS errors from frontend"
    Pass your frontend origin to the platform configuration:
    ```python
    platform = AgentPlatform.from_folder(
        "agents/",
        cors_origins=["http://localhost:3000"],
    )
    ```

---

## :material-compass: What's Next?

Now that your first agent is running, explore these resources:

!!! tip "Recommended Learning Path"
    **First Agent** → **Class-Based Agents** → **Cookbook** → **Configuration** → **Studio**

| Topic | Description |
|-------|-------------|
| **[Your First Agent](first-agent.md)** | Step-by-step tutorial building an agent from scratch with annotated code |
| **[Class-Based Agents](../guide/class-agents.md)** | Complete guide to `BaseGraphAgent`, graph wiring, ML lifecycle |
| **[Cookbook & Recipes](../guide/cookbook.md)** | 10 copy-paste patterns: RAG, routing, HITL, schemas, ML plugins |
| **[Agent Structure](../guide/agent-structure.md)** | Folder conventions, manifest fields, and auto-discovery |
| **[Agentomatic Studio](../guide/studio.md)** | Visual debugging with graph view, state inspection, and time-travel |
| **[Deep Agent Integration](../guide/deep-agents.md)** | Register and debug Deep Agent workflows with full Studio support |
| **[Chat Interface](../guide/debug-ui.md)** | Chainlit-based conversational testing |
| **[Prompt Management](../guide/prompts.md)** | Template versioning, hot-reload, and A/B testing |
| **[Prompt Optimization](../guide/optimization.md)** | Auto-tune prompts with DSPy-inspired optimization |
| **[Storage Backends](../guide/storage.md)** | Configure PostgreSQL, SQLite, or custom adapters |
| **[Middleware](../guide/middleware.md)** | Auth, rate limiting, metrics, and custom middleware |
| **[Configuration](../guide/configuration.md)** | Platform settings, CORS, environment variables |
| **[CLI Reference](../cli/commands.md)** | Every command and flag documented |
| **[API Reference](../architecture/api-reference.md)** | All 26 endpoints with request/response schemas |
