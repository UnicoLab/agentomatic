# Quick Start

Get from zero to a running agent API with visual debugging in under 60 seconds.

---

## 1. Install Agentomatic

=== "Full Install (Recommended)"

    Installs all features including Studio, Chat UI, database support, and optimization:

    ```bash
    pip install agentomatic[all]
    ```

=== "Minimal + Studio"

    Install only the core platform and visual debugger:

    ```bash
    pip install "agentomatic[studio,cli]"
    ```

=== "With uv (fast)"

    ```bash
    uv add agentomatic --extra all
    ```

??? info "Available Installation Extras"

    Agentomatic is modular — install only what you need:

    | Extra | What It Includes | When You Need It |
    |-------|-----------------|------------------|
    | `all` | Everything below | Production / full development |
    | `cli` | Rich terminal output, questionary prompts | Better CLI experience |
    | `ui` | Chainlit chat interface | Conversational testing at `/chat` |
    | `studio` | React-based visual debugger | Graph debugging at `/studio/ui/` |
    | `db` | SQLAlchemy + database drivers | PostgreSQL / SQLite thread persistence |
    | `optimize` | DeepEval, DSPy metrics | Automatic prompt tuning |
    | `metrics` | Prometheus client | `/metrics` endpoint for monitoring |
    | `telemetry` | OpenTelemetry SDK | Distributed tracing |

!!! tip "Verify your installation"
    ```bash
    agentomatic doctor
    ```
    This checks your Python version, installed packages, optional extras, and external service connections.

---

## 2. Scaffold Your First Agent

Generate a fully functional chatbot agent using the CLI:

```bash
agentomatic init my_chatbot --template basic
```

This creates a self-contained agent package under `agents/`:

```text
agents/my_chatbot/
├── __init__.py      # Manifest declaration + execution entrypoint
├── graph.py         # LangGraph StateGraph pipeline definition
├── nodes.py         # Node logic functions (LLM calls, processing)
├── config.py        # Pydantic configuration (model, temperature, etc.)
├── prompts.json     # Versioned system and user prompt templates
├── langgraph.json   # LangGraph Studio local settings
├── .env.example     # Environment variables blueprint
└── README.md        # Agent documentation
```

!!! note "Available Templates"
    | Template | Description |
    |----------|-------------|
    | `basic` | Simple single-node agent with LLM call |
    | `chatbot` | Multi-turn conversational bot with history |
    | `rag` | Retrieval-Augmented Generation with vector store |
    | `full` | All features: tools, RAG, config, custom schemas |
    | `custom` | Minimal scaffold for non-LangGraph agents |

    Select interactively by omitting the `--template` flag:
    ```bash
    agentomatic init my_agent  # Shows interactive picker
    ```

---

## 3. Run the Platform

Choose your preferred development mode:

=== "With Studio (Recommended)"

    Launch the platform with the visual debugging studio:

    ```bash
    agentomatic run --studio --reload
    ```

    | Service | URL |
    |---------|-----|
    | FastAPI Application | `http://localhost:8000` |
    | OpenAPI Swagger Docs | `http://localhost:8000/docs` |
    | **Agentomatic Studio** | `http://localhost:8000/studio/ui/` |

    !!! tip "Studio is the primary debug tool"
        Studio provides graph visualization, SSE node streaming, time-travel debugging, state inspection, and live state editing. See the [Studio Guide](../guide/studio.md) for details.

=== "With Chat UI"

    Launch with the Chainlit conversational interface:

    ```bash
    agentomatic run --with-ui --reload
    ```

    | Service | URL |
    |---------|-----|
    | FastAPI Application | `http://localhost:8000` |
    | OpenAPI Swagger Docs | `http://localhost:8000/docs` |
    | **Chainlit Chat UI** | `http://localhost:8000/chat` |

=== "With Both"

    Run Studio and Chat UI together:

    ```bash
    agentomatic run --studio --with-ui --reload
    ```

=== "Python Entry Point"

    Create a `main.py` file for programmatic control:

    ```python
    # main.py
    from agentomatic import AgentPlatform

    platform = AgentPlatform.from_folder(
        "agents/",
        title="My Agent Platform",
        enable_studio=True,  # (1)!
    )
    app = platform.build()
    ```

    1. Enables Agentomatic Studio at `/studio/ui/`

    ```bash
    uvicorn main:app --reload
    ```

---

## 4. Query Your Agent

Interact with the running agent using any client:

=== "curl"

    ```bash
    curl -X POST http://localhost:8000/api/v1/my_chatbot/invoke \
      -H "Content-Type: application/json" \
      -d '{"query": "Hello! What can you do?"}'
    ```

=== "Python"

    ```python
    import httpx

    response = httpx.post(
        "http://localhost:8000/api/v1/my_chatbot/invoke",
        json={"query": "Hello! What can you do?"},
    )
    print(response.json())
    ```

=== "SSE Streaming"

    ```python
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
       Steps: greeting_node
       ⏱ 114ms
    ```

=== "Multi-Turn Chat"

    ```bash
    curl -X POST http://localhost:8000/api/v1/my_chatbot/chat \
      -H "Content-Type: application/json" \
      -d '{
        "query": "What did I just ask?",
        "thread_id": "thread_abc123",
        "user_id": "user-1"
      }'
    ```

### Expected JSON Response

```json
{
  "response": "Hello! How can I assist you today?",
  "agent_type": "agent-my_chatbot",
  "thread_id": "thread_abc123",
  "suggestions": ["Introduce yourself", "What can you do?"],
  "citations": [],
  "steps_taken": ["greeting_node"],
  "metadata": {},
  "duration_ms": 114.2
}
```

---

## 5. Configuration Options

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

```python
# agents/my_chatbot/config.py
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    prompt_version: str = Field("v1", description="Active prompt template version")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048)
    llm_model: str = Field("gpt-4o-mini")
```

!!! info "Runtime Access"
    Configuration is accessible via the API at `GET /api/v1/my_chatbot/config` and can be modified via `POST /api/v1/my_chatbot/config`.

---

## :material-compass: What's Next?

Now that your first agent is running, explore these resources:

| Topic | Description |
|-------|-------------|
| **[Agent Structure](../guide/agent-structure.md)** | Deep dive into folder conventions, manifest fields, and override patterns |
| **[Agentomatic Studio](../guide/studio.md)** | Visual debugging with graph view, state inspection, and time-travel |
| **[Chat Interface](../guide/debug-ui.md)** | Chainlit-based conversational testing |
| **[Prompt Management](../guide/prompts.md)** | Template versioning, hot-reload, and A/B testing |
| **[Prompt Optimization](../guide/optimization.md)** | Auto-tune prompts with DSPy-inspired optimization |
| **[Storage Backends](../guide/storage.md)** | Configure PostgreSQL, SQLite, or custom adapters |
| **[CLI Reference](../cli/commands.md)** | Every command and flag documented |
| **[Architecture](../architecture/overview.md)** | Platform internals, request flow, and design decisions |
