# Quick Start

Get a multi-agent API running in under 60 seconds.

---

## 1. Install

```bash
pip install agentomatic[all]
```

## 2. Create an Agent

```bash
agentomatic init hello_world --template basic
```

This scaffolds a ready-to-run agent:

```
agents/hello_world/
├── __init__.py      # Agent manifest + entry point
├── graph.py         # LangGraph state graph
├── nodes.py         # Processing functions
├── prompts.json     # Prompt templates
├── langgraph.json   # LangGraph Studio config
├── .env.example     # Environment variables
└── README.md        # Agent docs
```

## 3. Run

=== "CLI (recommended)"
    ```bash
    agentomatic run
    # → Platform running at http://localhost:8000
    # → Swagger UI at http://localhost:8000/docs
    ```

=== "Python"
    ```python
    # main.py
    from agentomatic import AgentPlatform

    platform = AgentPlatform.from_folder("agents/")
    app = platform.build()
    ```
    ```bash
    uvicorn main:app --reload
    ```

---

## 4. Test Your Agent

=== "curl"
    ```bash
    curl -X POST http://localhost:8000/api/v1/hello_world/invoke \
      -H "Content-Type: application/json" \
      -d '{"query": "Hello!"}'
    ```

=== "Python"
    ```python
    import httpx

    resp = httpx.post(
        "http://localhost:8000/api/v1/hello_world/invoke",
        json={"query": "Hello!"},
    )
    print(resp.json())
    ```

=== "CLI"
    ```bash
    agentomatic test hello_world
    ```

Expected response:

```json
{
  "response": "Hello! How can I help you today?",
  "agent_type": "agent-hello_world",
  "thread_id": "t-abc123",
  "suggestions": ["Tell me more", "What can you do?"],
  "duration_ms": 142.5
}
```

---

## Auto-Generated Endpoints

Every agent gets these endpoints automatically — zero configuration:

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/{agent}/invoke` | Synchronous invocation |
| `POST` | `/api/v1/{agent}/invoke/stream` | SSE streaming |
| `POST` | `/api/v1/{agent}/chat` | Session-aware chat |
| `GET` | `/api/v1/{agent}/health` | Per-agent health |
| `GET` | `/api/v1/{agent}/config` | Agent configuration |
| `GET` | `/api/v1/{agent}/prompts` | Prompt versions |
| `GET` | `/api/v1/{agent}/card` | A2A agent card |
| `POST` | `/api/v1/{agent}/a2a/tasks` | A2A task submission |
| `GET` | `/api/v1/{agent}/threads` | List threads |
| `GET` | `/api/v1/{agent}/threads/{id}` | Get thread |
| `GET` | `/api/v1/{agent}/threads/{id}/messages` | Thread messages |
| `POST` | `/api/v1/{agent}/feedback` | Submit feedback |
| `GET` | `/api/v1/{agent}/feedback` | List feedback |

!!! tip "Interactive API docs"
    Visit `http://localhost:8000/docs` for a full Swagger UI where you can test all endpoints interactively.

---

## What's Next?

- **[Your First Agent](first-agent.md)** — Build a production-ready agent from scratch
- **[Templates](../guide/templates.md)** — Explore all 5 scaffolding templates
- **[Prompt Optimization](../guide/optimization.md)** — Auto-tune your prompts with DSPy-inspired strategies
- **[CLI Reference](../cli/commands.md)** — Full command documentation
