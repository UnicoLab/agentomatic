# Quick Start

## 1. Install

```bash
pip install agentomatic[all]
```

## 2. Create an Agent

```bash
agentomatic init hello_world --template basic
```

This creates:

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

## 3. Create `main.py`

```python
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/")
app = platform.build()
```

## 4. Run

```bash
uvicorn main:app --reload
```

## 5. Test

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

## Auto-Generated Endpoints

Every agent gets these endpoints automatically:

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
