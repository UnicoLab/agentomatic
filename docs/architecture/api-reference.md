# API Reference

## Platform Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Platform info |
| `GET` | `/health` | Aggregated health |
| `GET` | `/readiness` | Kubernetes readiness probe |
| `GET` | `/docs` | OpenAPI docs |
| `GET` | `/.well-known/agent.json` | A2A discovery |
| `GET` | `/api/v1/agents` | List all agents |
| `GET` | `/api/v1/storage/stats` | Storage statistics |
| `POST` | `/api/v1/feedback` | Submit feedback |
| `GET` | `/api/v1/feedback` | List feedback |
| `GET` | `/metrics` | Prometheus metrics |

## Per-Agent Endpoints

All prefixed with `/api/v1/{agent_name}/`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/invoke` | Sync invocation |
| `POST` | `/invoke/stream` | SSE streaming |
| `POST` | `/chat` | Session-aware chat |
| `GET` | `/health` | Agent health |
| `GET` | `/config` | Agent config |
| `GET` | `/prompts` | Prompt versions |
| `GET` | `/card` | A2A agent card |
| `POST` | `/a2a/tasks` | A2A task submission |
| `GET` | `/a2a/tasks/{id}` | A2A task status |
| `GET` | `/threads` | List threads |
| `GET` | `/threads/{id}` | Get thread |
| `GET` | `/threads/{id}/messages` | Thread messages |

## Request/Response Models

### Invoke Request

```json
{
    "query": "Hello!",
    "user_id": "user-123",
    "thread_id": "thread-abc",
    "prompt_version": "v1",
    "temperature": 0.7,
    "max_tokens": 1024,
    "context": {},
    "metadata": {}
}
```

### Invoke Response

```json
{
    "response": "Agent response text",
    "agent_type": "agent-name",
    "thread_id": "thread-abc",
    "suggestions": ["Follow-up 1"],
    "citations": [{"source": "..."}],
    "steps_taken": ["step1", "step2"],
    "metadata": {},
    "duration_ms": 142.5
}
```
