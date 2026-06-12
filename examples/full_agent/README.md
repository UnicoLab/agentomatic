# Full Agent Example

This example demonstrates **every customization option** available in Agentomatic.

## Agent Folder Structure

```
agents/weather/
├── __init__.py      ← REQUIRED: AgentManifest + node_fn
├── graph.py         ← Optional: LangGraph StateGraph with branching
├── nodes.py         ← Optional: Node functions
├── config.py        ← Optional: Agent-specific configuration
├── schemas.py       ← Optional: Custom Pydantic models
├── tools.py         ← Optional: LangChain-compatible tools
├── api.py           ← Optional: Custom FastAPI router (overrides auto-gen)
├── prompts.json     ← Optional: Versioned prompt templates
└── langgraph.json   ← Optional: LangGraph Studio integration
```

## File Descriptions

| File | Required | Purpose |
|------|----------|--------|
| `__init__.py` | ✅ Yes | Exports `manifest` and `node_fn` |
| `graph.py` | No | Defines the LangGraph `StateGraph` |
| `nodes.py` | No | Individual node functions |
| `config.py` | No | `{Name}Config` class for agent settings |
| `schemas.py` | No | Custom request/response Pydantic models |
| `tools.py` | No | LangChain `@tool` decorated functions |
| `api.py` | No | Custom `APIRouter` (replaces auto-generated) |
| `prompts.json` | No | Versioned prompts with template variables |
| `langgraph.json` | No | LangGraph Studio/CLI configuration |

## Running

```bash
pip install agentomatic[langgraph]
cd examples/full_agent
uvicorn main:app --reload --port 8001
```

## Testing Endpoints

```bash
# List agents
curl http://localhost:8001/api/v1/agents

# Custom endpoint (from api.py)
curl http://localhost:8001/api/v1/weather/locations

# Custom typed endpoint
curl -X POST http://localhost:8001/api/v1/weather/forecast \
  -H 'Content-Type: application/json' \
  -d '{"location": "Paris", "days": 3}'

# Admin stats (global custom router)
curl http://localhost:8001/admin/stats

# Health check
curl http://localhost:8001/health

# A2A discovery
curl http://localhost:8001/.well-known/agent.json
```
