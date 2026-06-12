# Agent Structure

Every agentomatic agent is a Python package (folder with `__init__.py`).

## Required Files

Only `__init__.py` is required. It must export:

- `manifest` — an `AgentManifest` instance
- `node_fn` — an async function, OR
- A `graph.py` module with `get_graph()`

```python
# agents/my_agent/__init__.py
from agentomatic import AgentManifest

manifest = AgentManifest(
    name="my_agent",
    slug="my-agent",
    description="Does something useful",
    intent_keywords=["help", "assist"],
    version="1.0.0",
    framework="langgraph",  # or "langchain", "custom"
)

async def node_fn(state: dict) -> dict:
    return {"response": "Hello!"}
```

## Optional Override Files

| File | Purpose | When to Use |
|---|---|---|
| `graph.py` | LangGraph StateGraph definition | Multi-step agents |
| `nodes.py` | Node processing functions | Complex logic |
| `config.py` | Agent-specific Pydantic config | Custom settings |
| `schemas.py` | Custom request/response models | Custom API contracts |
| `tools.py` | LangChain tools | Tool-using agents |
| `api.py` | Custom FastAPI router | Full endpoint control |
| `prompts.json` | Versioned prompt templates | A/B testing prompts |
| `langgraph.json` | LangGraph Studio config | Visual debugging |

!!! note "Override Priority"
    If `api.py` exports a `router`, it **replaces** the auto-generated endpoints entirely.
    Remove `api.py` to use auto-generated endpoints.

## Manifest Fields

```python
AgentManifest(
    name="my_agent",           # Unique name (used in URL path)
    slug="my-agent",           # Human-readable identifier
    description="...",         # Shown in docs and A2A cards
    intent_keywords=["..."],   # For intent routing
    version="1.0.0",           # Semantic version
    framework="langgraph",     # langgraph | langchain | custom
    is_subagent=True,          # False to hide from API
    metadata={},               # Custom metadata
)
```
