# Agent Structure & Overrides

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Directory Conventions and Customization Reference</h3>
</div>

---

Agentomatic follows a **convention-over-configuration** model. Each agent is a self-contained Python package (a folder containing `__init__.py`) stored under your configured agents directory (e.g. `agents/my_agent/`).

By dropping files with specific names into this folder, you override default behaviors, customize schemas, configure hyper-parameters, inject custom routers, and define tools.

---

## 📂 File Blueprint

Here is the complete list of files you can add to your agent folder and their override order:

```text
agents/my_agent/
├── __init__.py      # [REQUIRED] Manifest declaration + run entrypoint
├── graph.py         # [OPTIONAL] LangGraph StateGraph pipeline definition
├── nodes.py         # [OPTIONAL] Logic blocks (nodes) for your graph/agent
├── config.py        # [OPTIONAL] Custom Pydantic configuration schema
├── schemas.py       # [OPTIONAL] Custom Pydantic API input/output models
├── prompts.json     # [OPTIONAL] JSON-based versioned prompt templates
├── tools.py         # [OPTIONAL] LangChain tools for tool-calling agents
├── api.py           # [OPTIONAL] Custom router (REPLACES auto-generated endpoints)
├── langgraph.json   # [OPTIONAL] LangGraph Studio visual debugger config
├── .env.example     # [OPTIONAL] Template for agent-specific env variables
└── README.md        # [OPTIONAL] Markdown documentation for this agent
```

---

## 🛠️ Required Files

### `__init__.py` (Manifest & Entrypoint)

Every agent must export two core objects:
1. **`manifest`**: An `AgentManifest` instance declaring metadata.
2. **`node_fn`** (or **`graph_fn`**): The execution entry point callable.

```python
# agents/my_agent/__init__.py
from agentomatic import AgentManifest

# 1. Declare Agent Metadata
manifest = AgentManifest(
    name="my_agent",              # Unique snake_case name (used in URL paths)
    slug="my-agent",              # Human-readable slug (shown in UI)
    description="Analyzes documents and formats summaries.",
    intent_keywords=["summarize", "analyze", "explain"],
    version="1.2.0",              # Semantic version
    framework="custom",           # custom | langgraph | langchain
    is_subagent=False,            # Set True to hide from global list / cards
    metadata={"cost_center": "A1"} # Custom dictionary metadata
)

# 2. Define Execution logic
async def node_fn(state: dict) -> dict:
    """Invoked when the client hits /invoke or /chat."""
    query = state.get("current_query", "")
    return {
        "response": f"Processed: {query}",
        "steps_taken": ["parsing", "done"],
        "suggestions": ["Show analytics", "Draft email"]
    }
```

---

## 🔌 Optional Overrides

### 1. `graph.py` & `nodes.py` (LangGraph Pipelines)
For multi-step agents using **LangGraph**, separate your StateGraph configuration (`graph.py`) from your functional nodes (`nodes.py`).

In `__init__.py`, import the compiled graph and export it via `graph_fn`:

```python
# agents/my_agent/__init__.py
from .graph import get_graph
from agentomatic import AgentManifest

manifest = AgentManifest(name="my_agent", framework="langgraph", ...)
graph_fn = get_graph
```

### 2. `config.py` (Pydantic Settings)
Allows you to define a Pydantic `BaseModel` representing default configuration settings. Agentomatic instantiates this class and exposes it at `/api/v1/{agent_name}/config` and inside the chat playground.

```python
# agents/my_agent/config.py
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    prompt_version: str = Field("v1", description="Active prompt key")
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(2048)
    llm_model: str = Field("ollama/mistral:7b")
```

Access this configuration dynamically in your nodes:
```python
from agentomatic import AgentRegistry
# Retrieve from singleton registry
config = AgentRegistry().get("my_agent").config
```

### 3. `schemas.py` (API Contracts Overrides)
By default, Agentomatic exposes:
- Input schema: `AgentInvokeRequest`
- Output schema: `AgentInvokeResponse`

You can override these schemas entirely by creating a `schemas.py` file. **Agentomatic parses this file and automatically rebuilds the FastAPI OpenAPI contracts and Swagger documentation!**

```python
# agents/my_agent/schemas.py
from pydantic import BaseModel, Field

class CustomInvokeRequest(BaseModel):
    query: str = Field(..., description="The input query text")
    user_id: str = Field("guest")
    file_attachment_url: str | None = Field(None, description="Optional doc URL")

class CustomInvokeResponse(BaseModel):
    response: str = Field(..., description="Markdown response text")
    tokens_used: int = Field(0)
    success: bool = Field(True)
```

> 📚 For a complete reference on naming conventions, validation error handling, and runtime behavior, see the [Input & Output Schemas Guide](schemas.md).

### 4. `prompts.json` (Versioned Prompt Templates)
Decouples prompt text from code, enabling hot-reloads and A/B version testing.

```json
{
  "v1": {
    "system": "You are a concise assistant.",
    "user_template": "Query: {query}"
  },
  "v2": {
    "system": "You are a creative assistant.",
    "user_template": "Tell me a story about {query}"
  }
}
```

Format templates using the built-in `PromptManager`:
```python
from agentomatic import PromptManager
from pathlib import Path

pm = PromptManager("my_agent", Path(__file__).parent / "prompts.json")
prompt = pm.format_prompt(version="v1", prompt_type="user_template", query="AI")
```

### 5. `tools.py` (LangChain Tools)
If your agent uses LangChain's function calling capabilities, define your list of tools in `tools.py`:

```python
# agents/my_agent/tools.py
from langchain_core.tools import tool

@tool
def calculate_salary(employee_id: str) -> float:
    """Query base salary for a given employee id."""
    return 5500.00

agent_tools = [calculate_salary]
```

### 6. `api.py` (FastAPI Router Replacement)
If `api.py` exists and exports a variable named `router` (a FastAPI `APIRouter` instance), **Agentomatic drops all 12 auto-generated endpoints for this agent and mounts this router instead!** This gives you absolute control over custom paths, files uploads, WebSocket integrations, or custom authorization logic.

```python
# agents/my_agent/api.py
from fastapi import APIRouter, File, UploadFile
from agentomatic import APIResponse, handle_api_errors

router = APIRouter()

@router.post("/custom-action")
@handle_api_errors
async def custom_action(data: dict) -> APIResponse:
    return APIResponse(success=True, data={"result": "Custom processing"})
```
