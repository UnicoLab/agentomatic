# Prompt Management

Agentomatic provides native, JSON-based prompt versioning and template management out of the box. Instead of hardcoding prompts in Python files, you can define them in `prompts.json` within your agent's directory and load/format them dynamically using `PromptManager`.

---

## 📂 The `prompts.json` file

The `prompts.json` file is located in your agent's directory (e.g., `agents/my_agent/prompts.json`). It follows a simple nested structure:

- **Top-level keys**: Version tags (e.g., `"v1"`, `"v2"`, `"v1_formal"`).
- **Nested keys**: Prompt types (e.g., `"system"`, `"user_template"`, `"chat_template"`).

### Example Structure

```json
{
  "v1": {
    "system": "You are a helpful assistant. Keep your answers concise.",
    "user_template": "Question: {query}\n\nContext: {context}"
  },
  "v2": {
    "system": "You are a friendly assistant. Format your response using bullet points.",
    "user_template": "Hello! Here is a request from {username}: {query}"
  }
}
```

---

## ⚙️ Programmatic Access via `PromptManager`

You can import and use the `PromptManager` class directly from the root of `agentomatic`. It reads the `prompts.json` file, parses the templates, and provides helper methods to access and format them.

### Initialization

```python
from pathlib import Path
from agentomatic import PromptManager

# Load prompts for an agent
prompts_file = Path("agents/my_agent/prompts.json")
manager = PromptManager(agent_name="my_agent", prompts_file=prompts_file)
```

### Retrieving Raw Prompts

Use the `get_prompt()` method to get the raw, unformatted prompt template:

```python
# Returns: "You are a helpful assistant. Keep your answers concise."
system_prompt = manager.get_prompt(version="v1", prompt_type="system")

# Returns None if the version or prompt_type doesn't exist
missing_prompt = manager.get_prompt(version="v3", prompt_type="system")
```

### Formatting Prompts

Use the `format_prompt()` method to format variables into a template:

```python
# Formats variables into the 'user_template' template
user_prompt = manager.format_prompt(
    version="v1",
    prompt_type="user_template",
    query="What is the weather today?",
    context="Location: Paris, France"
)

# Output:
# Question: What is the weather today?
#
# Context: Location: Paris, France
```

### Listing Available Versions

```python
# Returns: ["v1", "v2"]
versions = manager.list_versions()
```

### Hot-Reloading from Disk

If you update the `prompts.json` file at runtime, you can reload the prompts:

```python
manager.reload(prompts_file)
```

---

## 🔌 Integration with Custom Routers and Nodes

### 1. Custom Router (`api.py`)

If you write a custom router using `api.py` and want to load prompt templates dynamically:

```python
from fastapi import APIRouter
from agentomatic import APIResponse, handle_api_errors, AgentRegistry

router = APIRouter()

@router.post("/custom-chat")
@handle_api_errors
async def custom_chat(query: str) -> APIResponse:
    # Retrieve the agent instance from the registry
    agent = AgentRegistry().get("my_agent")
    if not agent:
        return APIResponse(success=False, message="Agent not found")

    # Format user prompt
    user_prompt = agent.prompt_manager.format_prompt(
        version=agent.config.prompt_version,
        prompt_type="user_template",
        query=query
    )

    return APIResponse(success=True, data={"prompt_used": user_prompt})
```

### 2. LangGraph Nodes (`nodes.py`)

You can access the `PromptManager` from within your LangGraph processing nodes:

```python
from typing import Any
from agentomatic import PromptManager
from pathlib import Path

# Load prompts helper
prompts_path = Path(__file__).parent / "prompts.json"
pm = PromptManager("my_agent", prompts_path)

async def generate_response(state: dict[str, Any]) -> dict[str, Any]:
    query = state.get("current_query", "")

    # Format prompt templates dynamically
    system_msg = pm.get_prompt(version="v1", prompt_type="system")
    user_msg = pm.format_prompt(
        version="v1",
        prompt_type="user_template",
        query=query,
        context="No context"
    )

    # Call your LLM model here using prompt templates...
    return {"response": "Processed output"}
```

---

## 📚 Related Documentation

| Topic | Link |
|-------|------|
| Agent structure & discovery | [Agent Structure](agent-structure.md) |
| Prompt optimization | [Optimization](optimization.md) |
| A/B prompt routing | [Platform Features](platform-features.md) |
| Class-based agent integration | [Class-Based Agents](class-agents.md) |
