# Your First Agent

This tutorial walks through creating a production-ready agent from scratch.

## Step 1: Choose a Template

```bash
agentomatic init weather_bot --template full
```

The `full` template includes ALL override files:

```
agents/weather_bot/
├── __init__.py      # Manifest + entry
├── graph.py         # State graph
├── nodes.py         # Processing logic
├── config.py        # Agent config (Pydantic)
├── schemas.py       # Custom request/response models
├── tools.py         # LangChain tools
├── api.py           # Custom router (replaces auto-gen)
├── prompts.json     # Versioned prompts
├── langgraph.json   # Studio config
├── .env.example     # Env vars
└── README.md        # Agent docs
```

## Step 2: Implement Your Logic

Edit `nodes.py` — this is where your agent logic lives:

```python
async def process(state: dict[str, Any]) -> dict[str, Any]:
    query = state.get("current_query", "")

    # Your LLM call here
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="mistral:7b")
    response = await llm.ainvoke(query)

    return {
        "response": response.content,
        "agent_type": "agent-weather_bot",
        "suggestions": ["Check forecast", "Weather alerts"],
    }
```

## Step 3: Configure

Edit `config.py` for agent-specific settings:

```python
class WeatherBotConfig(BaseModel):
    prompt_version: str = "v1"
    temperature: float = 0.1
    max_tokens: int = 2048
    api_key: str = Field("", description="Weather API key")
```

## Step 4: Add Storage & Middleware

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore

platform = AgentPlatform.from_folder(
    "agents/",
    store=MemoryStore(),
    enable_auth=True,
    auth_api_key="my-secret-key",
    enable_rate_limit=True,
    enable_metrics=True,
)
app = platform.build()
```

## Step 5: Test with Debug UI

```bash
pip install agentomatic[ui]
agentomatic run --with-ui
# Open http://localhost:8000/chat
```

!!! tip "LangGraph Studio"
    Each agent's `langgraph.json` lets you debug with LangGraph Studio:
    ```bash
    langgraph dev agents/weather_bot/langgraph.json
    ```
