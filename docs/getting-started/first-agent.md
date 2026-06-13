# Your First Agent

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Step-by-Step RAG Agent Tutorial</h3>
</div>

---

This tutorial walks through building a production-ready **Retrieval-Augmented Generation (RAG)** search agent from scratch using Agentomatic.

---

## Step 1: Scaffold the Agent Folder

Use the `full` scaffolding template to include all optional configuration files and overrides:

```bash
agentomatic init search_bot --template full
```

This creates a dedicated directory containing:

```text
agents/search_bot/
├── __init__.py      # Required: agent manifest declaration and entry point
├── graph.py         # Optional: LangGraph orchestration flow
├── nodes.py         # Optional: node execution logic
├── config.py        # Optional: agent settings schema (Pydantic)
├── schemas.py       # Optional: custom request/response validation schemas
├── tools.py         # Optional: LangChain tools
├── api.py           # Optional: custom routers (overrides auto-generated endpoints)
├── prompts.json     # Optional: versioned prompt templates
├── langgraph.json   # Optional: local developer environment settings
├── .env.example     # Optional: environment blueprint
└── README.md        # Optional: agent readme documentation
```

---

## Step 2: Declare the Agent Manifest

Open `agents/search_bot/__init__.py`. Define your manifest properties:

```python
from agentomatic import AgentManifest
from .graph import get_graph

manifest = AgentManifest(
    name="search_bot",
    slug="search-bot",
    description="Knowledge base search assistant utilizing LangGraph and Vector Stores.",
    intent_keywords=["search", "find", "document", "knowledge"],
    version="1.0.0",
    framework="langgraph",
)

def graph_fn():
    """Retrieve the LangGraph compiled state graph."""
    return get_graph()
```

---

## Step 3: Implement RAG Nodes

Open `agents/search_bot/nodes.py`. Define your document retrieval and answer generation logic:

```python
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

async def retrieve_context(state: dict[str, Any]) -> dict[str, Any]:
    """Mock document retriever node."""
    query = state.get("current_query", "")
    logger_info = f"Querying KB: {query}"
    
    # Mock retrieval docs (swap with a real VectorDB like Qdrant/Chroma)
    docs = [
        "Company policy: Employees get 25 days of paid time off per year.",
        "Requesting leaves: Submit request in HR portal 2 weeks in advance."
    ]
    return {
        "citations": [{"source": "company_handbook.pdf", "page": 10}],
        "metadata": {"retrieved_docs": docs},
        "steps_taken": ["retrieve_docs"],
    }

async def generate_response(state: dict[str, Any]) -> dict[str, Any]:
    """Generates the final response based on retrieved docs."""
    query = state.get("current_query", "")
    context = state.get("metadata", {}).get("retrieved_docs", [])
    
    # Format a prompt template and query the LLM
    llm = ChatOllama(model="mistral:7b", temperature=0.1)
    prompt = f"Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}"
    
    result = await llm.ainvoke([HumanMessage(content=prompt)])
    
    return {
        "response": result.content,
        "suggestions": ["How to request leaves?", "PTO balance check"],
        "steps_taken": ["generate_response"],
    }
```

---

## Step 4: Configure Settings Schema

Open `agents/search_bot/config.py`. Define default agent settings and hyper-parameters using Pydantic:

```python
from pydantic import BaseModel, Field

class SearchBotConfig(BaseModel):
    prompt_version: str = Field("v1", description="Default prompt version")
    temperature: float = Field(0.2, description="Sampling temperature")
    max_tokens: int = Field(2048, description="Maximum completion tokens")
    vector_store_url: str = Field("http://localhost:6333", description="Vector database URL")
```

---

## Step 5: Test the Agent

Start the platform server locally, loading your newly created agent:

```bash
agentomatic run --reload --with-ui
```

1. **REST API endpoints**: Go to `http://localhost:8000/docs` to test `/api/v1/search_bot/invoke` interactively.
2. **Graphical Sandbox**: Go to `http://localhost:8000/chat` to test token-by-token streaming and rating collection.
