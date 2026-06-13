# Quick Start

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>API Platform in 60 Seconds</h3>
</div>

---

## 1. Install dependencies

Install Agentomatic with all features:

```bash
pip install agentomatic[all]
```

---

## 2. Scaffold a chatbot agent

Generate a functional basic chatbot agent using the CLI:

```bash
agentomatic init my_chatbot --template basic
```

This scaffolds a directory structure under `agents/`:

```text
agents/my_chatbot/
├── __init__.py      # Manifest declaration + route hook entry
├── graph.py         # LangGraph state graph definition
├── nodes.py         # Node logic functions
├── prompts.json     # Dynamic system and user prompt templates
├── langgraph.json   # LangGraph Studio local developer settings
├── .env.example     # Environment variables blueprint
└── README.md        # Agent documentation markdown
```

---

## 3. Run the API server

=== "CLI Launcher (recommended)"
    ```bash
    # Starts the web server and embeds the graphical Chat UI at /chat
    agentomatic run --with-ui --reload
    ```
    - **FastAPI application running at**: `http://localhost:8000`
    - **OpenAPI Swagger documentation at**: `http://localhost:8000/docs`
    - **Chainlit chat playground at**: `http://localhost:8000/chat`

=== "Python Entry Point"
    Create a `main.py` file to customize the platform stack:
    ```python
    # main.py
    from agentomatic import AgentPlatform

    platform = AgentPlatform.from_folder("agents/")
    app = platform.build()
```
    Run with uvicorn:
    ```bash
    uvicorn main:app --reload
    ```

---

## 4. Query the endpoints

You can interact with your newly deployed agent using Curl, Python, or the CLI.

=== "curl (REST API)"
    ```bash
    curl -X POST http://localhost:8000/api/v1/my_chatbot/invoke \
      -H "Content-Type: application/json" \
      -d '{"query": "Hello!"}'
    ```

=== "Python"
    ```python
    import httpx

    response = httpx.post(
        "http://localhost:8000/api/v1/my_chatbot/invoke",
        json={"query": "Hello!"},
    )
    print(response.json())
    ```

=== "Interactive CLI"
    ```bash
    # Opens a chat-like session inside your terminal
    agentomatic test my_chatbot
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

## 🧭 Explore Further

Now that you have your first API microservice running, here is where to look next:

- **[Your First Agent](first-agent.md)** — Step-by-step tutorial building a production-ready RAG agent.
- **[Agent Structure](../guide/agent-structure.md)** — Understand directory conventions and how overrides work.
- **[Prompt Optimization](../guide/optimization.md)** — Auto-tune your system prompts using machine learning.
- **[Storage Backends](../guide/storage.md)** — Configure PostgreSQL or custom Redis adapters.
