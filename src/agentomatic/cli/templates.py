"""Agent scaffolding templates.

Each template is a dict mapping relative file paths to their content.
Templates: basic, full, rag, chatbot, custom, deepagent.
"""

from __future__ import annotations


def _init_py(name: str, description: str, keywords: str, framework: str = "langgraph") -> str:
    return f'''"""Agent: {name}."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{description}",\n    intent_keywords=[{keywords}],\n    framework="{framework}",\n)\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    from .graph import get_graph\n    return await get_graph().ainvoke(state)\n'''


def _graph_py(name: str) -> str:
    return f'''"""LangGraph graph for {name}."""\nfrom __future__ import annotations\n\nfrom functools import lru_cache\n\nfrom langgraph.graph import END, StateGraph\n\nfrom agentomatic import BaseAgentState\n\nfrom . import nodes\n\n\ndef build_graph() -> StateGraph:\n    g = StateGraph(BaseAgentState)\n    g.add_node("process", nodes.process)\n    g.set_entry_point("process")\n    g.add_edge("process", END)\n    return g\n\n\n@lru_cache(maxsize=1)\ndef get_graph():\n    return build_graph().compile()\n'''


def _nodes_py(name: str) -> str:
    return f'''"""Node functions for {name}."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\nasync def process(state: dict[str, Any]) -> dict[str, Any]:\n    query = state.get("current_query", "")\n    return {{\n        "response": f"Hello from {name}! You asked: {{query}}",\n        "agent_type": "agent-{name}",\n        "suggestions": ["Tell me more", "Help me with something else"],\n    }}\n'''


def _config_py(name: str) -> str:
    title = name.replace("_", " ").title()
    return f'''"""Configuration for {title} agent."""\nfrom __future__ import annotations\n\nfrom pydantic import BaseModel, Field\n\n\nclass {title.replace(" ", "")}Config(BaseModel):\n    """Agent-specific configuration."""\n\n    prompt_version: str = Field("v1", description="Active prompt version")\n    temperature: float = Field(0.1, ge=0.0, le=2.0)\n    max_tokens: int = Field(2048, ge=1)\n    enable_memory: bool = Field(True, description="Enable conversation memory")\n'''


def _schemas_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Custom schemas for {name}."""\nfrom __future__ import annotations\n\nfrom pydantic import BaseModel, Field\n\n\nclass {title}Request(BaseModel):\n    """Custom request model."""\n    query: str = Field(..., description="User query")\n    context: dict = Field(default_factory=dict)\n\n\nclass {title}Response(BaseModel):\n    """Custom response model."""\n    answer: str\n    confidence: float = Field(0.0, ge=0.0, le=1.0)\n    sources: list[str] = Field(default_factory=list)\n'''


def _tools_py(name: str) -> str:
    return f'''"""LangChain-compatible tools for {name}."""\nfrom __future__ import annotations\n\n\ndef search(query: str) -> str:\n    """Search for information.\n\n    Args:\n        query: Search query string.\n\n    Returns:\n        Search results.\n    """\n    return f"Results for: {{query}}"\n'''


def _api_py(name: str) -> str:
    return f'''"""Custom API router for {name}.\n\nIf this file exports a `router`, it REPLACES the auto-generated endpoints.\nRemove this file to use auto-generated endpoints instead.\n"""\nfrom __future__ import annotations\n\nfrom fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n@router.get("/status")\nasync def status() -> dict:\n    """Custom status endpoint."""\n    return {{"agent": "{name}", "custom_router": True}}\n'''


def _prompts_json() -> str:
    return """{
    "v1": {
        "system": "You are a helpful AI assistant. Be concise and accurate.",
        "user_template": "{query}"
    },
    "v2": {
        "system": "You are an advanced AI assistant. Provide detailed, well-structured responses with examples when helpful.",
        "user_template": "Please help with the following: {query}"
    }
}
"""


def _langgraph_json() -> str:
    return """{
    "dependencies": ["."],
    "graphs": {
        "agent": "./graph.py:get_graph"
    },
    "env": ".env"
}
"""


def _env_example(name: str) -> str:
    upper = name.upper()
    return f"""# {name} agent configuration\n# Copy to .env and fill in values\n\n# LLM Settings\n{upper}_LLM_PROVIDER=ollama\n{upper}_LLM_MODEL=mistral:7b\n{upper}_TEMPERATURE=0.1\n{upper}_MAX_TOKENS=2048\n\n# Feature Flags\n{upper}_ENABLE_MEMORY=true\n{upper}_ENABLE_STREAMING=true\n"""


def _readme_md(name: str, template: str) -> str:
    title = name.replace("_", " ").title()
    return f"""# {title} Agent\n\nGenerated with `agentomatic init {name} --template {template}`.\n\n## Quick Start\n\n```bash\n# Start the platform\nagentomatic run\n\n# Test the agent\ncurl -X POST http://localhost:8000/api/v1/{name}/invoke \\\n  -H "Content-Type: application/json" \\\n  -d '{{"query": "Hello!"}}'\n```\n\n## Files\n\n| File | Purpose |\n|------|---------|\n| `__init__.py` | Agent manifest and entry point |\n| `graph.py` | LangGraph state graph |\n| `nodes.py` | Node processing functions |\n| `config.py` | Agent-specific configuration |\n| `prompts.json` | Versioned prompt templates |\n| `langgraph.json` | LangGraph Studio config |\n"""


# --- RAG-specific templates ---


def _rag_nodes_py(name: str) -> str:
    return f'''"""RAG node functions for {name}."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\nasync def retrieve(state: dict[str, Any]) -> dict[str, Any]:\n    """Retrieve relevant documents."""\n    query = state.get("current_query", "")\n    # TODO: Replace with real vector search\n    docs = [\n        {{"content": f"Document about {{query}}", "source": "knowledge_base"}},\n    ]\n    return {{"citations": docs, "steps_taken": ["retrieved_docs"]}}\n\n\nasync def generate(state: dict[str, Any]) -> dict[str, Any]:\n    """Generate response using retrieved context."""\n    query = state.get("current_query", "")\n    citations = state.get("citations", [])\n    context = "\\n".join(d.get("content", "") for d in citations)\n    return {{\n        "response": f"Based on the knowledge base: Answer to '{{query}}' using context: {{context}}",\n        "agent_type": "agent-{name}",\n        "steps_taken": ["generated_response"],\n    }}\n'''


def _rag_graph_py(name: str) -> str:
    return f'''"""RAG graph for {name}: retrieve -> generate."""\nfrom __future__ import annotations\n\nfrom functools import lru_cache\n\nfrom langgraph.graph import END, StateGraph\n\nfrom agentomatic import BaseAgentState\n\nfrom . import nodes\n\n\ndef build_graph() -> StateGraph:\n    g = StateGraph(BaseAgentState)\n    g.add_node("retrieve", nodes.retrieve)\n    g.add_node("generate", nodes.generate)\n    g.set_entry_point("retrieve")\n    g.add_edge("retrieve", "generate")\n    g.add_edge("generate", END)\n    return g\n\n\n@lru_cache(maxsize=1)\ndef get_graph():\n    return build_graph().compile()\n'''


# --- Chatbot-specific templates ---


def _chatbot_nodes_py(name: str) -> str:
    return f'''"""Chatbot node functions for {name} with conversation memory."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\nasync def respond(state: dict[str, Any]) -> dict[str, Any]:\n    """Generate a conversational response."""\n    query = state.get("current_query", "")\n    messages = state.get("messages", [])\n    history_len = len(messages)\n\n    # TODO: Replace with real LLM call\n    return {{\n        "response": f"[Turn {{history_len + 1}}] You said: {{query}}",\n        "agent_type": "agent-{name}",\n        "suggestions": ["Tell me more", "Change topic", "Goodbye"],\n    }}\n'''


def _chatbot_graph_py(name: str) -> str:
    return f'''"""Chatbot graph for {name} with memory."""\nfrom __future__ import annotations\n\nfrom functools import lru_cache\n\nfrom langgraph.graph import END, StateGraph\n\nfrom agentomatic import BaseAgentState\n\nfrom . import nodes\n\n\ndef build_graph() -> StateGraph:\n    g = StateGraph(BaseAgentState)\n    g.add_node("respond", nodes.respond)\n    g.set_entry_point("respond")\n    g.add_edge("respond", END)\n    return g\n\n\n@lru_cache(maxsize=1)\ndef get_graph():\n    return build_graph().compile()\n'''


# --- Deep Agent template ---


def _deepagent_init_py(name: str, description: str, keywords: str) -> str:
    return f'''"""Agent: {name} (Deep Agent harness)."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{description}",\n    intent_keywords=[{keywords}],\n    framework="langgraph",\n)\n\n\ndef graph_fn():\n    """Return the compiled deep agent graph."""\n    from .agent import create_agent\n    return create_agent()\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    """Invoke the deep agent."""\n    return await graph_fn().ainvoke(state)\n'''


def _deepagent_agent_py(name: str) -> str:
    safe_title = name.replace("_", " ").title()
    return f'''"""Deep Agent definition for {name}.\n\nUses LangChain\'s `deepagents` harness for planning, tools,\nsubagent delegation, and context management.\n"""\nfrom __future__ import annotations\n\nfrom functools import lru_cache\n\n\ndef internet_search(query: str, max_results: int = 5) -> str:\n    """Search the internet for information.\n\n    Args:\n        query: Search query string.\n        max_results: Maximum number of results.\n\n    Returns:\n        Search results as text.\n    """\n    # TODO: Replace with real search (Tavily, SerpAPI, etc.)\n    return f"Search results for: {{query}} ({{max_results}} results)"\n\n\n@lru_cache(maxsize=1)\ndef create_agent():\n    """Create and compile the deep agent."""\n    from deepagents import create_deep_agent\n\n    return create_deep_agent(\n        model="openai:gpt-4o",  # Change to your preferred model\n        system_prompt=(\n            "You are {safe_title}, "\n            "an expert AI assistant. Be thorough and accurate."\n        ),\n        tools=[internet_search],\n    )\n'''


# --- Custom (no LangGraph) template ---


def _custom_init_py(name: str, description: str, keywords: str) -> str:
    return f'''"""Agent: {name} (framework-agnostic)."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{description}",\n    intent_keywords=[{keywords}],\n    framework="custom",\n)\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    """Process the request directly — no graph framework needed."""\n    query = state.get("current_query", "")\n    return {{\n        "response": f"Hello from {name}! You asked: {{query}}",\n        "agent_type": "agent-{name}",\n    }}\n'''


# =====================================================================
# Template Registry
# =====================================================================

TEMPLATES: dict[str, str] = {
    "basic": "Minimal agent — 3 files, quick start",
    "full": "All overwrite files — config, schemas, api, tools, prompts",
    "rag": "RAG agent — retrieve → generate pipeline",
    "chatbot": "Conversational agent with memory",
    "deepagent": "Deep Agent — planning, tools, subagents (requires deepagents package)",
    "custom": "Framework-agnostic — no LangGraph dependency",
}


def get_template_files(template: str, name: str) -> dict[str, str]:
    """Get all files for a given template.

    Args:
        template: Template name (basic, full, rag, chatbot, custom).
        name: Agent name.

    Returns:
        Dict mapping relative file paths to content strings.
    """
    title = name.replace("_", " ").title()
    description = f"{title} agent"
    keywords = f'"{name}"'

    common = {
        "prompts.json": _prompts_json(),
        "langgraph.json": _langgraph_json(),
        ".env.example": _env_example(name),
        "README.md": _readme_md(name, template),
    }

    if template == "basic":
        return {
            "__init__.py": _init_py(name, description, keywords),
            "graph.py": _graph_py(name),
            "nodes.py": _nodes_py(name),
            **common,
        }

    elif template == "full":
        return {
            "__init__.py": _init_py(name, description, keywords),
            "graph.py": _graph_py(name),
            "nodes.py": _nodes_py(name),
            "config.py": _config_py(name),
            "schemas.py": _schemas_py(name),
            "tools.py": _tools_py(name),
            "api.py": _api_py(name),
            **common,
        }

    elif template == "rag":
        return {
            "__init__.py": _init_py(
                name, f"{title} RAG agent", f'"{name}", "search", "knowledge"'
            ),
            "graph.py": _rag_graph_py(name),
            "nodes.py": _rag_nodes_py(name),
            "config.py": _config_py(name),
            "tools.py": _tools_py(name),
            **common,
        }

    elif template == "chatbot":
        return {
            "__init__.py": _init_py(name, f"{title} chatbot", f'"{name}", "chat", "conversation"'),
            "graph.py": _chatbot_graph_py(name),
            "nodes.py": _chatbot_nodes_py(name),
            "config.py": _config_py(name),
            **common,
        }

    elif template == "deepagent":
        return {
            "__init__.py": _deepagent_init_py(name, f"{title} deep agent", keywords),
            "agent.py": _deepagent_agent_py(name),
            "config.py": _config_py(name),
            ".env.example": _env_example(name),
            "README.md": _readme_md(name, template),
            "prompts.json": _prompts_json(),
        }

    elif template == "custom":
        return {
            "__init__.py": _custom_init_py(name, description, keywords),
            ".env.example": _env_example(name),
            "README.md": _readme_md(name, template),
            "prompts.json": _prompts_json(),
        }

    else:
        raise ValueError(f"Unknown template: {template}. Choose from: {list(TEMPLATES.keys())}")
