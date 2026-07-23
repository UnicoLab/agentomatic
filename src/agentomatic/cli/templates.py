"""Agent scaffolding templates.

Each template is a dict mapping relative file paths to their content.
Templates: basic, full, rag, chatbot, custom, deepagent, legacy_dict.
"""

from __future__ import annotations


def _class_agent_get_graph_export(title: str) -> str:
    """Append a ``get_graph()`` export for langgraph.json / Studio tools."""
    return f'''

def get_graph():
    """Entrypoint for ``langgraph.json`` / LangGraph tooling.

    Prefer Agentomatic Studio at ``/studio/ui/`` for class-agent debugging.
    """
    return {title}Agent().graph
'''


def _class_agent_py(name: str, template: str = "basic") -> str:
    title = name.replace("_", " ").title().replace(" ", "")

    if template == "rag":
        return f'''"""RAG class-based agent: {name}."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentomatic.agents import BaseGraphAgent


@dataclass
class {title}State:
    """Agent state — per-run transient data."""
    request: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


class {title}Agent(BaseGraphAgent[{title}State]):
    """RAG class agent for {name}.

    LLM and prompts come from the active stack + ``prompts.json`` —
    do not hardcode model names here.
    """

    agent_name = "{name}"
    agent_description = "RAG agent"
    agent_framework = "graph_agent"

    def __init__(self, *, llm: Any = None, prompt_manager: Any = None) -> None:
        super().__init__()
        self.llm = llm
        self.prompt_manager = prompt_manager

    def _system_prompt(self) -> str:
        # Honour optimize/fit overrides via resolve_system_prompt (compiled_config,
        # system_prompt_override, prompt_manager) — required for train/optimize.
        return self.resolve_system_prompt(
            default="You are a helpful RAG assistant. Ground answers in retrieved context."
        )

    def build_graph(self):
        g = self.new_graph()
        g.add_node("retrieve", self.retrieve)
        g.add_node("generate", self.generate)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.set_finish_point("generate")
        return g.compile()

    def retrieve(self, state: {title}State) -> {title}State:
        # TODO: use get_connections("{name}").vector("kb").as_store() for real RAG
        state.citations = [
            {{"content": f"Document about {{state.request}}", "source": "knowledge_base"}}
        ]
        return state

    def generate(self, state: {title}State) -> {title}State:
        context = "\\n".join(d.get("content", "") for d in state.citations)
        prompt = self._system_prompt()
        if self.llm is not None:
            try:
                msg = (
                    f"{{prompt}}\\n\\nContext:\\n{{context}}\\n\\n"
                    f"Question: {{state.request}}"
                )
                result = self.llm.invoke(msg)
                text = getattr(result, "content", None) or str(result)
            except Exception as exc:  # noqa: BLE001
                text = f"(llm error: {{exc}}) Answer for '{{state.request}}'"
        else:
            text = (
                f"Based on the knowledge base: Answer to '{{state.request}}' "
                f"using context: {{context}}"
            )
        state.output = {{
            "response": text,
            "agent_type": "{name}",
            "citations": state.citations,
        }}
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> {title}State:
        return {title}State(request=input_data.get("current_query", ""))

    def state_to_output(self, state: {title}State) -> dict[str, Any]:
        return state.output
''' + _class_agent_get_graph_export(title)
    elif template == "chatbot":
        return f'''"""Chatbot class-based agent: {name}."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentomatic.agents import BaseGraphAgent


@dataclass
class {title}State:
    """Agent state — per-run transient data."""
    request: str = ""
    messages: list[Any] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


class {title}Agent(BaseGraphAgent[{title}State]):
    """Chatbot class agent for {name}."""

    agent_name = "{name}"
    agent_description = "Chatbot agent"
    agent_framework = "graph_agent"

    def __init__(self, *, llm: Any = None, prompt_manager: Any = None) -> None:
        super().__init__()
        self.llm = llm
        self.prompt_manager = prompt_manager

    def _system_prompt(self) -> str:
        return self.resolve_system_prompt(
            default="You are a friendly conversational assistant."
        )

    def build_graph(self):
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: {title}State) -> {title}State:
        history_len = len(state.messages)
        prompt = self._system_prompt()
        if self.llm is not None:
            try:
                result = self.llm.invoke(
                    f"{{prompt}}\\n\\nUser: {{state.request}}"
                )
                text = getattr(result, "content", None) or str(result)
            except Exception as exc:  # noqa: BLE001
                text = f"[Turn {{history_len + 1}}] (llm error: {{exc}})"
        else:
            text = f"[Turn {{history_len + 1}}] You said: {{state.request}}"
        state.output = {{
            "response": text,
            "agent_type": "{name}",
            "suggestions": ["Tell me more", "Change topic", "Goodbye"],
        }}
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> {title}State:
        return {title}State(
            request=input_data.get("current_query", ""),
            messages=input_data.get("messages", [])
        )

    def state_to_output(self, state: {title}State) -> dict[str, Any]:
        return state.output
''' + _class_agent_get_graph_export(title)
    else:
        # Basic / Full
        return f'''"""Basic class-based agent: {name}."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentomatic.agents import BaseGraphAgent


@dataclass
class {title}State:
    """Agent state — per-run transient data."""
    request: str = ""
    context: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


class {title}Agent(BaseGraphAgent[{title}State]):
    """Class agent for {name}.

    The platform injects ``llm`` from the active stack (see ``llm.py``) and
    ``prompt_manager`` from ``prompts.json``. Prefer those over hardcoding.

    Usage::

        agent = {title}Agent(llm=my_llm)
        result = agent.transform({{"current_query": "Hello"}})
    """

    agent_name = "{name}"
    agent_description = "{title} agent"
    agent_framework = "graph_agent"

    def __init__(self, *, llm: Any = None, prompt_manager: Any = None) -> None:
        super().__init__()
        self.llm = llm
        self.prompt_manager = prompt_manager

    def _system_prompt(self) -> str:
        return self.resolve_system_prompt(default="You are a helpful assistant.")

    def build_graph(self):
        g = self.new_graph()
        g.add_node("process", self.process)
        g.set_entry_point("process")
        g.set_finish_point("process")
        return g.compile()

    def process(self, state: {title}State) -> {title}State:
        prompt = self._system_prompt()
        state.context = [f"Processed: {{state.request}}"]
        if self.llm is not None:
            try:
                result = self.llm.invoke(
                    f"{{prompt}}\\n\\nUser: {{state.request}}"
                )
                text = getattr(result, "content", None) or str(result)
            except Exception as exc:  # noqa: BLE001
                text = f"Result for: {{state.request}} (llm error: {{exc}})"
        else:
            text = f"Result for: {{state.request}}"
        state.output = {{
            "response": text,
            "agent_type": "{name}",
        }}
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> {title}State:
        return {title}State(request=input_data.get("current_query", ""))

    def state_to_output(self, state: {title}State) -> dict[str, Any]:
        return state.output
''' + _class_agent_get_graph_export(title)


def _agent_manifest_init_py(name: str, description: str, keywords: str) -> str:
    """Generate ``__init__.py`` with an explicit AgentManifest (agent card)."""
    return f'''"""Agent: {name}."""
from __future__ import annotations

from agentomatic import AgentManifest

manifest = AgentManifest(
    name="{name}",
    slug="agent-{name}",
    description="{description}",
    intent_keywords=[{keywords}],
    framework="graph_agent",
    version="1.0.0",
)

__all__ = ["manifest"]
'''


def _llm_py(name: str) -> str:
    """Generate stack-driven ``llm.py`` for an agent package."""
    return f'''"""LLM wiring for {name} — driven by the active stack, not hardcoded.

The registry discovers ``llm_config`` / ``AgentLLMConfig`` and the platform
resolves the named profile from ``stacks/<active>.yaml``.

Roles map logical agent roles → stack LLM profile names
(e.g. ``default``, ``fast``, ``judge``).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentLLMConfig:
    """Per-agent LLM role → stack profile mapping."""

    roles: dict[str, str] = field(
        default_factory=lambda: {{
            "default": "default",
            # "planner": "default",
            # "fast": "fast",
        }}
    )


llm_config = AgentLLMConfig()
'''


def _config_py(name: str) -> str:
    title = name.replace("_", " ").title()
    return f'''"""Configuration for {title} agent."""\nfrom __future__ import annotations\n\nfrom pydantic import BaseModel, Field\n\n\nclass {title.replace(" ", "")}Config(BaseModel):\n    """Agent-specific configuration."""\n\n    prompt_version: str = Field("v1", description="Active prompt version")\n    temperature: float = Field(0.1, ge=0.0, le=2.0)\n    max_tokens: int = Field(8192, ge=1)\n    enable_memory: bool = Field(True, description="Enable conversation memory")\n'''


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


def _langgraph_json(*, graph_target: str = "./agent.py:get_graph") -> str:
    """Return langgraph.json pointing at a real module export."""
    return f"""{{
    "dependencies": ["."],
    "graphs": {{
        "agent": "{graph_target}"
    }},
    "env": ".env"
}}
"""


def _env_example(name: str) -> str:
    upper = name.upper()
    return f"""# {name} agent configuration\n# Copy to .env and fill in values\n\n# LLM Settings\n{upper}_LLM_PROVIDER=ollama\n{upper}_LLM_MODEL=mistral:7b\n{upper}_TEMPERATURE=0.1\n{upper}_MAX_TOKENS=2048\n\n# Feature Flags\n{upper}_ENABLE_MEMORY=true\n{upper}_ENABLE_STREAMING=true\n"""


def _readme_md(name: str, template: str) -> str:
    title = name.replace("_", " ").title()
    return f"""# {title} Agent\n\nGenerated with `agentomatic init {name} --template {template}`.\n\n## Quick Start\n\n```bash\n# Start the platform\nagentomatic run\n\n# Test the agent\ncurl -X POST http://localhost:8000/api/v1/{name}/invoke \\\n  -H "Content-Type: application/json" \\\n  -d '{{"query": "Hello!"}}'\n```\n\n## Files\n\n| File | Purpose |\n|------|---------|\n| `agent.py` | Agent class definition |\n| `config.py` | Agent-specific configuration |\n| `prompts.json` | Versioned prompt templates |\n| `langgraph.json` | LangGraph Studio config |\n"""


# --- Deep Agent template ---


def _deepagent_init_py(name: str, description: str, keywords: str) -> str:
    return f'''"""Agent: {name} (Deep Agent harness)."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{description}",\n    intent_keywords=[{keywords}],\n    framework="langgraph",\n)\n\n\ndef graph_fn():\n    """Return the compiled deep agent graph."""\n    from .agent import create_agent\n    return create_agent()\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    """Invoke the deep agent."""\n    return await graph_fn().ainvoke(state)\n'''


def _deepagent_agent_py(name: str) -> str:
    safe_title = name.replace("_", " ").title()
    return f'''"""Deep Agent definition for {name}.

Uses LangChain's `deepagents` harness for planning, tools,
subagent delegation, and context management.

The model id is resolved from the active stack's default LLM profile
(``stacks/<active>.yaml``) via ``get_llm_for_agent``, with an env-var
override. Never hardcode production model names here.
"""
from __future__ import annotations

import os
from functools import lru_cache


def internet_search(query: str, max_results: int = 5) -> str:
    """Search the internet for information.

    Args:
        query: Search query string.
        max_results: Maximum number of results.

    Returns:
        Search results as text.
    """
    # TODO: Replace with real search (Tavily, SerpAPI, etc.)
    return f"Search results for: {{query}} ({{max_results}} results)"


def _resolve_model() -> str:
    """Resolve the deep-agent model from env override or stack config."""
    override = os.getenv("{name.upper()}_MODEL") or os.getenv("DEEPAGENT_MODEL")
    if override:
        return override
    try:
        from agentomatic.providers.llm import get_llm_for_agent

        llm = get_llm_for_agent("{name}", role="default")
        # Best-effort extract a provider:model string
        model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        provider = getattr(llm, "_llm_type", None)
        if model:
            return f"{{provider}}:{{model}}" if provider else str(model)
    except Exception:  # noqa: BLE001
        pass
    # Provider-agnostic default: honour the platform task-model env vars before
    # falling back to a local model, rather than silently assuming OpenAI. Pin
    # explicitly with {name.upper()}_MODEL / DEEPAGENT_MODEL.
    return os.getenv(
        "AGENTOMATIC_TASK_MODEL",
        os.getenv("LLM__MODEL", "ollama/qwen2.5:7b"),
    )


@lru_cache(maxsize=1)
def create_agent():
    """Create and compile the deep agent."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=_resolve_model(),
        system_prompt=(
            "You are {safe_title}, "
            "an expert AI assistant. Be thorough and accurate."
        ),
        tools=[internet_search],
    )
'''


# --- Custom (no LangGraph) template ---


def _custom_init_py(name: str, description: str, keywords: str) -> str:
    return f'''"""Agent: {name} (framework-agnostic)."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{description}",\n    intent_keywords=[{keywords}],\n    framework="custom",\n)\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    """Process the request directly — no graph framework needed."""\n    query = state.get("current_query", "")\n    return {{\n        "response": f"Hello from {name}! You asked: {{query}}",\n        "agent_type": "agent-{name}",\n    }}\n'''


# --- Legacy Dict Pattern ---
def _legacy_init_py(
    name: str, description: str, keywords: str, framework: str = "langgraph"
) -> str:
    return f'''"""Agent: {name}."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{description}",\n    intent_keywords=[{keywords}],\n    framework="{framework}",\n)\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    from .graph import get_graph\n    return await get_graph().ainvoke(state)\n'''


def _legacy_graph_py(name: str) -> str:
    return f'''"""LangGraph graph for {name}."""\nfrom __future__ import annotations\n\nfrom functools import lru_cache\n\nfrom langgraph.graph import END, StateGraph\n\nfrom agentomatic import BaseAgentState\n\nfrom . import nodes\n\n\ndef build_graph() -> StateGraph:\n    g = StateGraph(BaseAgentState)\n    g.add_node("process", nodes.process)\n    g.set_entry_point("process")\n    g.add_edge("process", END)\n    return g\n\n\n@lru_cache(maxsize=1)\ndef get_graph():\n    return build_graph().compile()\n'''


def _legacy_nodes_py(name: str) -> str:
    return f'''"""Node functions for {name}."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\nasync def process(state: dict[str, Any]) -> dict[str, Any]:\n    query = state.get("current_query", "")\n    return {{\n        "response": f"Hello from {name}! You asked: {{query}}",\n        "agent_type": "agent-{name}",\n        "suggestions": ["Tell me more", "Help me with something else"],\n    }}\n'''


# =====================================================================
# Coordinator / Orchestrator Template
# =====================================================================


def _coordinator_agent_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Coordinator agent: {name}.

Routes user queries to the appropriate specialist agent via delegation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentomatic.agents import BaseGraphAgent


@dataclass
class {title}State:
    """Coordinator state — per-run transient data."""

    request: str = ""
    classification: str = ""
    delegated_to: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class {title}Agent(BaseGraphAgent[{title}State]):
    """Orchestrator that classifies queries and routes to specialist agents.

    Delegation targets are defined in ``delegation.py``.
    """

    agent_name = "{name}"
    agent_description = "Coordinator that routes queries to specialist agents"

    def build_graph(self):
        g = self.new_graph()
        g.add_node("classify", self.classify)
        g.add_node("route", self.route)
        g.set_entry_point("classify")
        g.add_edge("classify", "route")
        g.set_finish_point("route")
        return g.compile()

    def classify(self, state: {title}State) -> {title}State:
        """Classify the user query to determine routing.

        TODO: Replace keyword matching with an LLM classifier.
        """
        query = state.request.lower()
        # Add your routing logic here
        state.classification = "default"
        return state

    def route(self, state: {title}State) -> {title}State:
        """Route to the appropriate specialist via delegation."""
        from .delegation import get_handoff_tools

        tools = get_handoff_tools()
        if not tools:
            state.output = {{"response": f"No delegation targets configured"}}
            return state

        # Pick the first tool as default, or match by classification
        target_tool = tools[0]
        for tool in tools:
            tool_name = getattr(tool, "name", getattr(tool, "__name__", ""))
            if state.classification in tool_name:
                target_tool = tool
                break

        tool_name = getattr(target_tool, "name", getattr(target_tool, "__name__", ""))
        state.delegated_to = tool_name
        try:
            result = target_tool(state.request)
            state.output = {{
                "response": result,
                "routed_to": state.delegated_to,
            }}
        except Exception as exc:
            state.output = {{
                "response": f"Delegation failed: {{exc}}",
                "routed_to": state.delegated_to,
                "error": True,
            }}
        return state

    def input_to_state(self, data: dict[str, Any]) -> {title}State:
        return {title}State(request=data.get("current_query", ""))

    def state_to_output(self, state: {title}State) -> dict[str, Any]:
        return state.output
'''


def _coordinator_delegation_py(name: str) -> str:
    return f'''"""Delegation configuration for {name}.

This file is AUTO-DISCOVERED by the agentomatic registry.
It must export:
  - DELEGATION_TARGETS: list of agent names this agent can delegate to
  - get_handoff_tools(): function that returns handoff tools
"""
from __future__ import annotations

from agentomatic.delegation import AgentDelegator

# ── Agents this coordinator is allowed to delegate to ──────────────
# Add your specialist agent names here.
DELEGATION_TARGETS = [
    # "researcher",
    # "writer",
    # "coder",
]


def get_handoff_tools():
    """Create handoff tools for all delegation targets.

    Resolution order:
    1. langgraph-swarm (in-process) if installed and use_swarm=True
    2. HTTP via POST /api/v1/{{agent}}/invoke (fallback)

    Returns:
        List of LangChain-compatible handoff tools.
    """
    if not DELEGATION_TARGETS:
        return []

    delegator = AgentDelegator(use_swarm=True)
    return delegator.create_handoffs(
        targets=DELEGATION_TARGETS,
        descriptions={{
            # Add descriptions for each target:
            # "researcher": "Delegate factual research questions",
            # "writer": "Delegate content writing tasks",
        }},
    )
'''


def _coordinator_security_py(name: str) -> str:
    return f'''"""Security policy for {name}.

Auto-discovered by the registry. Enforces which agents
this coordinator may delegate to via ZeroTrustEnforcer.
"""
from __future__ import annotations

from agentomatic.security import AgentSecurityPolicy

from .delegation import DELEGATION_TARGETS

policy = AgentSecurityPolicy(
    allowed_delegation_targets=DELEGATION_TARGETS,
)
'''


# =====================================================================
# Pipeline Template
# =====================================================================


def _pipeline_yaml(name: str) -> str:
    return f"""# Pipeline: {name}
# Auto-discovered from pipelines/ or agents/*/pipeline.yaml
# API endpoint: POST /api/v1/pipelines/{name}/run

name: {name}
description: "Multi-step agent pipeline"
version: "1.0.0"

# Input contract
input:
  query:
    type: string
    required: true

# Output contract
output:
  response:
    type: string
  steps_completed:
    type: array

# Pipeline steps — executed in order
steps:
  - name: classify
    agent: classifier
    description: "Classify the input query"

  - name: process
    agent: processor
    description: "Process based on classification"
    condition: "len(ctx.steps.classify.output.get('response', '')) > 0"

  - name: format
    agent: formatter
    description: "Format the final response"

# Error handling
on_error: continue
timeout: 120.0
"""


def _pipeline_readme(name: str) -> str:
    title = name.replace("_", " ").title()
    return f"""# {title} Pipeline

## Overview

A multi-step agent pipeline that chains agents together in a fixed order.

## Folder Structure

```
pipelines/
  {name}.yaml          \u2190 Pipeline definition (auto-discovered)
agents/
  classifier/          \u2190 Step 1 agent
    agent.py
  processor/           \u2190 Step 2 agent
    agent.py
  formatter/           \u2190 Step 3 agent
    agent.py
```

## Usage

### Start the platform

```bash
agentomatic run
```

### Call the pipeline

```bash
curl -X POST http://localhost:8000/api/v1/pipelines/{name}/run \\
  -H "Content-Type: application/json" \\
  -d \'{{"input": {{"query": "Hello, world!"}}}}\u0027
```

### Other endpoints

```bash
# List all pipelines
curl http://localhost:8000/api/v1/pipelines

# Validate before running
curl http://localhost:8000/api/v1/pipelines/{name}/validate

# Get Mermaid diagram
curl http://localhost:8000/api/v1/pipelines/{name}/visualize
```

## Pipeline vs. Delegation

| | Pipeline | Delegation |
|---|---------|-----------|
| Routing | Static \u2014 you define the order | Dynamic \u2014 the model decides |
| Best for | Deterministic workflows | Open-ended routing |
| API | `POST /api/v1/pipelines/{{name}}/run` | `POST /api/v1/{{agent}}/invoke` |
"""


def _pipeline_eval_py(name: str) -> str:
    return f'''"""End-to-end evaluation script for the {name} pipeline.

Runs the entire pipeline against a dataset and measures quality
at both the pipeline level and per-step level.

Usage::

    python -m pipelines.{name}.eval
    python -m pipelines.{name}.eval --dataset custom_dataset.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load evaluation examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


async def run_pipeline(input_data: dict[str, Any]) -> dict[str, Any]:
    """Execute the pipeline via HTTP API.

    Requires the platform to be running: ``agentomatic run``
    """
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/pipelines/{name}/run",
            json={{"input": input_data}},
        )
        response.raise_for_status()
        return response.json()


def score_result(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, float]:
    """Score a single pipeline result against expected output.

    Customize these metrics for your use case.
    """
    scores = {{}}

    # 1. Has response
    scores["has_response"] = 1.0 if actual.get("output", {{}}).get("response") else 0.0

    # 2. Status is success
    scores["pipeline_success"] = 1.0 if actual.get("status") == "success" else 0.0

    # 3. All steps completed
    steps = actual.get("steps", {{}})
    if steps:
        completed = sum(1 for s in steps.values() if s.get("status") == "success")
        scores["step_completion"] = completed / len(steps)
    else:
        scores["step_completion"] = 0.0

    # 4. Custom: check expected output keys
    if "expected_output" in expected:
        exp = expected["expected_output"]
        out = actual.get("output", {{}})
        matching = sum(1 for k in exp if k in out)
        scores["output_match"] = matching / len(exp) if exp else 1.0

    return scores


async def evaluate(dataset_path: str, split: str = "all") -> None:
    """Run evaluation over the full dataset."""
    examples = load_dataset(dataset_path)
    print(f"\\nEvaluating pipeline '{name}' on {{len(examples)}} examples\\n")
    print("-" * 60)

    all_scores: list[dict[str, float]] = []
    failures = 0

    for i, example in enumerate(examples):
        input_data = example.get("input", example)
        t0 = time.perf_counter()

        try:
            result = await run_pipeline(input_data)
            elapsed = (time.perf_counter() - t0) * 1000
            scores = score_result(example, result)
            all_scores.append(scores)

            status = "PASS" if scores.get("pipeline_success", 0) == 1.0 else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"  [{{status}}] Example {{i + 1}} "
                f"({{elapsed:.0f}}ms) — {{scores}}"
            )
        except Exception as exc:
            failures += 1
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [ERROR] Example {{i + 1}} ({{elapsed:.0f}}ms) — {{exc}}")
            all_scores.append({{"pipeline_success": 0.0, "error": 1.0}})

    # Aggregate scores
    print("\\n" + "=" * 60)
    print(f"\\n  Pipeline: {name}")
    print(f"  Examples: {{len(examples)}}")
    print(f"  Passed:   {{len(examples) - failures}}")
    print(f"  Failed:   {{failures}}")
    print(f"  Pass rate: {{(len(examples) - failures) / len(examples):.1%}}")

    if all_scores:
        avg_scores = {{}}
        for key in all_scores[0]:
            values = [s.get(key, 0.0) for s in all_scores]
            avg_scores[key] = sum(values) / len(values)
        print(f"\\n  Average scores:")
        for k, v in avg_scores.items():
            print(f"    {{k}}: {{v:.3f}}")

    # Save report
    report_path = DATA_DIR / "eval_report.json"
    report = {{
        "pipeline": "{name}",
        "num_examples": len(examples),
        "pass_rate": (len(examples) - failures) / len(examples),
        "failures": failures,
        "avg_scores": avg_scores if all_scores else {{}},
    }}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\\n  Report saved to {{report_path}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate {name} pipeline")
    parser.add_argument(
        "--dataset", type=str, default=str(DATA_DIR / "dataset.jsonl"),
        help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--split", choices=["test", "train", "all"], default="all",
        help="Dataset split to evaluate"
    )
    args = parser.parse_args()

    asyncio.run(evaluate(args.dataset, args.split))


if __name__ == "__main__":
    main()
'''


def _pipeline_optimize_py(name: str) -> str:
    return f'''"""Pipeline-level optimization for {name}.

Tests different agent configurations, prompt versions, and pipeline
parameters to find the best-performing combination.

Usage::

    python -m pipelines.{name}.optimize
    python -m pipelines.{name}.optimize --strategy grid
    python -m pipelines.{name}.optimize --strategy ablation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load optimization examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


async def run_pipeline_with_config(
    input_data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the pipeline with optional config overrides."""
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/pipelines/{name}/run",
            json={{
                "input": input_data,
                "metadata": metadata or {{}},
            }},
        )
        response.raise_for_status()
        return response.json()


def score_result(result: dict[str, Any]) -> float:
    """Score a pipeline result. Customize for your use case."""
    if result.get("status") != "success":
        return 0.0

    score = 0.0
    steps = result.get("steps", {{}})
    if steps:
        completed = sum(1 for s in steps.values() if s.get("status") == "success")
        score += completed / len(steps) * 0.5  # 50% for step completion

    if result.get("output", {{}}).get("response"):
        score += 0.5  # 50% for having a response

    return score


async def grid_search(dataset_path: str) -> None:
    """Run grid search over pipeline configurations."""
    examples = load_dataset(dataset_path)
    print(f"Grid search on {{len(examples)}} examples\\n")

    # Define your parameter grid here
    param_grid = {{
        "timeout": [60.0, 120.0],
        "on_error": ["fail_fast", "continue"],
        # Add pipeline-specific parameters:
        # "temperature": [0.1, 0.5, 0.8],
        # "system_prompt": ["prompt_v1", "prompt_v2"],
    }}

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"Testing {{len(combos)}} configurations...\\n")

    best_score = -1.0
    best_config: dict[str, Any] = {{}}
    results = []

    for combo in combos:
        config = dict(zip(keys, combo))
        print(f"  Config: {{config}}")

        scores = []
        for example in examples[:5]:  # limit for speed
            input_data = example.get("input", example)
            try:
                result = await run_pipeline_with_config(
                    input_data, metadata={{"config_override": config}},
                )
                scores.append(score_result(result))
            except Exception:
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        results.append({{"config": config, "avg_score": avg}})
        print(f"    → avg score: {{avg:.3f}}\\n")

        if avg > best_score:
            best_score = avg
            best_config = config

    print("=" * 60)
    print(f"\\nBest config (score={{best_score:.3f}}):")
    for k, v in best_config.items():
        print(f"  {{k}}: {{v}}")

    # Save results
    report_path = DATA_DIR / "optimize_report.json"
    report_path.write_text(json.dumps({{
        "pipeline": "{name}",
        "best_config": best_config,
        "best_score": best_score,
        "all_results": results,
    }}, indent=2))
    print(f"\\nReport saved to {{report_path}}")


async def ablation_study(dataset_path: str) -> None:
    """Run ablation study — remove one step at a time to measure impact."""
    examples = load_dataset(dataset_path)
    print(f"Ablation study on {{len(examples)}} examples\\n")

    # First run the full pipeline as baseline
    baseline_scores = []
    for example in examples[:5]:
        input_data = example.get("input", example)
        try:
            result = await run_pipeline_with_config(input_data)
            baseline_scores.append(score_result(result))
        except Exception:
            baseline_scores.append(0.0)

    baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    print(f"  Baseline (all steps): {{baseline:.3f}}")

    # Get pipeline steps
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        config_resp = await client.get(
            "http://localhost:8000/api/v1/pipelines/{name}/config"
        )
        config = config_resp.json()

    steps = config.get("steps", [])
    print(f"  Steps: {{[s.get('name', '?') for s in steps]}}\\n")

    # Test without each step
    for step in steps:
        step_name = step.get("name", "unknown")
        print(f"  Without '{{step_name}}':")
        # Note: actual step skipping requires pipeline support
        # This is a framework for your ablation logic
        print(f"    → would skip step '{{step_name}}' and re-run")

    print(f"\\nBaseline score: {{baseline:.3f}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize {name} pipeline")
    parser.add_argument(
        "--strategy", choices=["grid", "ablation"], default="grid",
        help="Optimization strategy"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATA_DIR / "dataset.jsonl"),
        help="Path to dataset (JSONL)"
    )
    args = parser.parse_args()

    if args.strategy == "grid":
        asyncio.run(grid_search(args.dataset))
    elif args.strategy == "ablation":
        asyncio.run(ablation_study(args.dataset))


if __name__ == "__main__":
    main()
'''


def _pipeline_dataset_jsonl(name: str) -> str:
    return (
        '{"input": {"query": "What is machine learning?"}, '
        '"expected_output": {"response": "Machine learning is..."}, '
        '"split": "test"}\n'
        '{"input": {"query": "Explain neural networks"}, '
        '"expected_output": {"response": "Neural networks are..."}, '
        '"split": "test"}\n'
        '{"input": {"query": "What is NLP?"}, '
        '"expected_output": {"response": "NLP stands for..."}, '
        '"split": "train"}\n'
    )


def _pipeline_run_py(name: str) -> str:
    return f'''"""Run the {name} pipeline locally (without the platform server).

Usage::

    python -m pipelines.{name}.run "What is machine learning?"
    python -m pipelines.{name}.run --input-file input.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


async def run_via_api(query: str) -> dict[str, Any]:
    """Execute the pipeline via the platform REST API."""
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/pipelines/{name}/run",
            json={{"input": {{"query": query}}}},
        )
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run {name} pipeline")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="JSON file with input data"
    )
    args = parser.parse_args()

    if args.input_file:
        input_data = json.loads(Path(args.input_file).read_text())
        query = input_data.get("query", json.dumps(input_data))
    elif args.query:
        query = args.query
    else:
        print("Usage: python -m pipelines.{name}.run \\"your query\\"")
        sys.exit(1)

    print(f"Running pipeline '{name}'...")
    print(f"  Query: {{query}}\\n")

    result = asyncio.run(run_via_api(query))

    print(f"  Status: {{result.get('status', 'unknown')}}")
    print(f"  Duration: {{result.get('duration_ms', 0):.0f}}ms")
    print(f"\\n  Output:")
    print(json.dumps(result.get("output", {{}}), indent=4))

    if result.get("steps"):
        print(f"\\n  Steps:")
        for step_name, step_data in result["steps"].items():
            status = step_data.get("status", "?")
            dur = step_data.get("duration_ms", 0)
            print(f"    {{step_name}}: {{status}} ({{dur:.0f}}ms)")


if __name__ == "__main__":
    main()
'''


def _pipeline_makefile(name: str) -> str:
    return f""".PHONY: run validate eval optimize help

help:  ## Show this help
\t@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \\
\t\tawk 'BEGIN {{FS = ":.*?## "}}; {{printf "\\033[36m%-15s\\033[0m %s\\n", $$1, $$2}}'

run:  ## Run the pipeline via API
\tpython -m pipelines.{name}.run "Hello world"

validate:  ## Validate the pipeline
\tcurl -s http://localhost:8000/api/v1/pipelines/{name}/validate | python -m json.tool

eval:  ## Evaluate pipeline quality
\tpython -m pipelines.{name}.eval

optimize:  ## Optimize pipeline configuration
\tpython -m pipelines.{name}.optimize --strategy grid

visualize:  ## Get pipeline Mermaid diagram
\tcurl -s http://localhost:8000/api/v1/pipelines/{name}/visualize | python -m json.tool

all: validate eval optimize  ## Full lifecycle: validate → eval → optimize
"""


# =====================================================================
# Template Registry
# =====================================================================

TEMPLATES: dict[str, str] = {
    "basic": "Minimal class-based agent (recommended) — 1 file, quick start",
    "class": "Alias for basic — class-owned BaseGraphAgent",
    "full": "All files — class agent with config, schemas, tools, dataset, train/eval scripts",
    "coordinator": "Orchestrator — classify & route queries to specialist agents via delegation",
    "pipeline": "Pipeline — multi-step YAML workflow chaining multiple agents",
    "rag": "RAG class-based agent — retrieve → generate pipeline",
    "chatbot": "Conversational class-based agent with memory",
    "deepagent": "Deep Agent — planning, tools, subagents (requires deepagents package)",
    "custom": "Framework-agnostic — no LangGraph dependency",
    "legacy_dict": "Legacy functional agent — 3 files (__init__, graph, nodes)",
    "plugin": "ML Model Plugin — wrap classical ML models with auto-generated REST endpoints",
    "endpoint": "Custom Endpoint — call deployed model services (httpx + auth) and aggregate",
    "connection": "Connections — per-agent authenticated database + HTTP service connections",
    "ingestion": "Ingestor — package your own doc-ingestion code (any library) as a task-run job",
    "extraction": (
        "Extraction agent — scope-parameterized markdown extractor for parallel "
        "fan-out via map pipeline steps"
    ),
}


# --- ML Plugin template ---


def _plugin_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""ML Model Plugin: {name}."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from agentomatic.plugins import BaseMLPlugin

class {title}Input(BaseModel):
    """Input schema for {name}."""
    text: str = Field(..., description="Input text to process")

class {title}Output(BaseModel):
    """Output schema for {name}."""
    result: str = Field(..., description="Processing result")
    confidence: float = Field(0.0, description="Model confidence score")

class {title}Plugin(BaseMLPlugin[{title}Input, {title}Output]):
    """Classical ML model wrapper for {name}."""

    async def load_model(self) -> None:
        """Load the ML model weights into memory.
        This is called automatically during platform startup.
        """
        # TODO: Load your model here (e.g., joblib.load, torch.load)
        self.model = "dummy_model_instance"

    async def predict(self, inputs: {title}Input) -> {title}Output:
        """Run inference using the loaded model."""
        # TODO: Run your actual model prediction here
        return {title}Output(
            result=f"Processed: {{inputs.text}}",
            confidence=0.95
        )

    def model_card(self) -> dict[str, Any]:
        """Return metadata about the model."""
        return {{
            "name": "{name}",
            "version": "1.0.0",
            "framework": "custom"
        }}
'''


def _plugin_readme(name: str) -> str:
    title = name.replace("_", " ").title()
    return f"""# {title} Plugin\n\nGenerated with `agentomatic init {name} --template plugin`.\n\n## Quick Start\n\nPlace this folder inside your `plugins/` directory. When you run `agentomatic run --plugins-dir plugins`, the platform will automatically discover this plugin and mount its REST endpoints.\n\n## Endpoints\n\n- `POST /api/v1/plugins/{name}/predict`\n- `GET /api/v1/plugins/{name}/health`\n- `GET /api/v1/plugins/{name}/model_card`\n"""


def _plugin_dataset_jsonl(name: str) -> str:
    return (
        '{"text": "Sample positive input", "label": 1}\n'
        '{"text": "Sample negative input", "label": 0}\n'
    )


def _plugin_train_py(name: str) -> str:
    return f'''"""Training script for {name} ML plugin.

Demonstrates a standard classical ML training loop with a JSONL dataset,
train / eval split, and pluggable weighted metrics — swap the ``fit`` /
``predict`` stubs for scikit-learn / PyTorch / etc.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
DATASET = DATA_DIR / "dataset.jsonl"
MODEL_PATH = DATA_DIR / "model_weights.pkl"

# Weighted evaluation config — mirror this in eval.py + optimize.py.
# Weights should reflect what actually matters for your task.
METRIC_WEIGHTS = {{
    "accuracy": 0.6,
    "confidence": 0.4,
}}


def load_data(filepath: str | Path) -> list[dict]:
    """Load a JSONL dataset from *filepath*."""
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]


def train_eval_split(data: list[dict], eval_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Deterministic tail-split (last N% held out for eval)."""
    if not data:
        return data, []
    n_eval = max(1, int(len(data) * eval_fraction))
    return data[:-n_eval], data[-n_eval:]


def train() -> None:
    """Train the {name} plugin and persist model weights."""
    logger.info("Training {name} plugin...")

    data = load_data(DATASET)
    train_data, eval_data = train_eval_split(data)
    logger.info(
        "Loaded %d examples (train=%d, eval=%d).",
        len(data), len(train_data), len(eval_data),
    )

    # Placeholder — replace with your real classical-ML train + save path.
    # from sklearn.linear_model import LogisticRegression
    # import joblib
    # X = [d["text"] for d in train_data]
    # y = [d["label"] for d in train_data]
    # model = LogisticRegression().fit(X, y)
    # joblib.dump(model, MODEL_PATH)

    logger.error(
        "Plugin train.py is a stub — implement sklearn/joblib training and save "
        "to %s before relying on `make train`.",
        MODEL_PATH,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    train()
'''


def _plugin_eval_py(name: str) -> str:
    title = name.replace("_", "").title()
    return f'''"""Evaluation script for {name} ML plugin.

Loads the plugin, runs it against the labelled dataset, and computes a
weighted score across multiple criteria (accuracy + confidence).

The accuracy is computed *honestly*: a prediction only counts as correct
when :func:`to_label` maps the plugin output to the same class as the
example's ground-truth ``label``. If the plugin still returns the stub
response (no derivable label), eval exits non-zero instead of reporting a
fabricated score — swap :func:`to_label` and ``predict`` for your real
model, then add F1 / ROC / etc.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .plugin import {title}Input, {title}Plugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
DATASET = DATA_DIR / "dataset.jsonl"

# Keep aligned with train.py / optimize.py.
METRIC_WEIGHTS = {{
    "accuracy": 0.6,
    "confidence": 0.4,
}}


def load_data(filepath: Path) -> list[dict]:
    """Load a JSONL dataset from *filepath*."""
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_label(result: str) -> int | None:
    """Map a raw plugin output string to a class label.

    Adapt this to your label space. Returns ``None`` when the output cannot
    be interpreted as a label so eval never counts an unparseable prediction
    as correct (which would inflate accuracy).
    """
    text = str(result).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return None


def weighted_score(component_scores: dict[str, float], weights: dict[str, float]) -> float:
    """Return a weight-normalised composite score across components."""
    total_w = sum(weights.get(k, 0.0) for k in component_scores) or 1.0
    return sum(component_scores.get(k, 0.0) * weights.get(k, 0.0) for k in component_scores) / total_w


async def evaluate() -> None:
    """Load the plugin, run predictions, and report honest weighted metrics."""
    logger.info("Evaluating {name} plugin...")
    plugin = {title}Plugin()
    await plugin.load_model()

    if not DATASET.exists():
        logger.error("No dataset at %s — add labelled JSONL rows before evaluating.", DATASET)
        raise SystemExit(1)

    examples = load_data(DATASET)
    labelled = [ex for ex in examples if "label" in ex]
    if not labelled:
        logger.error(
            "Dataset has no `label` field — cannot compute accuracy. Add "
            'ground-truth labels (e.g. {{"text": ..., "label": 1}}) first.'
        )
        raise SystemExit(1)

    correct = 0
    scored = 0
    confidence_sum = 0.0
    for ex in labelled:
        result = await plugin.predict({title}Input(text=ex["text"]))
        predicted = to_label(result.result)
        if predicted is not None:
            scored += 1
            if predicted == ex["label"]:
                correct += 1
        confidence_sum += float(getattr(result, "confidence", 0.0))

    if scored == 0:
        logger.error(
            "Could not derive a single prediction/label from plugin output — the "
            "plugin still returns the stub response. Implement `predict()` and "
            "`to_label()` so eval compares real predictions to ground truth "
            "instead of reporting a fabricated score."
        )
        raise SystemExit(1)

    scores = {{
        "accuracy": correct / scored,
        "confidence": confidence_sum / len(labelled),
    }}
    composite = weighted_score(scores, METRIC_WEIGHTS)

    logger.info("Scored %d/%d labelled examples.", scored, len(labelled))
    logger.info("Component scores: %s", scores)
    logger.info("Weighted composite score: %.3f", composite)


if __name__ == "__main__":
    asyncio.run(evaluate())
'''


def _plugin_optimize_py(name: str) -> str:
    return f'''"""Hyperparameter optimization script for {name} plugin."""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize() -> None:
    """Optimize hyperparameters (stub — exit until you wire a real search)."""
    logger.error(
        "Plugin optimize.py is a stub. Implement GridSearchCV / Optuna / "
        "random search (then retrain + save), or call `agentomatic optimize` "
        "for prompt-level tuning of class agents."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    optimize()
'''


def _plugin_predict_py(name: str) -> str:
    title = name.replace("_", "").title()
    return f'''"""Local inference script for {name} plugin."""
import asyncio
from .plugin import {title}Plugin, {title}Input

async def predict(text: str):
    plugin = {title}Plugin()
    await plugin.load_model()
    result = await plugin.predict({title}Input(text=text))
    print(f"Result: {{result.result}}")
    print(f"Confidence: {{result.confidence}}")

if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "Default input text"
    asyncio.run(predict(text))
'''


def _plugin_makefile(name: str) -> str:
    return f"""# Makefile for {name} plugin
.PHONY: train eval optimize predict clean

train:
	python -m plugins.{name}.train

eval:
	python -m plugins.{name}.eval

optimize:
	python -m plugins.{name}.optimize

predict:
	python -m plugins.{name}.predict "Test input"

clean:
	rm -rf __pycache__ *.pkl *.pt
"""


def _dataset_jsonl(name: str) -> str:
    return (
        f'{{"id": "{name}_001", "split": "train", "input": {{"current_query": '
        f'"Help me with task planning"}}, "expected_output": {{"response": '
        f'"Here is a structured plan..."}}, "metadata": {{"domain": "general", '
        f'"difficulty": "easy"}}}}\n'
        f'{{"id": "{name}_002", "split": "train", "input": {{"current_query": '
        f'"Summarize this document"}}, "expected_output": {{"response": '
        f'"Summary: ..."}}, "metadata": {{"domain": "general", '
        f'"difficulty": "medium"}}}}\n'
        f'{{"id": "{name}_003", "split": "train", "input": {{"current_query": '
        f'"Compare option A vs B"}}, "expected_output": {{"response": '
        f'"Comparison: A is..."}}, "metadata": {{"domain": "general", '
        f'"difficulty": "medium"}}}}\n'
        f'{{"id": "{name}_004", "split": "train", "input": {{"current_query": '
        f'"Write a brief report"}}, "expected_output": {{"response": '
        f'"Report: ..."}}, "metadata": {{"domain": "general", '
        f'"difficulty": "easy"}}}}\n'
        f'{{"id": "{name}_005", "split": "test", "input": {{"current_query": '
        f'"Analyze the risks"}}, "expected_output": {{"response": '
        f'"Risks identified: ..."}}, "metadata": {{"domain": "general", '
        f'"difficulty": "hard"}}}}\n'
        f'{{"id": "{name}_006", "split": "test", "input": {{"current_query": '
        f'"Explain in simple terms"}}, "expected_output": {{"response": '
        f'"In simple terms: ..."}}, "metadata": {{"domain": "general", '
        f'"difficulty": "easy"}}}}\n'
    )


def _train_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''#!/usr/bin/env python3
"""Fit **{name}** prompts via Agentomatic ``train_and_report``.

Flat script: ``TrainCliSettings`` (env + CLI) → agent → fit → report.
Knobs: ``AGENTOMATIC_*`` env vars and/or ``--help`` flags.

For staged Keras-like control (load → metrics → compile → fit → evaluate),
see the commented block at the bottom of ``main`` and optimization.md.

Usage (from project root)::

    AGENTOMATIC_STACK=local uv run python agents/{name}/train.py
    AGENTOMATIC_STACK=gemini uv run python agents/{name}/train.py \\
        --augment --n-examples 40 --persist --optimizer rewrite
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # → project root (agents/<name>/..)
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentomatic.config.settings import load_environment
from agentomatic.optimize import TrainCliSettings, print_train_result, train_and_report
from agentomatic.providers import apply_stack_defaults, get_llm_for_agent
from agentomatic.stacks.manager import StackManager
from rich.console import Console

from agents.{name}.agent import {title}Agent

AGENT = "{name}"
HERE = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"  # nested layouts (e.g. ai_platform/) may use ROOT.parent
console = Console()


def main(argv: list[str] | None = None) -> int:
    """Fit {name} prompts (settings → agent → train_and_report)."""
    # --- settings (AGENTOMATIC_* env + optional CLI overrides) ---
    cli = TrainCliSettings.parse(argv)

    # --- environment / stack ---
    load_environment(ENV_PATH)
    stacks = StackManager(ROOT / "stacks")
    stacks.load(cli.stack)
    apply_stack_defaults(stacks)

    # --- agent ---
    llm = get_llm_for_agent(AGENT, role="default", stack_manager=stacks)
    agent = {title}Agent(llm=llm)

    # --- fit + HolySheet report (one-shot convenience) ---
    result = train_and_report(
        agent,
        config=cli.to_train_config(
            agent_name=AGENT,
            agent_dir=HERE,
            stacks_dir=ROOT / "stacks",
            env_path=ENV_PATH,
            required_keys=["response"],
            judge_criteria=(
                "Evaluate whether the response is relevant to the query, "
                "accurate, well-structured, and actionable. Provide graded "
                "0–1 scores with clear motivation."
            ),
            judge_dimensions=["relevance", "accuracy", "structure"],
        ),
    )
    print_train_result(result, console=console)

    # --- optional staged Keras-like path (same primitives as train_and_report) ---
    # from agentomatic.optimize import (
    #     build_default_metrics, compile_agent, evaluate_agent, fit_agent,
    #     generate_fit_report, load_data, prepare_dataset,
    # )
    # data, _ = prepare_dataset(
    #     load_data(HERE / "datasets" / "all.jsonl"),
    #     augment=cli.augment, n_examples=cli.n_examples, persist=cli.persist,
    #     seed_path=HERE / "datasets" / "all.jsonl",
    # )
    # entry = stacks.get_llm_config("default")
    # model = f"{{entry.provider}}/{{entry.model}}"
    # metrics, loss, fit_metric = build_default_metrics(
    #     model=model, required_keys=["response"],
    #     judge_criteria="…", judge_dimensions=["relevance", "accuracy", "structure"],
    # )
    # compiled = compile_agent(
    #     agent, dataset=data, metrics=metrics, loss=loss, fit_metric=fit_metric,
    #     optimizer=cli.optimizer, task_model=model, rewrite_model=model,
    #     llm_base_url=entry.base_url, llm_api_key=entry.api_key or "local",
    #     agent_name=AGENT, max_trials=cli.trials, patience=cli.patience,
    # )
    # history = fit_agent(compiled, data, epochs=cli.epochs, trials=cli.trials)
    # scores = evaluate_agent(compiled, data.test or data.validation).scores
    # generate_fit_report(compiled.fit_result, output_path=HERE / "reports" / f"train_{{AGENT}}.html",
    #                     keras_history=history.history, eval_scores=scores)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _eval_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''#!/usr/bin/env python3
"""Evaluate **{name}** via Agentomatic ``evaluate_and_report``.

Flat script: ``EvalCliSettings`` (env + CLI) → agent → evaluate → report.
Knobs: ``AGENTOMATIC_*`` env vars and/or ``--help`` flags. Mirrors ``train.py``.

Usage (from project root)::

    AGENTOMATIC_STACK=local uv run python agents/{name}/eval.py
    AGENTOMATIC_STACK=gemini uv run python agents/{name}/eval.py \\
        --split test --prefer-augmented --limit 3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # → project root (agents/<name>/..)
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentomatic.config.settings import load_environment
from agentomatic.optimize import EvalCliSettings, evaluate_and_report, print_eval_result
from agentomatic.providers import apply_stack_defaults, get_llm_for_agent
from agentomatic.stacks.manager import StackManager
from rich.console import Console

from agents.{name}.agent import {title}Agent

AGENT = "{name}"
HERE = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"  # nested layouts (e.g. ai_platform/) may use ROOT.parent
console = Console()


def main(argv: list[str] | None = None) -> int:
    """Evaluate {name} (settings → agent → evaluate_and_report)."""
    # --- settings (AGENTOMATIC_* env + optional CLI overrides) ---
    cli = EvalCliSettings.parse(argv)

    # --- environment / stack ---
    load_environment(ENV_PATH)
    stacks = StackManager(ROOT / "stacks")
    stacks.load(cli.stack)
    apply_stack_defaults(stacks)

    # --- agent ---
    llm = None if not cli.judge else get_llm_for_agent(AGENT, role="default", stack_manager=stacks)
    agent = {title}Agent(llm=llm)
    if cli.compiled:
        agent.load_compiled(cli.compiled)
        console.print(f"Loaded compiled config from {{cli.compiled}}")

    # --- evaluate + HolySheet report ---
    result = evaluate_and_report(
        agent,
        config=cli.to_eval_config(
            agent_name=AGENT,
            agent_dir=HERE,
            stacks_dir=ROOT / "stacks",
            env_path=ENV_PATH,
            required_keys=["response"],
            judge_criteria=(
                "Evaluate whether the response is relevant to the query, "
                "accurate, well-structured, and actionable. Provide graded "
                "0–1 scores with clear motivation."
            ),
            judge_dimensions=["relevance", "accuracy", "structure"],
        ),
    )
    print_eval_result(result, agent_name=AGENT, console=console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _optimize_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Prompt / parameter optimization script for {name}.

Two families of optimization are supported:

- **prompt_only**: rewrite the system prompt only. Fast, low-risk, uses
  the built-in ``GridSearchOptimizer`` on a curated list of prompt
  variants defined below.
- **param_search**: PromptFitter-based parameter search over the space
  declared in ``search_space.yaml``. Also supports ``rewrite``,
  ``gepa_like``, ``mipro_like``, ``few_shot`` fitter modes.

Usage::

    # Prompt-only (fast, no external LLM required)
    python -m agents.{name}.optimize --strategy prompt_only

    # Parameter search using the local search_space.yaml grid
    python -m agents.{name}.optimize --strategy param_search

    # Full PromptFitter with GEPA-style rewrites + params
    python -m agents.{name}.optimize --strategy gepa_like \\
        --search-space agents/{name}/search_space.yaml

Equivalent CLI (skips this script entirely)::

    agentomatic optimize {name} \\
        --dataset agents/{name}/dataset.jsonl \\
        --mode param_search \\
        --search-space agents/{name}/search_space.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import {title}Agent

from agentomatic.agents import AgentDataset
from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
    WeightedMetric,
)
from agentomatic.agents.optimizers import GridSearchOptimizer, PromptFitterBridge

DATA_DIR = Path(__file__).parent
COMPILED_DIR = Path("compiled") / "{name}"
DEFAULT_SEARCH_SPACE = DATA_DIR / "search_space.yaml"


# Reuse the same weighted metric config as train.py / eval.py.
METRICS = [
    ("exact_response", ExactKeyMatchMetric(["response"]), 0.5),
    ("contains_terms", ContainsTermsMetric(["Result"]), 0.3),
    (
        "has_output",
        CallableMetric(
            "has_output",
            lambda example, pred: 1.0 if pred.get("response") else 0.0,
        ),
        0.2,
    ),
]


def build_metrics() -> list:
    """Return metrics used by all optimization strategies."""
    individual = [m for _, m, _ in METRICS]
    composite = WeightedMetric(METRICS, name="composite")
    return [*individual, composite]


def run_prompt_only(agent: {title}Agent, dataset: AgentDataset) -> None:
    """Optimise the system prompt only via GridSearch over prompt variants."""
    optimizer = GridSearchOptimizer(
        param_grid={{
            "system_prompt": [
                "You are a helpful assistant.",
                "You are a precise, detail-oriented assistant.",
                "You are a concise assistant. Be brief and accurate.",
            ],
        }},
        max_examples=10,  # limit examples per combo for speed
    )

    print("Running prompt-only optimization (GridSearch on prompts)...")
    print(f"  Training examples: {{len(dataset.train)}}")
    agent.compile(dataset, build_metrics(), optimizer=optimizer)
    agent.fit(dataset)

    print("\\nOptimized config:")
    for key, value in agent.compiled_config.items():
        print(f"  {{key}}: {{value}}")

    report = agent.evaluate(dataset.test, build_metrics())
    print(f"\\nTest pass rate: {{report.pass_rate:.1%}}")
    print(report.summary())

    agent.save(str(COMPILED_DIR))
    print(f"\\nSaved optimized agent to {{COMPILED_DIR}}/")


def run_param_search(
    agent: {title}Agent,
    dataset: AgentDataset,
    *,
    search_space_path: Path | None = None,
    mode: str = "param_search",
) -> None:
    """Optimize prompt + parameters via PromptFitter and the search space.

    Loads a ``PromptSearchSpace`` from ``search_space.yaml`` (or the
    ``--search-space`` argument) and dispatches to the requested fitter
    mode (``param_search``, ``rewrite``, ``gepa_like``, ``mipro_like``,
    ``few_shot``).
    """
    try:
        from agentomatic.optimize import load_search_space  # type: ignore
    except ImportError:
        print(
            "agentomatic[optimize] is required for param_search. "
            "Install with: pip install 'agentomatic[optimize]'"
        )
        return

    ss_path = Path(search_space_path or DEFAULT_SEARCH_SPACE)
    if ss_path.exists():
        space = load_search_space(ss_path)
        print(f"Loaded search space from {{ss_path}}")
    else:
        from agentomatic.optimize import PromptSearchSpace  # type: ignore
        space = PromptSearchSpace()
        print(f"No search_space.yaml at {{ss_path}} — using defaults")

    print(f"Active spaces: {{space.active_spaces()}}")
    print(f"Total combinations: {{space.total_search_size()}}")

    optimizer = PromptFitterBridge(
        agent_name="{name}",
        task_model=os.environ.get(
            "AGENTOMATIC_TASK_MODEL",
            os.environ.get("LLM__MODEL", "ollama/qwen2.5:7b"),
        ),
        rewrite_model=os.environ.get(
            "AGENTOMATIC_REWRITE_MODEL",
            os.environ.get("REWRITE_LLM__MODEL", "openai/gpt-4.1"),
        ),
        optimizer=mode,
        search_space=space,
    )

    print(f"Running PromptFitter (mode={{mode}})...")
    agent.compile(dataset, build_metrics(), optimizer=optimizer)
    agent.fit(dataset)

    print("\\nOptimized config:")
    for key, value in agent.compiled_config.items():
        print(f"  {{key}}: {{value}}")

    agent.save(str(COMPILED_DIR))
    print(f"\\nSaved optimized agent to {{COMPILED_DIR}}/")


def main() -> None:
    """Run optimization."""
    parser = argparse.ArgumentParser(description="Optimize {name}")
    parser.add_argument(
        "--strategy",
        choices=["prompt_only", "param_search", "rewrite", "gepa_like",
                 "mipro_like", "few_shot"],
        default="prompt_only",
        help=(
            "Optimization mode. 'prompt_only' rewrites prompts only. "
            "The rest go through PromptFitter with a search space."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATA_DIR / "dataset.jsonl"),
        help="Path to training dataset (JSONL)"
    )
    parser.add_argument(
        "--search-space", type=str, default=None,
        help="Path to a search_space.yaml (defaults to agents/{name}/search_space.yaml)"
    )
    args = parser.parse_args()

    agent = {title}Agent(llm=None)
    dataset = AgentDataset.from_jsonl(args.dataset)

    if args.strategy == "prompt_only":
        run_prompt_only(agent, dataset)
    else:
        run_param_search(
            agent,
            dataset,
            search_space_path=Path(args.search_space) if args.search_space else None,
            mode=args.strategy,
        )


if __name__ == "__main__":
    main()
'''


def _predict_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Batch prediction script for {name}.

Runs the agent on a list of inputs and saves results.

Usage::

    python -m agents.{name}.predict "What is X?"
    python -m agents.{name}.predict --input queries.jsonl --output results.jsonl
    python -m agents.{name}.predict --compiled compiled/{name} "Hello"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import {title}Agent

DATA_DIR = Path(__file__).parent


def main() -> None:
    """Run batch or single prediction."""
    parser = argparse.ArgumentParser(description="Run {name} predictions")
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Single query to run (interactive mode)"
    )
    parser.add_argument(
        "--compiled", type=str, default=None,
        help="Path to compiled agent directory"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="JSONL file with queries (one per line)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file for results"
    )
    args = parser.parse_args()

    # Create agent
    agent = {title}Agent(llm=None)
    if args.compiled:
        agent.load_compiled(args.compiled)
        print(f"Loaded compiled config from {{args.compiled}}")

    # Single query mode
    if args.query:
        result = agent.transform({{"current_query": args.query}})
        print(json.dumps(result, indent=2, default=str))
        return

    # Batch mode
    if args.input:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else DATA_DIR / "predictions.jsonl"

        queries = []
        with open(input_path) as f:
            for line in f:
                queries.append(json.loads(line.strip()))

        print(f"Processing {{len(queries)}} queries...")
        results = []
        for i, query_data in enumerate(queries, 1):
            try:
                result = agent.transform(query_data)
                results.append({{"input": query_data, "output": result, "status": "ok"}})
            except Exception as exc:
                results.append({{"input": query_data, "error": str(exc), "status": "error"}})
            print(f"  [{{i}}/{{len(queries)}}] done")

        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\\n")

        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"\\nCompleted: {{ok}}/{{len(results)}} successful")
        print(f"Results saved to {{output_path}}")
        return

    # Interactive mode (no args)
    print(f"{{agent.agent_name}} — interactive mode (Ctrl+C to exit)")
    print()
    while True:
        try:
            query = input("Query> ").strip()
            if not query:
                continue
            result = agent.transform({{"current_query": query}})
            print(json.dumps(result, indent=2, default=str))
            print()
        except (KeyboardInterrupt, EOFError):
            print("\\nBye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
'''


def _search_space_yaml(name: str) -> str:
    """Return the default PromptSearchSpace YAML for the ``full`` template.

    Keys mirror :meth:`agentomatic.optimize.PromptSearchSpace.to_dict` so
    the file loads cleanly via :func:`agentomatic.optimize.load_search_space`.
    """
    return f"""# Default search space for {name}
#
# Loaded by `agentomatic optimize {name} --mode param_search --search-space
# agents/{name}/search_space.yaml` or by `agents.{name}.optimize` via the
# `PromptFitterBridge`.  All keys are optional — omit any you don't need.

# Toggle which knobs are considered by PromptFitter.
optimize_system_prompt: true
optimize_user_template: false
optimize_few_shot: true
optimize_model_params: true
optimize_rag_params: false
optimize_tool_params: false
optimize_model_choice: false

# Model hyper-parameter grid (candidates per parameter).
model_param_space:
  temperature: [0.0, 0.2, 0.5, 0.7]
  top_p: [0.9, 1.0]
  max_tokens: [800, 1200, 2000]

# RAG parameter grid — populate if `optimize_rag_params` is enabled.
rag_param_space: {{}}

# Tool parameter grid — populate if `optimize_tool_params` is enabled.
tool_param_space: {{}}

# Alternative model choices for `optimize_model_choice: true`.
model_choices: []
fallback_models: []
routing_weight_space: {{}}

# Few-shot config.
max_few_shot_examples: 5
few_shot_selection_strategy: diversity_weighted   # top_k | diversity_weighted | random_search
"""


def _makefile(name: str) -> str:
    return f"""# Makefile for {name} agent — ML lifecycle commands
#
# Usage:
#   make train              — Compile, fit, and save the agent
#   make eval               — Evaluate on the test split
#   make optimize           — Prompt-only optimization (fast, no external LLM)
#   make optimize-params    — Full parameter search via PromptFitter
#   make optimize-gepa      — GEPA-style feedback-guided prompt mutations
#   make predict            — Start interactive prediction mode
#   make all                — Full pipeline: train → eval

.PHONY: train eval optimize optimize-params optimize-gepa predict all clean

AGENT = {name}
COMPILED = compiled/$(AGENT)

train:
\t@echo "\\n🏋️  Training $(AGENT)..."
\tpython -m agents.$(AGENT).train

eval:
\t@echo "\\n📊 Evaluating $(AGENT)..."
\tpython -m agents.$(AGENT).eval --split test

eval-all:
\t@echo "\\n📊 Evaluating $(AGENT) on all splits..."
\tpython -m agents.$(AGENT).eval --split all

optimize:
\t@echo "\\n🔧 Optimizing $(AGENT) — prompt only..."
\tpython -m agents.$(AGENT).optimize --strategy prompt_only

optimize-params:
\t@echo "\\n🔧 Optimizing $(AGENT) — param search via search_space.yaml..."
\tpython -m agents.$(AGENT).optimize --strategy param_search

optimize-gepa:
\t@echo "\\n🔧 Optimizing $(AGENT) — GEPA-style mutations..."
\tpython -m agents.$(AGENT).optimize --strategy gepa_like

predict:
\t@echo "\\n🔮 Starting interactive prediction..."
\tpython -m agents.$(AGENT).predict

all: train eval
\t@echo "\\n✅ Pipeline complete!"

clean:
\t@echo "Cleaning compiled artifacts..."
\trm -rf $(COMPILED)
"""


# --- Custom Endpoint template ---


def _endpoint_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Custom Endpoint: {name}.

A custom endpoint calls one or more deployed model services via
authenticated ``httpx`` requests and aggregates their responses. It is
auto-discovered by the platform and also usable as a pipeline step.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agentomatic.endpoints import (
    AggregationStrategy,
    AuthType,
    BaseEndpoint,
    UpstreamAuthConfig,
    UpstreamConfig,
)


class {title}Request(BaseModel):
    """Input schema for {name}."""

    payload: dict = Field(default_factory=dict, description="Data forwarded to upstreams.")


class {title}Endpoint(BaseEndpoint):
    """Fan out to deployed model services and aggregate the results."""

    endpoint_name = "{name}"
    endpoint_description = "Aggregate predictions from deployed model services."
    endpoint_version = "1.0.0"

    #: Route path (mounted under /api/v1/endpoints/{name}).
    path = "/call"
    methods = ["POST"]

    #: How to combine responses (ALL | FIRST_SUCCESS | MAJORITY).
    aggregation = AggregationStrategy.ALL

    #: Deployed model services this endpoint calls. Secrets use ${{ENV}}.
    upstreams = [
        UpstreamConfig(
            name="model_a",
            base_url="${{MODEL_A_URL}}",
            path="/v1/predict",
            method="POST",
            auth=UpstreamAuthConfig(
                type=AuthType.BEARER,
                api_key="${{MODEL_A_TOKEN}}",
            ),
        ),
        # Add more upstreams here. For OAuth2 client-credentials:
        # UpstreamConfig(
        #     name="model_b",
        #     base_url="${{MODEL_B_URL}}",
        #     path="/v1/predict",
        #     auth=UpstreamAuthConfig(
        #         type=AuthType.OAUTH2_CLIENT_CREDENTIALS,
        #         token_url="${{MODEL_B_TOKEN_URL}}",
        #         client_id="${{MODEL_B_CLIENT_ID}}",
        #         client_secret="${{MODEL_B_CLIENT_SECRET}}",
        #     ),
        # ),
    ]

    # The default ``handle`` fans out to every upstream and aggregates.
    # Override it for fully custom behaviour:
    #
    # async def handle(self, request):
    #     ok, aggregated, results = await self.models.fan_out(request.payload)
    #     return {{"ok": ok, "aggregated": aggregated}}
'''


def _endpoint_readme(name: str) -> str:
    title = name.replace("_", " ").title()
    return (
        f"# {title} Endpoint\n\n"
        f"Generated with `agentomatic init {name} --template endpoint`.\n\n"
        "## Quick Start\n\n"
        "Place this folder inside your `endpoints/` directory and run "
        "`agentomatic run` (or pass `endpoints_dir=` to `AgentPlatform`). The "
        "platform auto-discovers this endpoint and mounts its routes.\n\n"
        "## Endpoints\n\n"
        f"- `POST /api/v1/endpoints/{name}/call` — fan out to upstreams\n"
        f"- `GET  /api/v1/endpoints/{name}/health` — readiness\n"
        f"- `GET  /api/v1/endpoints/{name}/info` — metadata\n\n"
        "## Use in a pipeline\n\n"
        "```yaml\n"
        "steps:\n"
        f"  - endpoint: {name}\n"
        "    name: fetch_predictions\n"
        "    input:\n"
        "      payload: $input\n"
        "```\n\n"
        "## Configuration\n\n"
        "Set the referenced environment variables (see `.env.example`). All "
        "string fields support `${ENV}` interpolation so secrets never live in "
        "code.\n"
    )


def _endpoint_env_example(name: str) -> str:
    return (
        "# Deployed model service endpoints + credentials.\n"
        "MODEL_A_URL=https://model-a.internal.example.com\n"
        "MODEL_A_TOKEN=replace-me\n"
        "\n"
        "# Example OAuth2 client-credentials upstream:\n"
        "# MODEL_B_URL=https://model-b.internal.example.com\n"
        "# MODEL_B_TOKEN_URL=https://auth.example.com/oauth/token\n"
        "# MODEL_B_CLIENT_ID=replace-me\n"
        "# MODEL_B_CLIENT_SECRET=replace-me\n"
    )


# --- Connections template ---


def _connections_py(name: str) -> str:
    upper = name.upper()
    return f'''"""Per-agent connections for {name}.

Declare authenticated databases and HTTP services this agent needs. The
platform discovers this ``connections.py`` and registers each connection
under the agent's scope, initialising them on startup.

Access them at runtime with minimal code::

    from agentomatic.connections import get_connections

    conns = get_connections("{name}")
    async with conns.database("main").session() as session:
        ...
    result = await conns.http("scoring_api").post("/score", payload={{"x": 1}})
"""
from __future__ import annotations

from agentomatic.connections import (
    ConnectionPurpose,
    CustomConnectionConfig,
    DatabaseConnectionConfig,
    HttpConnectionConfig,
    VectorConnectionConfig,
)
from agentomatic.endpoints import AuthType, UpstreamAuthConfig

#: Discovered automatically by the platform. Any string field supports
#: ${{ENV}} interpolation so credentials stay out of source control.
#: Tag each connection with a ``purpose`` (memory, rag, vector, cache…) so
#: features can look it up by intent via ``get_connections(...).by_purpose()``.
CONNECTIONS = [
    DatabaseConnectionConfig(
        name="main",
        url="${{{upper}_DB_URL}}",
        # Optionally splice credentials into the URL at connect time:
        username="${{{upper}_DB_USER}}",
        password="${{{upper}_DB_PASSWORD}}",
        pool_size=5,
    ),
    # Conversation memory backed by this agent's own database. Build a store
    # with ``await conns.database("memory").create_store()``.
    DatabaseConnectionConfig(
        name="memory",
        url="${{{upper}_MEMORY_DB_URL}}",
        purpose=ConnectionPurpose.MEMORY,
    ),
    # Vector store for RAG / semantic search. Provider clients are lazy and
    # optional: install the one you use (e.g. ``pip install qdrant-client``).
    VectorConnectionConfig(
        name="kb",
        provider="qdrant",
        url="${{{upper}_QDRANT_URL}}",
        api_key="${{{upper}_QDRANT_API_KEY}}",
        collection="knowledge_base",
        purpose=ConnectionPurpose.RAG,
    ),
    HttpConnectionConfig(
        name="scoring_api",
        base_url="${{{upper}_SCORING_URL}}",
        auth=UpstreamAuthConfig(
            type=AuthType.OAUTH2_CLIENT_CREDENTIALS,
            token_url="${{{upper}_TOKEN_URL}}",
            client_id="${{{upper}_CLIENT_ID}}",
            client_secret="${{{upper}_CLIENT_SECRET}}",
        ),
    ),
    # Any other backend (redis, mongo, elasticsearch…) with zero new classes:
    # point ``factory`` at a callable or dotted path; ${{ENV}} is resolved and
    # the client lifecycle is managed for you. Fetch it with
    # ``await get_connections("{name}").client("cache")``.
    CustomConnectionConfig(
        name="cache",
        factory="redis.asyncio.from_url",
        args=["${{{upper}_REDIS_URL}}"],
        purpose=ConnectionPurpose.CACHE,
    ),
]
'''


def _connections_readme(name: str) -> str:
    title = name.replace("_", " ").title()
    return (
        f"# {title} Connections\n\n"
        f"Generated with `agentomatic init {name} --template connection`.\n\n"
        "This scaffolds a `connections.py` for an agent named "
        f"`{name}`. Drop the `connections.py` into your agent's package "
        "(`agents/{name}/connections.py`) so it is discovered alongside the "
        "agent, or keep it here as a reference.\n\n"
        "## Usage\n\n"
        "```python\n"
        "from agentomatic.connections import get_connections\n\n"
        f'conns = get_connections("{name}")\n'
        'async with conns.database("main").session() as session:\n'
        "    ...\n"
        'result = await conns.http("scoring_api").post("/score", payload={"x": 1})\n'
        "```\n\n"
        "## Configuration\n\n"
        "Set the environment variables in `.env.example`. The database URL is a "
        "SQLAlchemy async URL (e.g. `postgresql+asyncpg://host/db`).\n"
    )


def _connections_env_example(name: str) -> str:
    upper = name.upper()
    return (
        f"# Database (SQLAlchemy async URL, e.g. postgresql+asyncpg://host/db)\n"
        f"{upper}_DB_URL=postgresql+asyncpg://localhost:5432/{name}\n"
        f"{upper}_DB_USER=replace-me\n"
        f"{upper}_DB_PASSWORD=replace-me\n"
        "\n"
        f"# Conversation memory database\n"
        f"{upper}_MEMORY_DB_URL=postgresql+asyncpg://localhost:5432/{name}_memory\n"
        "\n"
        f"# Vector store for RAG / vector search (Qdrant by default)\n"
        f"{upper}_QDRANT_URL=http://localhost:6333\n"
        f"{upper}_QDRANT_API_KEY=replace-me\n"
        "\n"
        f"# Generic factory-based backend (redis cache by default)\n"
        f"{upper}_REDIS_URL=redis://localhost:6379/0\n"
        "\n"
        f"# HTTP scoring service (OAuth2 client-credentials)\n"
        f"{upper}_SCORING_URL=https://scoring.example.com\n"
        f"{upper}_TOKEN_URL=https://auth.example.com/oauth/token\n"
        f"{upper}_CLIENT_ID=replace-me\n"
        f"{upper}_CLIENT_SECRET=replace-me\n"
    )


def _ingestor_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Ingestor: {name}.

Agentomatic packages your ingestion code as a first-class, deployable resource.
You bring the implementation using *any* libraries you like (docling,
unstructured, pymupdf4llm, langchain-text-splitters, your vector DB client, …);
Agentomatic provides discovery, REST endpoints, task/queue execution with live
progress + cancellation, and status reporting.

Place this folder inside your ``ingestion/`` directory. On ``agentomatic run``
it is auto-discovered and mounted at:

- ``POST /api/v1/ingestion/{name}/run``        (synchronous)
- ``POST /api/v1/ingestion/{name}/run/async``  (background task -> task id)
- ``GET  /api/v1/ingestion/{name}/info``
- ``GET  /api/v1/ingestion/{name}/health``
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agentomatic.ingestion import BaseIngestor, IngestionResult


class {title}Request(BaseModel):
    """Input schema for the {name} ingestor (customise freely)."""

    source: str = Field(..., description="Path / glob / URL / bucket URI to ingest")
    collection: str = Field("default", description="Target collection / index name")


class {title}Ingestor(BaseIngestor[{title}Request]):
    """Ingest documents into a vector store.

    Replace the body of :meth:`ingest` with your real implementation. The
    example below shows the shape — swap in your preferred libraries.
    """

    ingestor_name = "{name}"
    ingestor_description = "Parse documents and upsert them into a vector store."
    ingestor_version = "1.0.0"

    async def setup(self) -> None:
        """Initialise clients/models once at startup (override as needed)."""
        # e.g. self.client = AsyncQdrantClient(url=os.environ["QDRANT_URL"])
        # e.g. self.embedder = get_embeddings("openai")
        self.client = None

    async def ingest(self, request: {title}Request, ctx) -> IngestionResult:
        """Run the ingestion. Bring your own libraries here.

        Example flow (pseudo-code)::

            import pymupdf4llm                       # your PDF -> markdown lib
            from langchain_text_splitters import MarkdownTextSplitter

            markdown = pymupdf4llm.to_markdown(request.source)
            chunks = MarkdownTextSplitter().split_text(markdown)

            upserted = 0
            for i, chunk in enumerate(chunks):
                if ctx.cancelled:                    # cooperative cancellation
                    break
                # vector = self.embedder.embed_query(chunk)
                # await self.client.upsert(request.collection, ...)
                upserted += 1
                await ctx.report(              # live progress for the frontend
                    current=i + 1,
                    total=len(chunks),
                    message=f"upserting chunk {{i + 1}}/{{len(chunks)}}",
                )
            return IngestionResult(
                documents=1,
                chunks=len(chunks),
                upserted=upserted,
                collection=request.collection,
            )
        """
        # --- Placeholder implementation (replace with your own) ---
        await ctx.report(message=f"Ingesting {{request.source}}", current=1, total=1)
        return IngestionResult(
            documents=1,
            chunks=0,
            upserted=0,
            collection=request.collection,
            output={{"note": "Replace ingest() with your real implementation."}},
        )
'''


def _ingestor_readme(name: str) -> str:
    return f"""# {name.replace("_", " ").title()} Ingestor

Generated with `agentomatic init {name} --template ingestion`.

Agentomatic provides the **ops** (discovery, REST, task/queue execution,
progress, status); you provide the **implementation** using any libraries you
like.

## Install this ingestor

Place this folder inside your project's `ingestion/` directory:

```
ingestion/
└── {name}/
    ├── __init__.py
    └── ingestor.py
```

## Run it

```bash
agentomatic run

# Synchronous
curl -X POST http://localhost:8000/api/v1/ingestion/{name}/run \\
  -H 'content-type: application/json' \\
  -d '{{"source": "./docs", "collection": "kb"}}'

# As a background task (returns a pollable task id)
curl -X POST http://localhost:8000/api/v1/ingestion/{name}/run/async \\
  -H 'content-type: application/json' \\
  -d '{{"source": "./docs", "collection": "kb"}}'

# Then poll / stream progress via the unified task API
curl http://localhost:8000/api/v1/tasks/<task_id>
curl -N http://localhost:8000/api/v1/tasks/<task_id>/events
```

## Bring your own libraries

Edit `ingestor.py` and use whatever you prefer — e.g. `pymupdf4llm` / `docling`
/ `unstructured` for parsing, `langchain-text-splitters` for chunking, and your
vector DB client (Qdrant, Chroma, pgvector, …) for upserts. Report progress
with `await ctx.report(...)` and honour `ctx.cancelled` in long loops.
"""


def _extraction_agent_py(name: str) -> str:
    """Return the extraction agent module content for template ``extraction``.

    The scaffold produces a scope-parameterized class agent that reads a
    markdown blob (``markdown`` in state) and extracts a set of fields for a
    given ``scope``.  Designed to be fanned out via a pipeline ``map`` step,
    one agent invocation per scope, with retry/progress/checkpoint support.
    """
    title = name.replace("_", " ").title().replace(" ", "")
    body = '''"""Scope-parameterized extraction agent: {name}.

Extracts a set of structured fields from a markdown document for one *scope*
at a time.  Designed to be fanned out via a pipeline ``map`` step so N scopes
are extracted concurrently::

    steps:
      - ingestion: markdown
        name: to_md
        input:
          source: $.input.source

      - name: extract_all
        map:
          agent: {name}
          items: $.input.scopes         # e.g. ["parties", "dates", "amounts"]
          item_key: scope
          max_concurrency: 4
          retry:
            max_attempts: 3
            backoff: exponential
            base_delay: 1.0
          input:
            markdown: $.steps.to_md.output.path
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentomatic.agents import BaseGraphAgent


@dataclass
class {title}State:
    """Per-run state — carries markdown + the current scope."""

    markdown: str = ""
    scope: str = ""
    index: int = 0
    fields: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


class {title}Agent(BaseGraphAgent[{title}State]):
    """Extract structured fields from a markdown document for one scope.

    The agent is *scope-parameterized*: each pipeline map iteration passes a
    different ``scope`` (e.g. ``"parties"``, ``"dates"``, ``"amounts"``) so
    the same agent produces N parallel extractions.  Replace the body of
    :meth:`extract` with your LLM/regex/parser of choice.
    """

    agent_name = "{name}"
    agent_description = "Scope-parameterized markdown extraction agent"

    def __init__(self, *, llm: Any = None) -> None:
        super().__init__()
        self.llm = llm

    def build_graph(self):
        g = self.new_graph()
        g.add_node("load", self.load)
        g.add_node("extract", self.extract)
        g.set_entry_point("load")
        g.add_edge("load", "extract")
        g.set_finish_point("extract")
        return g.compile()

    def load(self, state: {title}State) -> {title}State:
        """Resolve markdown from either an inline string or a file path."""
        raw = state.markdown
        if not raw:
            return state
        candidate = Path(raw).expanduser()
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size < 5_000_000:
            try:
                state.markdown = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                state.markdown = candidate.read_bytes().decode("utf-8", errors="replace")
        return state

    def extract(self, state: {title}State) -> {title}State:
        """Extract fields for ``state.scope`` from ``state.markdown``.

        Replace this method with your real extraction logic (LLM prompt,
        regex chains, structured output parser, etc).
        """
        scope = state.scope or "default"
        snippet = state.markdown[:200].replace("\\n", " ")
        state.fields = {{
            "scope": scope,
            "sample": snippet,
            "length": len(state.markdown),
        }}
        state.output = {{
            "response": f"Extracted scope '{{scope}}' ({{len(state.markdown)}} chars)",
            "scope": scope,
            "fields": state.fields,
        }}
        return state

    def input_to_state(self, data: dict[str, Any]) -> {title}State:
        """Coerce pipeline input into per-run state."""
        scope = data.get("scope") or data.get("item") or ""
        markdown = data.get("markdown") or data.get("current_query") or ""
        try:
            index = int(data.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        return {title}State(markdown=str(markdown), scope=str(scope), index=index)

    def state_to_output(self, state: {title}State) -> dict[str, Any]:
        return state.output
'''
    return body.format(name=name, title=title) + _class_agent_get_graph_export(title)


def _extraction_pipeline_yaml(name: str) -> str:
    return f"""# Extraction pipeline for {name}
#
# ingestion → map(extractor, scope)
#
# POST /api/v1/pipelines/{name}/run  {{"input": {{
#   "source": "/path/to/document.pdf",
#   "scopes": ["parties", "dates", "amounts"]
# }}}}

name: {name}
description: "Parallel scope extraction over markdown ingestion"
version: "1.0.0"

steps:
  - ingestion: markdown
    name: to_md
    input:
      source: $.input.source
      output_dir: $.input.output_dir

  - name: extract_all
    map:
      agent: {name}
      items: $.input.scopes
      item_key: scope
      max_concurrency: 4
      retry:
        max_attempts: 3
        backoff: exponential
        base_delay: 1.0
      input:
        markdown: $.steps.to_md.output.path

on_error: fail_fast
timeout: 300.0
"""


def _extraction_readme(name: str) -> str:
    title = name.replace("_", " ").title()
    return f"""# {title} Extraction Agent

Generated with `agentomatic init {name} --template extraction`.

A **scope-parameterized** extraction agent designed to be fanned out via a
pipeline `map` step — one agent invocation per scope, with retry / progress
/ checkpoint support baked in.

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Extraction agent (edit `extract()` with your logic) |
| `pipeline.yaml` | Ingestion → map fan-out pipeline |
| `README.md` | This file |

## Quick start

```bash
agentomatic run

curl -X POST http://localhost:8000/api/v1/pipelines/{name}/run \\
  -H 'content-type: application/json' \\
  -d '{{"input": {{
        "source": "./document.pdf",
        "scopes": ["parties", "dates", "amounts"]
      }}}}'
```

## Customise

Open `agent.py` and replace the body of `extract()` with your real
extraction logic (LLM prompt, regex chains, structured-output parser,
etc.).  Each map iteration receives a different `scope` under
`state.scope`, so a single agent produces N parallel extractions.
"""


def get_template_files(template: str, name: str) -> dict[str, str]:
    """Get all files for a given template.

    Args:
        template: Template name.
        name: Agent name.

    Returns:
        Dict mapping relative file paths to content strings.
    """
    if template == "class":
        template = "basic"

    title = name.replace("_", " ").title()
    description = f"{title} agent"
    keywords = f'"{name}"'

    common = {
        "prompts.json": _prompts_json(),
        "langgraph.json": _langgraph_json(),
        ".env.example": _env_example(name),
        "README.md": _readme_md(name, template),
    }
    legacy_common = {
        "prompts.json": _prompts_json(),
        "langgraph.json": _langgraph_json(graph_target="./graph.py:get_graph"),
        ".env.example": _env_example(name),
        "README.md": _readme_md(name, template),
    }

    if template == "basic":
        return {
            "__init__.py": _agent_manifest_init_py(name, description, keywords),
            "agent.py": _class_agent_py(name, "basic"),
            "llm.py": _llm_py(name),
            **common,
        }

    elif template == "full":
        return {
            "__init__.py": _agent_manifest_init_py(name, description, keywords),
            "agent.py": _class_agent_py(name, "full"),
            "llm.py": _llm_py(name),
            "config.py": _config_py(name),
            "schemas.py": _schemas_py(name),
            "tools.py": _tools_py(name),
            "api.py": _api_py(name),
            "dataset.jsonl": _dataset_jsonl(name),
            "train.py": _train_py(name),
            "eval.py": _eval_py(name),
            "optimize.py": _optimize_py(name),
            "predict.py": _predict_py(name),
            "search_space.yaml": _search_space_yaml(name),
            "Makefile": _makefile(name),
            **common,
        }

    elif template == "coordinator":
        return {
            "__init__.py": _agent_manifest_init_py(
                name, f"{title} coordinator / orchestrator", keywords
            ),
            "agent.py": _coordinator_agent_py(name),
            "llm.py": _llm_py(name),
            "delegation.py": _coordinator_delegation_py(name),
            "security.py": _coordinator_security_py(name),
            "config.py": _config_py(name),
            **common,
        }

    elif template == "pipeline":
        return {
            "pipeline.yaml": _pipeline_yaml(name),
            "README.md": _pipeline_readme(name),
            "dataset.jsonl": _dataset_jsonl(name),
            "eval.py": _pipeline_eval_py(name),
            "optimize.py": _pipeline_optimize_py(name),
            "run.py": _pipeline_run_py(name),
            "Makefile": _pipeline_makefile(name),
        }

    elif template == "rag":
        return {
            "__init__.py": _agent_manifest_init_py(name, f"{title} RAG agent", keywords),
            "agent.py": _class_agent_py(name, "rag"),
            "llm.py": _llm_py(name),
            "config.py": _config_py(name),
            "tools.py": _tools_py(name),
            **common,
        }

    elif template == "chatbot":
        return {
            "__init__.py": _agent_manifest_init_py(name, f"{title} chatbot agent", keywords),
            "agent.py": _class_agent_py(name, "chatbot"),
            "llm.py": _llm_py(name),
            "config.py": _config_py(name),
            **common,
        }

    elif template == "deepagent":
        return {
            "__init__.py": _deepagent_init_py(name, f"{title} deep agent", keywords),
            "agent.py": _deepagent_agent_py(name),
            "llm.py": _llm_py(name),
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

    elif template == "legacy_dict":
        return {
            "__init__.py": _legacy_init_py(name, description, keywords),
            "graph.py": _legacy_graph_py(name),
            "nodes.py": _legacy_nodes_py(name),
            **legacy_common,
        }

    elif template == "plugin":
        return {
            "__init__.py": '"""ML Model Plugin package."""\n',
            "plugin.py": _plugin_py(name),
            "README.md": _plugin_readme(name),
            "dataset.jsonl": _plugin_dataset_jsonl(name),
            "train.py": _plugin_train_py(name),
            "eval.py": _plugin_eval_py(name),
            "optimize.py": _plugin_optimize_py(name),
            "predict.py": _plugin_predict_py(name),
            "Makefile": _plugin_makefile(name),
        }

    elif template == "endpoint":
        return {
            "__init__.py": '"""Custom Endpoint package."""\n',
            "endpoint.py": _endpoint_py(name),
            "README.md": _endpoint_readme(name),
            ".env.example": _endpoint_env_example(name),
        }

    elif template == "connection":
        return {
            "__init__.py": '"""Connections package."""\n',
            "connections.py": _connections_py(name),
            "README.md": _connections_readme(name),
            ".env.example": _connections_env_example(name),
        }

    elif template == "ingestion":
        return {
            "__init__.py": '"""Ingestor package."""\n',
            "ingestor.py": _ingestor_py(name),
            "README.md": _ingestor_readme(name),
        }

    elif template == "extraction":
        return {
            "__init__.py": _agent_manifest_init_py(name, f"{title} extraction agent", keywords),
            "agent.py": _extraction_agent_py(name),
            "llm.py": _llm_py(name),
            "pipeline.yaml": _extraction_pipeline_yaml(name),
            "prompts.json": _prompts_json(),
            ".env.example": _env_example(name),
            "README.md": _extraction_readme(name),
        }

    else:
        raise ValueError(f"Unknown template: {template}. Choose from: {list(TEMPLATES.keys())}")
