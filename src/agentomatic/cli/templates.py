"""Agent scaffolding templates.

Each template is a dict mapping relative file paths to their content.
Templates: basic, full, rag, chatbot, custom, deepagent, legacy_dict.
"""

from __future__ import annotations


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
    """RAG class agent for {name}."""

    agent_name = "{name}"
    agent_description = "RAG agent"

    def __init__(self, *, llm: Any = None) -> None:
        super().__init__()
        self.llm = llm

    def build_graph(self):
        g = self.new_graph()
        g.add_node("retrieve", self.retrieve)
        g.add_node("generate", self.generate)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.set_finish_point("generate")
        return g.compile()

    def retrieve(self, state: {title}State) -> {title}State:
        # TODO: Replace with real vector search
        state.citations = [
            {{"content": f"Document about {{state.request}}", "source": "knowledge_base"}}
        ]
        return state

    def generate(self, state: {title}State) -> {title}State:
        context = "\\n".join(d.get("content", "") for d in state.citations)
        state.output = {{
            "response": f"Based on the knowledge base: Answer to '{{state.request}}' using context: {{context}}",
            "agent_type": "{name}",
            "citations": state.citations,
        }}
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> {title}State:
        return {title}State(request=input_data.get("current_query", ""))

    def state_to_output(self, state: {title}State) -> dict[str, Any]:
        return state.output
'''
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

    def __init__(self, *, llm: Any = None) -> None:
        super().__init__()
        self.llm = llm

    def build_graph(self):
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: {title}State) -> {title}State:
        history_len = len(state.messages)
        state.output = {{
            "response": f"[Turn {{history_len + 1}}] You said: {{state.request}}",
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
'''
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

    Usage::
        agent = {title}Agent(llm=my_llm)
        result = agent.transform({{"current_query": "Hello"}})
    """

    agent_name = "{name}"
    agent_description = "{title} agent"

    def __init__(self, *, llm: Any = None) -> None:
        super().__init__()
        self.llm = llm
        self.system_prompt = "You are a helpful assistant."

    def build_graph(self):
        g = self.new_graph()
        g.add_node("process", self.process)
        g.set_entry_point("process")
        g.set_finish_point("process")
        return g.compile()

    def process(self, state: {title}State) -> {title}State:
        state.context = [f"Processed: {{state.request}}"]
        state.output = {{
            "response": f"Result for: {{state.request}}",
            "agent_type": "{name}",
        }}
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> {title}State:
        return {title}State(request=input_data.get("current_query", ""))

    def state_to_output(self, state: {title}State) -> dict[str, Any]:
        return state.output
'''


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
    return f"""# {title} Agent\n\nGenerated with `agentomatic init {name} --template {template}`.\n\n## Quick Start\n\n```bash\n# Start the platform\nagentomatic run\n\n# Test the agent\ncurl -X POST http://localhost:8000/api/v1/{name}/invoke \\\n  -H "Content-Type: application/json" \\\n  -d '{{"query": "Hello!"}}'\n```\n\n## Files\n\n| File | Purpose |\n|------|---------|\n| `agent.py` | Agent class definition |\n| `config.py` | Agent-specific configuration |\n| `prompts.json` | Versioned prompt templates |\n| `langgraph.json` | LangGraph Studio config |\n"""


# --- Deep Agent template ---


def _deepagent_init_py(name: str, description: str, keywords: str) -> str:
    return f'''"""Agent: {name} (Deep Agent harness)."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{description}",\n    intent_keywords=[{keywords}],\n    framework="langgraph",\n)\n\n\ndef graph_fn():\n    """Return the compiled deep agent graph."""\n    from .agent import create_agent\n    return create_agent()\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    """Invoke the deep agent."""\n    return await graph_fn().ainvoke(state)\n'''


def _deepagent_agent_py(name: str) -> str:
    safe_title = name.replace("_", " ").title()
    return f'''"""Deep Agent definition for {name}.\n\nUses LangChain\'s `deepagents` harness for planning, tools,\nsubagent delegation, and context management.\n"""\nfrom __future__ import annotations\n\nfrom functools import lru_cache\n\n\ndef internet_search(query: str, max_results: int = 5) -> str:\n    """Search the internet for information.\n\n    Args:\n        query: Search query string.\n        max_results: Maximum number of results.\n\n    Returns:\n        Search results as text.\n    """\n    # TODO: Replace with real search (Tavily, SerpAPI, etc.)\n    return f"Search results for: {{query}} ({{max_results}} results)"\n\n\n@lru_cache(maxsize=1)\ndef create_agent():\n    """Create and compile the deep agent."""\n    from deepagents import create_deep_agent\n\n    return create_deep_agent(\n        model="openai:gpt-4o",  # Change to your preferred model\n        system_prompt=(\n            "You are {safe_title}, "\n            "an expert AI assistant. Be thorough and accurate."\n        ),\n        tools=[internet_search],\n    )\n'''


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
Demonstrates a standard classical ML training loop.
"""
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return [json.loads(line) for line in f]

def train():
    logger.info(f"Training {name} plugin...")
    data = load_data("dataset.jsonl")
    logger.info(f"Loaded {{len(data)}} training examples.")

    # TODO: Implement classical ML training (e.g. Scikit-learn, PyTorch)
    # X = [d["text"] for d in data]
    # y = [d["label"] for d in data]
    # model = LogisticRegression().fit(X, y)

    logger.info("Saving model weights to disk...")
    # TODO: joblib.dump(model, "model_weights.pkl")
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
'''


def _plugin_eval_py(name: str) -> str:
    return f'''"""Evaluation script for {name} ML plugin.
Evaluates the loaded model against a test dataset.
"""
import json
import asyncio
import logging
from plugin import {name.replace("_", "").title()}Plugin, {name.replace("_", "").title()}Input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def evaluate():
    logger.info(f"Evaluating {name} plugin...")
    plugin = {name.replace("_", "").title()}Plugin()
    await plugin.load_model()

    # Mock data loading
    examples = [{{"text": "Test input", "expected_label": 1}}]

    correct = 0
    for ex in examples:
        result = await plugin.predict({name.replace("_", "").title()}Input(text=ex["text"]))
        # TODO: Compute real metrics like F1, Accuracy
        if result.result:
            correct += 1

    logger.info(f"Accuracy: {{correct / len(examples):.2f}}")

if __name__ == "__main__":
    asyncio.run(evaluate())
'''


def _plugin_optimize_py(name: str) -> str:
    return f'''"""Hyperparameter optimization script for {name} plugin."""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize():
    logger.info(f"Optimizing {name} plugin hyperparameters...")
    # TODO: Implement GridSearchCV, Optuna, or random search
    # Best params -> retrain model -> save
    logger.info("Optimization complete.")

if __name__ == "__main__":
    optimize()
'''


def _plugin_predict_py(name: str) -> str:
    title = name.replace("_", "").title()
    return f'''"""Local inference script for {name} plugin."""
import asyncio
from plugin import {title}Plugin, {title}Input

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
    return f'''"""Train script for {name} — compile, fit, and save.

Usage::

    python -m agents.{name}.train
    # or
    python agents/{name}/train.py
"""
from __future__ import annotations

from pathlib import Path

from .agent import {title}Agent

from agentomatic.agents import AgentDataset
from agentomatic.agents.metrics import (
    ContainsTermsMetric,
    ExactKeyMatchMetric,
)
from agentomatic.agents.optimizers import NoOpOptimizer

DATA_DIR = Path(__file__).parent
COMPILED_DIR = Path("compiled") / "{name}"


def main() -> None:
    """Compile, fit, and save the agent."""
    # 1. Create agent
    agent = {title}Agent(llm=None)
    print(f"Agent: {{agent.agent_name}}")

    # 2. Load dataset
    dataset = AgentDataset.from_jsonl(str(DATA_DIR / "dataset.jsonl"))
    print(f"Dataset: {{len(dataset)}} examples "
          f"(train={{len(dataset.train)}}, test={{len(dataset.test)}})")

    # 3. Compile — register metrics + optimizer
    metrics = [
        ExactKeyMatchMetric(["response"]),
        ContainsTermsMetric(["Result"]),
    ]
    agent.compile(dataset, metrics, optimizer=NoOpOptimizer())
    print("Compiled with 2 metrics + NoOpOptimizer")

    # 4. Fit — run optimization loop
    agent.fit(dataset)
    print("Fit complete")

    # 5. Quick evaluation on test set
    report = agent.evaluate(dataset.test, metrics)
    print(f"Test pass rate: {{report.pass_rate:.1%}}")

    # 6. Save compiled agent
    agent.save(str(COMPILED_DIR))
    print(f"Saved to {{COMPILED_DIR}}/")


if __name__ == "__main__":
    main()
'''


def _eval_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Evaluate script for {name} — detailed quality reporting.

Usage::

    python -m agents.{name}.eval
    python -m agents.{name}.eval --compiled compiled/{name}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import {title}Agent

from agentomatic.agents import AgentDataset
from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
)

DATA_DIR = Path(__file__).parent


def main() -> None:
    """Evaluate the agent and print a detailed report."""
    parser = argparse.ArgumentParser(description="Evaluate {name}")
    parser.add_argument(
        "--compiled", type=str, default=None,
        help="Path to compiled agent directory (loads saved config)"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATA_DIR / "dataset.jsonl"),
        help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--split", choices=["test", "train", "all"], default="test",
        help="Dataset split to evaluate on"
    )
    args = parser.parse_args()

    # 1. Create agent
    agent = {title}Agent(llm=None)

    # Load compiled state if provided
    if args.compiled:
        agent.load_compiled(args.compiled)
        print(f"Loaded compiled config from {{args.compiled}}")

    # 2. Load dataset
    dataset = AgentDataset.from_jsonl(args.dataset)
    if args.split == "test":
        examples = dataset.test
    elif args.split == "train":
        examples = dataset.train
    else:
        examples = dataset.examples
    print(f"Evaluating on {{len(examples)}} examples (split={{args.split}})")

    # 3. Define metrics
    metrics = [
        ExactKeyMatchMetric(["response"]),
        ContainsTermsMetric(["Result"]),
        CallableMetric(
            "has_output",
            lambda example, pred: 1.0 if pred.get("response") else 0.0,
        ),
    ]

    # 4. Evaluate
    report = agent.evaluate(examples, metrics)

    # 5. Print report
    print()
    print(report.summary())
    print()

    # 6. Show per-example details
    for result in report.example_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{{status}}] {{result.example_id}} "
              f"({{result.duration_ms:.0f}}ms) — {{result.scores}}")
        if result.error:
            print(f"         ERROR: {{result.error}}")

    # 7. Save report as JSON
    report_path = DATA_DIR / "eval_report.json"
    report_data = {{
        "agent": report.agent_name,
        "dataset": report.dataset_name,
        "pass_rate": report.pass_rate,
        "num_examples": report.num_examples,
        "scores": report.scores,
    }}
    report_path.write_text(json.dumps(report_data, indent=2))
    print(f"\\nReport saved to {{report_path}}")


if __name__ == "__main__":
    main()
'''


def _optimize_py(name: str) -> str:
    title = name.replace("_", " ").title().replace(" ", "")
    return f'''"""Prompt optimization script for {name}.

Supports two optimization strategies:
  - GridSearch: brute-force parameter sweep (temperature, prompt_version)
  - PromptFitter: LLM-powered prompt rewriting (requires agentomatic[optimize])

Usage::

    python -m agents.{name}.optimize
    python -m agents.{name}.optimize --strategy grid
    python -m agents.{name}.optimize --strategy prompt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .agent import {title}Agent

from agentomatic.agents import AgentDataset
from agentomatic.agents.metrics import (
    ContainsTermsMetric,
    ExactKeyMatchMetric,
)
from agentomatic.agents.optimizers import GridSearchOptimizer, PromptFitterBridge

DATA_DIR = Path(__file__).parent
COMPILED_DIR = Path("compiled") / "{name}"


def run_grid_search(agent: {title}Agent, dataset: AgentDataset) -> None:
    """Run GridSearch optimization."""
    metrics = [
        ExactKeyMatchMetric(["response"]),
        ContainsTermsMetric(["Result"]),
    ]

    optimizer = GridSearchOptimizer(
        param_grid={{
            "system_prompt": [
                "You are a helpful assistant.",
                "You are a precise, detail-oriented assistant.",
                "You are a concise assistant. Be brief and accurate.",
            ],
            # Add more parameters to tune:
            # "temperature": [0.0, 0.2, 0.5, 0.8],
            # "retrieval_top_k": [3, 5, 8],
        }},
        max_examples=10,  # limit examples per combo for speed
    )

    print("Running GridSearch optimization...")
    print(f"  Training examples: {{len(dataset.train)}}")
    agent.compile(dataset, metrics, optimizer=optimizer)
    agent.fit(dataset)

    print("\\nOptimized config:")
    for key, value in agent.compiled_config.items():
        print(f"  {{key}}: {{value}}")

    # Evaluate optimized agent
    report = agent.evaluate(dataset.test, metrics)
    print(f"\\nTest pass rate: {{report.pass_rate:.1%}}")
    print(report.summary())

    # Save
    agent.save(str(COMPILED_DIR))
    print(f"\\nSaved optimized agent to {{COMPILED_DIR}}/")


def run_prompt_fitter(agent: {title}Agent, dataset: AgentDataset) -> None:
    """Run PromptFitter optimization (LLM-powered prompt rewriting)."""
    metrics = [
        ExactKeyMatchMetric(["response"]),
        ContainsTermsMetric(["Result"]),
    ]

    optimizer = PromptFitterBridge(
        agent_name="{name}",
        task_model="ollama/qwen2.5:7b",      # model that runs your agent
        rewrite_model="openai/gpt-4.1",       # model that rewrites prompts
    )

    print("Running PromptFitter optimization...")
    print("  This uses an LLM to rewrite your system prompt for better quality.")
    agent.compile(dataset, metrics, optimizer=optimizer)
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
        "--strategy", choices=["grid", "prompt"], default="grid",
        help="Optimization strategy: grid (GridSearch) or prompt (PromptFitter)"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATA_DIR / "dataset.jsonl"),
        help="Path to training dataset (JSONL)"
    )
    args = parser.parse_args()

    agent = {title}Agent(llm=None)
    dataset = AgentDataset.from_jsonl(args.dataset)

    if args.strategy == "grid":
        run_grid_search(agent, dataset)
    elif args.strategy == "prompt":
        run_prompt_fitter(agent, dataset)


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


def _makefile(name: str) -> str:
    return f"""# Makefile for {name} agent — ML lifecycle commands
#
# Usage:
#   make train      — Compile, fit, and save the agent
#   make eval       — Evaluate on the test split
#   make optimize   — Run GridSearch parameter optimization
#   make predict    — Start interactive prediction mode
#   make all        — Full pipeline: train → eval

.PHONY: train eval optimize predict all clean

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
\t@echo "\\n🔧 Optimizing $(AGENT) with GridSearch..."
\tpython -m agents.$(AGENT).optimize --strategy grid

optimize-prompt:
\t@echo "\\n🔧 Optimizing $(AGENT) with PromptFitter..."
\tpython -m agents.$(AGENT).optimize --strategy prompt

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


def get_template_files(template: str, name: str) -> dict[str, str]:
    """Get all files for a given template.

    Args:
        template: Template name.
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
            "__init__.py": '"""Agent package."""\n',
            "agent.py": _class_agent_py(name, "basic"),
            **common,
        }

    elif template == "full":
        return {
            "__init__.py": '"""Agent package."""\n',
            "agent.py": _class_agent_py(name, "full"),
            "config.py": _config_py(name),
            "schemas.py": _schemas_py(name),
            "tools.py": _tools_py(name),
            "api.py": _api_py(name),
            "dataset.jsonl": _dataset_jsonl(name),
            "train.py": _train_py(name),
            "eval.py": _eval_py(name),
            "optimize.py": _optimize_py(name),
            "predict.py": _predict_py(name),
            "Makefile": _makefile(name),
            **common,
        }

    elif template == "coordinator":
        return {
            "__init__.py": '"""Agent package."""\n',
            "agent.py": _coordinator_agent_py(name),
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
            "__init__.py": '"""Agent package."""\n',
            "agent.py": _class_agent_py(name, "rag"),
            "config.py": _config_py(name),
            "tools.py": _tools_py(name),
            **common,
        }

    elif template == "chatbot":
        return {
            "__init__.py": '"""Agent package."""\n',
            "agent.py": _class_agent_py(name, "chatbot"),
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

    elif template == "legacy_dict":
        return {
            "__init__.py": _legacy_init_py(name, description, keywords),
            "graph.py": _legacy_graph_py(name),
            "nodes.py": _legacy_nodes_py(name),
            **common,
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

    else:
        raise ValueError(f"Unknown template: {template}. Choose from: {list(TEMPLATES.keys())}")
