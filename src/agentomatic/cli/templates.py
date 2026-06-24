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
# Template Registry
# =====================================================================

TEMPLATES: dict[str, str] = {
    "basic": "Minimal class-based agent (recommended) — 1 file, quick start",
    "full": "All files — class agent with config, schemas, tools, dataset, train/eval scripts",
    "rag": "RAG class-based agent — retrieve → generate pipeline",
    "chatbot": "Conversational class-based agent with memory",
    "deepagent": "Deep Agent — planning, tools, subagents (requires deepagents package)",
    "custom": "Framework-agnostic — no LangGraph dependency",
    "legacy_dict": "Legacy functional agent — 3 files (__init__, graph, nodes)",
    "plugin": "ML Model Plugin — wrap classical ML models with auto-generated REST endpoints",
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
            "__init__.py": '"""Agent package."""\\n',
            "agent.py": _class_agent_py(name, "basic"),
            **common,
        }

    elif template == "full":
        return {
            "__init__.py": '"""Agent package."""\\n',
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

    elif template == "rag":
        return {
            "__init__.py": '"""Agent package."""\\n',
            "agent.py": _class_agent_py(name, "rag"),
            "config.py": _config_py(name),
            "tools.py": _tools_py(name),
            **common,
        }

    elif template == "chatbot":
        return {
            "__init__.py": '"""Agent package."""\\n',
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
            "__init__.py": '"""ML Model Plugin package."""\\n',
            "plugin.py": _plugin_py(name),
            "README.md": _plugin_readme(name),
        }

    else:
        raise ValueError(f"Unknown template: {template}. Choose from: {list(TEMPLATES.keys())}")
