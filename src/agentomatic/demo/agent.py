"""Demo agent with Studio integration for E2E testing.

Provides a self-contained agent that simulates multi-step reasoning
with artificial latency, producing markdown-formatted responses.
Decorated with ``@studio_graph`` and ``@studio_state`` so the Studio
debug UI can visualise the execution topology and inspect state.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agentomatic.core.manifest import AgentManifest
from agentomatic.studio.decorators import studio_graph, studio_state

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

manifest = AgentManifest(
    name="demo_assistant",
    slug="agentomatic-demo-assistant",
    description=(
        "Built-in demo agent that simulates multi-step reasoning "
        "for E2E testing of Agentomatic Studio."
    ),
    intent_keywords=["demo", "test", "hello", "help"],
    version="1.0.0",
    is_subagent=True,
    framework="custom",
)

# ---------------------------------------------------------------------------
# Internal step simulation helpers
# ---------------------------------------------------------------------------

_STEPS: list[dict[str, str]] = [
    {
        "name": "Research",
        "description": "Gathering relevant information about the query",
    },
    {
        "name": "Analyze",
        "description": "Evaluating sources and identifying key insights",
    },
    {
        "name": "Synthesize",
        "description": "Combining findings into a coherent narrative",
    },
    {
        "name": "Respond",
        "description": "Formatting the final answer with citations",
    },
]

# In-memory state store keyed by thread_id (for Studio inspection)
_state_store: dict[str, dict[str, Any]] = {}


def _build_markdown_response(query: str, steps_taken: list[str]) -> str:
    """Generate a realistic markdown response about the query."""
    return (
        f"## Analysis: {query}\n\n"
        f"After completing **{len(steps_taken)} reasoning steps** "
        f"({' → '.join(steps_taken)}), here are the findings:\n\n"
        f"### Key Insights\n\n"
        f'1. **Context** — The query *"{query}"* was processed through '
        f"the Agentomatic demo pipeline.\n"
        f"2. **Methodology** — Each step simulated real-world latency "
        f"(~0.5 s per node) to mirror production agent behaviour.\n"
        f"3. **Result** — The pipeline completed successfully with full "
        f"Studio traceability.\n\n"
        f"### Pipeline Summary\n\n"
        f"| Step | Status |\n"
        f"|------|--------|\n" + "".join(f"| {s} | ✅ Complete |\n" for s in steps_taken) + "\n"
        "> **Tip:** Open the Studio UI at `/studio/ui/` to visualise "
        "this run's graph and inspect per-node state.\n"
    )


# ---------------------------------------------------------------------------
# Node function (the agent's entry point)
# ---------------------------------------------------------------------------


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Simulate multi-step reasoning with artificial latency.

    Args:
        state: A dict containing at least ``current_query`` (str).

    Returns:
        Dict with ``response``, ``steps_taken``, ``duration_ms``,
        and ``suggestions``.
    """
    query: str = state.get("current_query", state.get("query", "Hello!"))
    thread_id: str = state.get("thread_id", "demo")

    start = time.perf_counter()
    steps_taken: list[str] = []

    # Update shared state for Studio inspection
    _state_store[thread_id] = {
        "query": query,
        "status": "running",
        "steps_completed": [],
    }

    for step in _STEPS:
        await asyncio.sleep(0.5)  # simulate work
        steps_taken.append(step["name"])
        _state_store[thread_id]["steps_completed"] = list(steps_taken)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response = _build_markdown_response(query, steps_taken)

    _state_store[thread_id].update(
        {
            "status": "complete",
            "response": response,
            "duration_ms": elapsed_ms,
        }
    )

    return {
        "response": response,
        "steps_taken": steps_taken,
        "duration_ms": elapsed_ms,
        "suggestions": [
            "Try a different query",
            "Check the Studio graph view",
            "Inspect thread state in Studio",
        ],
    }


# ---------------------------------------------------------------------------
# Studio decorators
# ---------------------------------------------------------------------------


@studio_graph
def demo_graph_topology() -> dict[str, Any]:
    """Provide the Studio with a custom graph topology.

    Returns:
        A dict with ``nodes`` and ``edges`` describing the demo pipeline.
    """
    return {
        "nodes": [
            {"id": "__start__", "name": "Input", "type": "start"},
            {"id": "research", "name": "Research", "type": "agent"},
            {"id": "analyze", "name": "Analyze", "type": "agent"},
            {"id": "synthesize", "name": "Synthesize", "type": "tool"},
            {"id": "respond", "name": "Respond", "type": "agent"},
            {"id": "__end__", "name": "Output", "type": "end"},
        ],
        "edges": [
            {"source": "__start__", "target": "research"},
            {"source": "research", "target": "analyze"},
            {"source": "analyze", "target": "synthesize"},
            {"source": "synthesize", "target": "respond"},
            {"source": "respond", "target": "__end__"},
        ],
    }


@studio_state
async def demo_state_provider(thread_id: str) -> dict[str, Any]:
    """Return the current state for a given thread.

    Args:
        thread_id: The thread identifier to look up.

    Returns:
        Dict with the latest state snapshot, or an empty marker.
    """
    return _state_store.get(thread_id, {"status": "unknown", "thread_id": thread_id})
