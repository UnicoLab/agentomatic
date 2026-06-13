"""LangGraph graph for hello agent."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from agentomatic import BaseAgentState

from . import nodes


def build_graph() -> StateGraph:
    """Build the hello agent graph."""
    g = StateGraph(BaseAgentState)
    g.add_node("greet", nodes.greet)
    g.set_entry_point("greet")
    g.add_edge("greet", END)
    return g


@lru_cache(maxsize=1)
def get_graph():
    """Get the compiled graph (cached)."""
    return build_graph().compile()
