"""LangGraph graph for weather agent.

This file demonstrates a multi-step graph with branching.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from agentomatic import BaseAgentState

from . import nodes


def build_graph() -> StateGraph:
    """Build a multi-step weather agent graph."""
    g = StateGraph(BaseAgentState)
    
    # Add nodes
    g.add_node("classify", nodes.classify_query)
    g.add_node("forecast", nodes.get_forecast)
    g.add_node("alert", nodes.get_alerts)
    g.add_node("respond", nodes.format_response)
    
    # Set entry point
    g.set_entry_point("classify")
    
    # Conditional routing after classification
    g.add_conditional_edges(
        "classify",
        nodes.route_query,
        {
            "forecast": "forecast",
            "alert": "alert",
        },
    )
    
    # Both paths converge to respond
    g.add_edge("forecast", "respond")
    g.add_edge("alert", "respond")
    g.add_edge("respond", END)
    
    return g


@lru_cache(maxsize=1)
def get_graph():
    """Get the compiled graph (cached singleton)."""
    return build_graph().compile()
