from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from agentomatic.core.registry import agent

class State(TypedDict):
    messages: list[str]

def node_a(state: State):
    return {"messages": state.get("messages", []) + ["A"]}

def node_b(state: State):
    return {"messages": state.get("messages", []) + ["B"]}

builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

@agent(name="demo_agent")
def get_graph():
    return builder.compile()
