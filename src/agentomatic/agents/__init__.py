"""Class-owned graph agents — ML-like lifecycle for GenAI agents.

Define agents as Python classes with LangGraph-style graph wiring::

    from dataclasses import dataclass
    from agentomatic.agents import BaseGraphAgent

    @dataclass
    class MyState:
        query: str = ""
        response: str = ""

    class MyAgent(BaseGraphAgent[MyState]):
        agent_name = "my_agent"

        def __init__(self, *, llm=None):
            super().__init__()
            self.llm = llm

        def build_graph(self):
            g = self.new_graph()
            g.add_node("process", self.process)
            g.set_entry_point("process")
            g.set_finish_point("process")
            return g.compile()

        def process(self, state):
            state.response = "hello"
            return state

        def input_to_state(self, data):
            return MyState(query=data.get("query", ""))

        def state_to_output(self, state):
            return {"response": state.response}

    # ML-like workflow
    agent = MyAgent(llm=my_llm)
    result = agent.transform({"query": "hello"})
"""

from __future__ import annotations

from agentomatic.agents.base import BaseGraphAgent
from agentomatic.agents.builder import GraphBuilder
from agentomatic.agents.decorators import agent_node
from agentomatic.agents.graph import AgentGraph, GraphNode
from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
    OptimizeMetricAdapter,
)
from agentomatic.agents.optimizers import (
    GridSearchOptimizer,
    NoOpOptimizer,
    PromptFitterBridge,
)
from agentomatic.agents.types import (
    AgentDataset,
    AgentExample,
    EvaluationReport,
    ExampleResult,
    Metric,
    Optimizer,
    TraceEvent,
)

__all__ = [
    # Core
    "BaseGraphAgent",
    "AgentGraph",
    "GraphNode",
    "GraphBuilder",
    "agent_node",
    # Types
    "AgentDataset",
    "AgentExample",
    "EvaluationReport",
    "ExampleResult",
    "Metric",
    "Optimizer",
    "TraceEvent",
    # Metrics
    "ExactKeyMatchMetric",
    "ContainsTermsMetric",
    "CallableMetric",
    "OptimizeMetricAdapter",
    # Optimizers
    "NoOpOptimizer",
    "GridSearchOptimizer",
    "PromptFitterBridge",
]
