# Delegation & Swarm Orchestration

Agentomatic supports advanced multi-agent patterns out of the box, allowing agents to hand off tasks to one another or operate in collaborative "swarms".

## Agent Handoffs

An agent handoff occurs when an active agent decides that another agent is better suited to answer the user's query. Agentomatic provides an `AgentDelegator` to facilitate this process.

You can explicitly create handoff configurations using `create_agent_handoff`.

```python
from agentomatic.delegation import AgentDelegator, create_agent_handoff

# Setup a delegator that knows about specialized agents
delegator = AgentDelegator()
delegator.register_target("support_agent", "Handles customer support queries")
delegator.register_target("billing_agent", "Handles payment and invoice queries")

# Inside your router node, the agent can decide to hand off
target = delegator.decide_target(user_query="I need a refund")
if target == "billing_agent":
    # Transfer control
    pass
```

## Swarm Orchestration

A Swarm is a group of agents that collaboratively solve a complex problem without rigid, predefined pipelines. The `SwarmOrchestrator` manages the lifecycle of this collaboration.

```python
from agentomatic.delegation import SwarmOrchestrator

orchestrator = SwarmOrchestrator()

orchestrator.add_agent("researcher", researcher_agent)
orchestrator.add_agent("coder", coder_agent)
orchestrator.add_agent("reviewer", reviewer_agent)

# Start a swarm execution
result = await orchestrator.run_swarm(
    task="Research the latest LLM architectures and write a summary script.",
    max_turns=10
)
```

The Swarm Orchestrator will automatically track conversational context between the agents, enforce turn limits to prevent infinite loops, and handle the A2A (Agent-to-Agent) communication protocol securely.
