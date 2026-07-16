"""Shared helpers for invoking a :class:`RegisteredAgent`.

Used by pipelines, flow handles, and any other component that needs to
call a discovered agent with a state/input dict without knowing whether
the agent is folder-based or a class-owned ``BaseGraphAgent``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentomatic.core.manifest import RegisteredAgent


def _input_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build a ``transform()`` input dict from a pipeline/flow state dict.

    Flattens ``context`` into the top-level payload so class-agent
    ``input_to_state`` implementations can read fields that arrived under
    :class:`~agentomatic.core.router_factory.AgentInvokeRequest`'s
    ``context`` dict without digging into a nested key. Explicit top-level
    keys always win over colliding context keys.
    """
    query = state.get("current_query", state.get("query", ""))
    skip = {"messages", "thread_id", "context"}
    payload: dict[str, Any] = {
        "query": query,
        **{k: v for k, v in state.items() if k not in skip},
    }
    context = state.get("context")
    if isinstance(context, dict):
        for key, value in context.items():
            payload.setdefault(key, value)
    # Keep nested context available for agents that still read it explicitly.
    if context is not None:
        payload["context"] = context
    if not payload.get("query"):
        payload["query"] = query
    if "current_query" not in payload:
        payload["current_query"] = query
    return payload


async def invoke_registered_agent(
    agent: RegisteredAgent,
    state: dict[str, Any],
) -> Any:
    """Invoke a registered agent, preferring class-agent ``atransform``.

    Class agents registered via :meth:`BaseGraphAgent.as_registered_agent`
    expose both ``graph_fn`` and ``node_fn``. Calling ``graph.ainvoke`` with a
    raw dict skips ``input_to_state`` and breaks dataclass-typed states.
    When a ``BaseGraphAgent`` ``class_instance`` is attached we therefore
    route through ``atransform`` so ``input_to_state`` / ``state_to_output``
    run correctly.

    Folder-based LangGraph agents keep the previous preference order:
    ``graph_fn`` then ``node_fn``.

    Args:
        agent: Discovered / registered agent.
        state: BaseAgentState-compatible (or free-form) input dict.

    Returns:
        Agent output (dict or whatever the callable returns).

    Raises:
        RuntimeError: If the agent has no invokable callable.
    """
    instance = getattr(agent, "class_instance", None)
    if instance is not None:
        from agentomatic.agents.base import BaseGraphAgent

        if isinstance(instance, BaseGraphAgent):
            return await instance.atransform(_input_from_state(state))

    if agent.graph_fn is not None:
        graph = agent.graph_fn()
        return await graph.ainvoke(state)

    if agent.node_fn is not None:
        return await agent.node_fn(state)

    name = getattr(agent, "name", "?")
    raise RuntimeError(f"Agent '{name}' has no callable (class_instance/graph_fn/node_fn)")
