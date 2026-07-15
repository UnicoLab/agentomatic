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
    """Build a ``transform()`` input dict from a pipeline/flow state dict."""
    query = state.get("current_query", state.get("query", ""))
    payload = {
        "query": query,
        **{k: v for k, v in state.items() if k not in ("messages", "thread_id")},
    }
    if "query" not in payload or payload["query"] == "":
        payload["query"] = query
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
