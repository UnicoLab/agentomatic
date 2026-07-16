"""Shared helpers for invoking a :class:`RegisteredAgent`.

Used by pipelines, flow handles, and any other component that needs to
call a discovered agent with a state/input dict without knowing whether
the agent is folder-based or a class-owned ``BaseGraphAgent``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentomatic.core.manifest import RegisteredAgent

# Keys managed by the platform when building invoke state. Everything else
# from the client payload is preserved as a top-level state field.
_STRUCTURAL_KEYS = frozenset(
    {
        "query",
        "current_query",
        "user_id",
        "thread_id",
        "context",
        "metadata",
        "messages",
        "steps_taken",
        "response",
        "suggestions",
        "citations",
        "prompt_version",
    }
)


def build_invoke_state(
    payload: dict[str, Any] | Any,
    *,
    default_thread_id: str | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Build agent state from a full client payload with complete passthrough.

    Every field present in the client JSON is preserved. Known framework
    keys are normalised (``query`` → ``current_query``, etc.); unknown
    top-level keys remain at the top level so ``input_to_state`` sees them.
    Nested ``context`` is kept intact and later flattened by
    :func:`_input_from_state`.

    Args:
        payload: Request body dict, Pydantic model, or scalar query.
        default_thread_id: Thread id when the payload omits one.
        prompt_version: Optional override for A/B / explicit version.

    Returns:
        A BaseAgentState-compatible dict ready for
        :func:`invoke_registered_agent`.
    """
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        data = {"query": payload}

    query = data.get("query") or data.get("current_query") or ""
    if not query:
        for key, val in data.items():
            if key.endswith("_query") and isinstance(val, str):
                query = val
                break

    user_id = data.get("user_id") or "default-user"
    thread_id = data.get("thread_id") or default_thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    context = data.get("context") if isinstance(data.get("context"), dict) else {}
    if not isinstance(context, dict):
        context = {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Preserve every non-structural top-level key (extras + known knobs).
    extras = {k: v for k, v in data.items() if k not in _STRUCTURAL_KEYS}

    # Mirror control knobs into metadata for back-compat with agents that
    # historically only looked there.
    for key in ("temperature", "max_tokens", "prompt_version"):
        if key in extras and key not in metadata:
            metadata[key] = extras[key]

    chosen_version = prompt_version
    if chosen_version is None:
        chosen_version = data.get("prompt_version") or "v1"

    state: dict[str, Any] = {
        "current_query": query,
        "query": query,
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": [],
        "context": context,
        "metadata": metadata,
        "steps_taken": [],
        "response": "",
        "suggestions": [],
        "citations": [],
        "prompt_version": chosen_version,
        **extras,
    }
    return state


def _input_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build a ``transform()`` input dict from a pipeline/flow state dict.

    Flattens ``context`` into the top-level payload so class-agent
    ``input_to_state`` implementations can read fields that arrived under
    :class:`~agentomatic.core.router_factory.AgentInvokeRequest`'s
    ``context`` dict without digging into a nested key. Explicit top-level
    keys always win over colliding context keys. The full nested
    ``context`` dict is retained under ``payload["context"]``.

    Args:
        state: BaseAgentState-compatible (or free-form) input dict.

    Returns:
        Dict passed to ``BaseGraphAgent.input_to_state`` / ``atransform``.
    """
    query = state.get("current_query", state.get("query", ""))
    # Drop conversation bookkeeping that agents rarely want as input fields.
    skip = {"messages", "thread_id"}
    payload: dict[str, Any] = {
        "query": query,
        **{k: v for k, v in state.items() if k not in skip and k != "context"},
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
