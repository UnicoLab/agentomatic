"""Adapter factory for Agentomatic Studio.

Resolves the best :class:`~agentomatic.studio.adapter.StudioAdapter` for
a given :class:`~agentomatic.core.manifest.RegisteredAgent`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from agentomatic.studio.adapter import StudioAdapter

if TYPE_CHECKING:
    from agentomatic.core.manifest import RegisteredAgent
    from agentomatic.storage.base import BaseStore


def resolve_adapter(
    agent: RegisteredAgent,
    store: BaseStore | None = None,
) -> StudioAdapter:
    """Resolve the best Studio adapter for an agent.

    Resolution order:

    1. **Custom adapter** — If the agent has a ``_studio_adapter``
       attribute (set via decorators or manual registration), use it.
    2. **Class / AgentGraph agents** — If ``class_instance`` is set or
       ``manifest.framework == "graph_agent"``, use
       :class:`GraphAgentAdapter` (must run *before* the generic
       ``graph_fn`` → LangGraph branch — class agents also expose
       ``graph_fn``).
    3. **LangGraph adapter** — If the agent has a ``graph_fn``, or the
       manifest declares ``langgraph`` / ``deepagent``.
    4. **LangChain adapter** — If the agent's manifest declares
       ``framework='langchain'``.
    5. **Generic adapter** — For everything else.

    Args:
        agent: The registered agent to create an adapter for.
        store: Optional storage backend for checkpoint operations.

    Returns:
        A :class:`StudioAdapter` instance.
    """
    # 1. User-registered custom adapter
    custom = getattr(agent, "_studio_adapter", None)
    if custom is not None:
        logger.debug(f"Using custom studio adapter for agent '{agent.name}'")
        return custom

    framework = getattr(agent.manifest, "framework", "") or ""

    # 2. Class-owned BaseGraphAgent — BEFORE generic graph_fn → LangGraph
    class_instance = getattr(agent, "class_instance", None)
    is_class_agent = False
    if class_instance is not None:
        try:
            from agentomatic.agents.base import BaseGraphAgent

            is_class_agent = isinstance(class_instance, BaseGraphAgent)
        except ImportError:  # pragma: no cover
            is_class_agent = False
    if is_class_agent or framework == "graph_agent":
        from agentomatic.studio.adapters.graph_agent import GraphAgentAdapter

        logger.debug(f"Using GraphAgent studio adapter for agent '{agent.name}'")
        return GraphAgentAdapter(agent=agent, store=store)

    # 3. LangGraph agents (including deep_agent) → full adapter
    if agent.graph_fn is not None:
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        logger.debug(f"Using LangGraph studio adapter for agent '{agent.name}'")
        return LangGraphAdapter(agent=agent, store=store)

    if framework in ("langgraph", "deepagent"):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        logger.debug(f"Using LangGraph studio adapter (framework hint) for agent '{agent.name}'")
        return LangGraphAdapter(agent=agent, store=store)

    # 4. LangChain agents → enhanced adapter
    if framework == "langchain":
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        logger.debug(f"Using LangChain studio adapter for agent '{agent.name}'")
        return LangChainAdapter(agent=agent, store=store)

    # 5. Everything else → generic trace adapter
    from agentomatic.studio.adapters.generic import GenericAdapter

    logger.debug(f"Using generic studio adapter for agent '{agent.name}'")
    return GenericAdapter(agent=agent, store=store)


__all__ = ["resolve_adapter", "StudioAdapter"]
