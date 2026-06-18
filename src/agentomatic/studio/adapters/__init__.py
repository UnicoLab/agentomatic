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
    2. **LangGraph adapter** — If the agent has a ``graph_fn``, use the
       full-featured :class:`LangGraphAdapter`.
    3. **Generic adapter** — For everything else, use the trace-based
       :class:`GenericAdapter` which provides maximum useful information.

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

    # 2. LangGraph agents → full adapter
    if agent.graph_fn is not None:
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        logger.debug(f"Using LangGraph studio adapter for agent '{agent.name}'")
        return LangGraphAdapter(agent=agent, store=store)

    # 3. Everything else → generic trace adapter
    from agentomatic.studio.adapters.generic import GenericAdapter

    logger.debug(f"Using generic studio adapter for agent '{agent.name}'")
    return GenericAdapter(agent=agent, store=store)


__all__ = ["resolve_adapter", "StudioAdapter"]
