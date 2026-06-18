"""Studio decorators for framework-agnostic agents.

These decorators allow any agent to incrementally opt-in to Studio
capabilities without requiring LangGraph. Simply decorate functions
on your agent module and the Studio will automatically pick them up.

Example usage::

    from agentomatic.studio.decorators import studio_graph, studio_state

    @studio_graph
    def my_topology():
        return {
            "nodes": [
                {"id": "__start__", "name": "Start", "type": "start"},
                {"id": "process", "name": "Process Data", "type": "agent"},
                {"id": "validate", "name": "Validate", "type": "condition"},
                {"id": "__end__", "name": "End", "type": "end"},
            ],
            "edges": [
                {"source": "__start__", "target": "process"},
                {"source": "process", "target": "validate"},
                {"source": "validate", "target": "__end__", "condition": "valid"},
                {"source": "validate", "target": "process", "condition": "retry"},
            ]
        }

    @studio_state
    async def get_my_state(thread_id: str) -> dict:
        return await my_database.get_thread_state(thread_id)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def studio_graph(fn: Callable[[], dict[str, Any]]) -> Callable[[], dict[str, Any]]:
    """Decorator to register a custom graph topology provider.

    The decorated function should return a dict with ``'nodes'`` and
    ``'edges'`` keys. Each node should have ``id``, ``name``, and
    ``type`` fields. Each edge should have ``source`` and ``target``
    fields, with an optional ``condition`` label.

    Supported node types: ``'start'``, ``'end'``, ``'agent'``,
    ``'tool'``, ``'condition'``, ``'human'``.

    Args:
        fn: A callable that returns the graph topology dict.

    Returns:
        The same callable, marked with ``_is_studio_graph = True``.
    """
    fn._is_studio_graph = True  # type: ignore[attr-defined]
    return fn


def studio_state(fn: Callable) -> Callable:
    """Decorator to register a custom state provider.

    The decorated function receives a ``thread_id`` string and should
    return a dict representing the current state. It can be sync or async.

    Args:
        fn: A callable ``(thread_id: str) -> dict``.

    Returns:
        The same callable, marked with ``_is_studio_state = True``.
    """
    fn._is_studio_state = True  # type: ignore[attr-defined]
    return fn


def studio_stream(fn: Callable) -> Callable:
    """Decorator to register a custom SSE event stream provider.

    The decorated function should be an async generator that yields
    :class:`~agentomatic.studio.models.StudioRunEvent` instances.

    Signature::

        async def my_streamer(
            state: dict,
            config: dict | None,
            breakpoints: list[str] | None,
        ) -> AsyncGenerator[StudioRunEvent, None]:
            yield StudioRunEvent(event="node_start", ...)
            ...

    Args:
        fn: An async generator callable.

    Returns:
        The same callable, marked with ``_is_studio_stream = True``.
    """
    fn._is_studio_stream = True  # type: ignore[attr-defined]
    return fn


def register_studio_hooks(agent: Any) -> None:
    """Scan an agent module and attach discovered studio hooks.

    Called during agent discovery to find functions decorated with
    ``@studio_graph``, ``@studio_state``, or ``@studio_stream`` and
    attach them to the ``RegisteredAgent`` for the adapter factory
    to pick up.

    Args:
        agent: A :class:`~agentomatic.core.manifest.RegisteredAgent`.
    """
    if not agent.module_path:
        return

    try:
        import importlib

        mod = importlib.import_module(agent.module_path)
    except ImportError:
        return

    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if obj is None or not callable(obj):
            continue

        if getattr(obj, "_is_studio_graph", False):
            agent._studio_graph_fn = obj
        elif getattr(obj, "_is_studio_state", False):
            agent._studio_state_fn = obj
        elif getattr(obj, "_is_studio_stream", False):
            agent._studio_stream_fn = obj
