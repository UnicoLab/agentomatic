"""Decorator API for declaring graph nodes on agent methods.

Example::

    class MyAgent(BaseGraphAgent[MyState]):
        @agent_node(entrypoint=True)
        def extract(self, state: MyState) -> MyState:
            ...

        @agent_node(after="extract")
        def process(self, state: MyState) -> MyState:
            ...

        @agent_node(after="process", finish=True)
        def format(self, state: MyState) -> MyState:
            ...
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Attribute name used to store metadata on decorated methods
AGENT_NODE_ATTR = "_agent_node_meta"


@dataclass(frozen=True)
class AgentNodeMeta:
    """Metadata attached to a decorated node method.

    Attributes:
        name: Override node name (defaults to method name).
        after: Name of the predecessor node.
        entrypoint: Whether this is the graph entrypoint.
        finish: Whether this is the finish node.
        description: Human-readable description.
        metadata: Arbitrary metadata.
    """

    name: str | None = None
    after: str | None = None
    entrypoint: bool = False
    finish: bool = False
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def agent_node(
    name: str | None = None,
    *,
    after: str | None = None,
    entrypoint: bool = False,
    finish: bool = False,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to mark a method as a graph node.

    Args:
        name: Override node name (defaults to method name).
        after: Name of the predecessor node (creates an edge).
        entrypoint: Whether this is the graph entrypoint.
        finish: Whether this is the finish node.
        description: Human-readable description.
        metadata: Arbitrary metadata.

    Returns:
        The original method with ``_agent_node_meta`` attached.

    Example::

        @agent_node(entrypoint=True)
        def plan(self, state):
            ...

        @agent_node(after="plan", finish=True)
        def execute(self, state):
            ...
    """
    meta = AgentNodeMeta(
        name=name,
        after=after,
        entrypoint=entrypoint,
        finish=finish,
        description=description,
        metadata=metadata or {},
    )

    def decorator(fn: F) -> F:
        setattr(fn, AGENT_NODE_ATTR, meta)
        return fn

    return decorator


def get_node_meta(fn: Any) -> AgentNodeMeta | None:
    """Extract node metadata from a method, if present.

    Args:
        fn: Method or function to inspect.

    Returns:
        ``AgentNodeMeta`` if decorated, else ``None``.
    """
    return getattr(fn, AGENT_NODE_ATTR, None)
