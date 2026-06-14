"""Default agent state for LangGraph-based agents."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

try:
    from langgraph.graph import add_messages

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

    def add_messages(left, right):  # type: ignore[misc]
        """Fallback: simple list concatenation."""
        if left is None:
            return right
        return left + right


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts (last-writer-wins per key)."""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


def _last_value(left: Any, right: Any) -> Any:
    """Last-writer-wins reducer for scalar fields."""
    return right if right is not None else left


class BaseAgentState(TypedDict, total=False):
    """Default state shared across all agents.

    Users can subclass or replace this with their own TypedDict.
    All fields are optional (``total=False``) for maximum flexibility.

    Fields that may be written by parallel branches use Annotated
    reducers to avoid LangGraph "multiple values per step" errors.
    """

    messages: Annotated[list, add_messages]
    thread_id: str
    user_id: str
    current_query: str

    # Response — use reducers for parallel-safe writes
    response: Annotated[str, _last_value]
    agent_type: Annotated[str, _last_value]
    suggestions: Annotated[list[str], operator.add]
    citations: Annotated[list[dict[str, Any]], operator.add]

    # Routing (for orchestrators)
    routing_decision: Annotated[str, _last_value]

    # Context — arbitrary data from frontend/caller for agent code to consume
    context: Annotated[dict[str, Any], _merge_dicts]
    prompt_version: Annotated[str, _last_value]

    # Processing
    steps_taken: Annotated[list[str], operator.add]
    metadata: Annotated[dict[str, Any], _merge_dicts]
    error: Annotated[str | None, _last_value]
