"""Default agent state for LangGraph-based agents."""

from __future__ import annotations

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


class BaseAgentState(TypedDict, total=False):
    """Default state shared across all agents.

    Users can subclass or replace this with their own TypedDict.
    All fields are optional (``total=False``) for maximum flexibility.
    """

    messages: Annotated[list, add_messages]
    thread_id: str
    user_id: str
    current_query: str

    # Response
    response: str
    agent_type: str
    suggestions: list[str]
    citations: list[dict[str, Any]]

    # Routing (for orchestrators)
    routing_decision: str

    # Processing
    steps_taken: list[str]
    metadata: dict[str, Any]
    error: str | None
