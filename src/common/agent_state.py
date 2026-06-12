"""Common agent state management."""

from typing import Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Common state schema for all agents."""

    # Core fields
    messages: list[BaseMessage]
    context: str
    agent_name: str

    # Processing fields
    classification: str | None
    output: str | None
    completed: bool
    error: str | None

    # Metadata
    metadata: dict[str, Any] | None
