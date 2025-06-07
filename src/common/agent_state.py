"""Common agent state management."""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Common state schema for all agents."""

    # Core fields
    messages: List[BaseMessage]
    context: str
    agent_name: str

    # Processing fields
    classification: Optional[str]
    output: Optional[str]
    completed: bool
    error: Optional[str]

    # Metadata
    metadata: Optional[Dict[str, Any]]