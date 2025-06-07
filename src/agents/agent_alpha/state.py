"""State management for Alpha Agent."""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AlphaAgentState(TypedDict):
    """State schema for Alpha Agent workflow."""

    # Message history
    messages: List[BaseMessage]

    # Context and step tracking
    context: str
    current_step: str

    # Input/Output processing
    processed_input: Optional[str]
    analysis_result: Optional[str]
    final_response: Optional[str]

    # Completion and error handling
    completed: bool
    error: Optional[str]

    # Metadata
    metadata: Optional[Dict[str, Any]]