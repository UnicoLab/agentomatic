"""Node functions for hello agent."""

from __future__ import annotations

from typing import Any


async def greet(state: dict[str, Any]) -> dict[str, Any]:
    """Greet the user with a friendly message."""
    query = state.get("current_query", "")
    return {
        "response": f"👋 Hello from Agentomatic! You said: {query}",
        "agent_type": "agentomatic-hello",
        "suggestions": ["Tell me more", "What can you do?"],
    }
