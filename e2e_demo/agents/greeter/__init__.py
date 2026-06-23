"""Greeter agent — a simple demo agent for E2E testing.

This agent auto-discovers from the folder structure and responds
with a friendly greeting, demonstrating:
  - Auto-discovery from agents folder
  - AgentManifest definition
  - node_fn for stateless invocation
  - graph_fn for LangGraph-style graph (optional)
  - /invoke, /chat, /health endpoints
  - Studio integration hooks
"""

from __future__ import annotations

import random
from typing import Any

from agentomatic import AgentManifest

# ── Manifest ────────────────────────────────────────────────────────
manifest = AgentManifest(
    name="greeter",
    slug="greeter",
    version="0.1.0",
    framework="custom",
    description="A friendly demo greeter agent for E2E testing",
    metadata={
        "author": "agentomatic-team",
        "tags": ["demo", "greeting", "e2e"],
    },
)


# ── Node function (required) ────────────────────────────────────────
GREETINGS = [
    "Hello there! 👋 I'm the Greeter Agent.",
    "Hey! Great to see you! 🎉",
    "Greetings, human! 🤖 How can I help?",
    "Welcome! I'm your friendly neighborhood AI. 🌟",
    "Hi! I'm running on agentomatic v0.4.1 — all systems go! 🚀",
]


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Process an incoming request and return a greeting.

    This is the main entry point for the agent. It receives the full
    state dict (with messages, metadata, etc.) and returns a response.
    """
    query = state.get("query", "")
    user_id = state.get("user_id", "anonymous")
    messages = state.get("messages", [])

    greeting = random.choice(GREETINGS)  # noqa: S311

    # Build the response
    response_text = f"{greeting}\n\nYou said: '{query}'\n"
    if messages:
        response_text += f"(I can see {len(messages)} messages in your history)\n"
    response_text += f"\n— with ❤️ from the greeter agent for user {user_id}"

    return {
        "response": response_text,
        "messages": messages,
        "metadata": {
            "agent": "greeter",
            "user_id": user_id,
            "greeting_used": greeting,
        },
    }


# ── Studio hooks (optional) ─────────────────────────────────────────
def studio_state_provider(thread_id: str | None = None) -> dict[str, Any]:
    """Return the current state for Studio inspection."""
    return {
        "greetings_available": len(GREETINGS),
        "agent_version": manifest.version,
        "thread_id": thread_id,
    }
