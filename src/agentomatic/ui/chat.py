"""Chainlit chat application for agentomatic debug UI.

This file is the Chainlit "target" mounted by the platform.
It provides:
- Agent selection from the registry
- Real-time streaming responses
- Tool call visualization
- Intermediate step display
- Feedback collection
"""

from __future__ import annotations

import json
import time
from typing import Any

try:
    import chainlit as cl

    HAS_CHAINLIT = True
except ImportError:
    HAS_CHAINLIT = False

if HAS_CHAINLIT:
    # Default API base — overridden by AGENTOMATIC_API_URL env var
    import os

    import httpx

    API_BASE = os.getenv("AGENTOMATIC_API_URL", "http://localhost:8000")
    API_PREFIX = os.getenv("AGENTOMATIC_API_PREFIX", "/api/v1")

    async def _fetch_agents() -> dict[str, Any]:
        """Fetch available agents from the platform API."""
        async with httpx.AsyncClient(base_url=API_BASE, timeout=10) as client:
            resp = await client.get(f"{API_PREFIX}/agents")
            resp.raise_for_status()
            return resp.json().get("agents", {})

    async def _invoke_agent(
        agent_name: str, query: str, thread_id: str | None = None
    ) -> dict[str, Any]:
        """Invoke an agent via the platform API."""
        async with httpx.AsyncClient(base_url=API_BASE, timeout=60) as client:
            payload = {
                "query": query,
                "user_id": "debug-ui",
                "thread_id": thread_id,
            }
            resp = await client.post(
                f"{API_PREFIX}/{agent_name}/invoke",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    @cl.on_chat_start
    async def on_start():
        """Initialize the chat session."""
        # Fetch available agents
        try:
            agents = await _fetch_agents()
        except Exception:
            agents = {}

        agent_names = list(agents.keys())

        if not agent_names:
            await cl.Message(
                content="⚠️ No agents found. Make sure the platform is running with agents registered.",
            ).send()
            return

        # Store agents info
        cl.user_session.set("agents", agents)
        cl.user_session.set("agent_names", agent_names)

        # Let user pick an agent
        settings = await cl.ChatSettings(
            [
                cl.input_widget.Select(
                    id="agent",
                    label="🤖 Agent",
                    values=agent_names,
                    initial_value=agent_names[0],
                ),
            ]
        ).send()

        selected = settings.get("agent", agent_names[0])
        cl.user_session.set("selected_agent", selected)

        # Welcome message
        agent_info = agents.get(selected, {})
        welcome = (
            f"## 🤖 {selected}\n\n"
            f"**Description:** {agent_info.get('description', 'N/A')}\n\n"
            f"**Version:** {agent_info.get('version', 'N/A')}\n\n"
            f"**Framework:** {agent_info.get('framework', 'N/A')}\n\n"
            f"---\n\n"
            f"Type a message to start chatting!"
        )
        await cl.Message(content=welcome).send()

    @cl.on_settings_update
    async def on_settings_update(settings: dict):
        """Handle agent selection change."""
        selected = settings.get("agent")
        if selected:
            cl.user_session.set("selected_agent", selected)
            agents = cl.user_session.get("agents", {})
            info = agents.get(selected, {})
            await cl.Message(
                content=f"Switched to **{selected}** — {info.get('description', '')}",
            ).send()

    @cl.on_message
    async def on_message(message: cl.Message):
        """Process user message."""
        agent_name = cl.user_session.get("selected_agent")
        if not agent_name:
            await cl.Message(content="Please select an agent first.").send()
            return

        # Show thinking indicator
        msg = cl.Message(content="")
        await msg.send()

        t0 = time.perf_counter()

        try:
            # Create a step for the API call
            async with cl.Step(name=f"invoke/{agent_name}", type="tool") as step:
                step.input = json.dumps({"query": message.content, "agent": agent_name}, indent=2)

                result = await _invoke_agent(agent_name, message.content)

                step.output = json.dumps(result, indent=2, default=str)

            duration = (time.perf_counter() - t0) * 1000

            # Display response
            response_text = result.get("response", str(result))

            # Show metadata as steps
            steps_taken = result.get("steps_taken", [])
            if steps_taken:
                async with cl.Step(name="Chain of Thought", type="llm") as cot_step:
                    cot_step.output = "\n".join(f"→ {s}" for s in steps_taken)

            citations = result.get("citations", [])
            if citations:
                async with cl.Step(name="Citations", type="retrieval") as cite_step:
                    cite_step.output = json.dumps(citations, indent=2)

            # Build final message
            footer = f"\n\n---\n*⏱ {duration:.0f}ms · 🤖 {result.get('agent_type', agent_name)}*"

            # Add suggestions as actions
            suggestions = result.get("suggestions", [])
            actions = []
            for i, s in enumerate(suggestions[:4]):
                actions.append(
                    cl.Action(
                        name=f"suggestion_{i}",
                        payload={"value": s},
                        label=s,
                    )
                )

            msg.content = response_text + footer
            msg.actions = actions
            await msg.update()

        except httpx.HTTPStatusError as exc:
            msg.content = f"❌ API Error: {exc.response.status_code} — {exc.response.text}"
            await msg.update()
        except httpx.ConnectError:
            msg.content = (
                "❌ Cannot connect to the platform API.\n\n"
                f"Make sure the server is running at `{API_BASE}`"
            )
            await msg.update()
        except Exception as exc:
            msg.content = f"❌ Error: {exc}"
            await msg.update()

    @cl.action_callback("suggestion_0")
    @cl.action_callback("suggestion_1")
    @cl.action_callback("suggestion_2")
    @cl.action_callback("suggestion_3")
    async def on_suggestion(action: cl.Action):
        """Handle suggestion button clicks."""
        value = action.payload.get("value", "")
        if value:
            fake_msg = cl.Message(content=value, author="user")
            await on_message(fake_msg)
