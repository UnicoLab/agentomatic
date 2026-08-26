"""Coordinator fixture that delegates through Agentomatic's protected API."""

from __future__ import annotations

import os
from typing import Any

import httpx

from agentomatic import AgentManifest

manifest = AgentManifest(
    name="coordinator",
    slug="coordinator",
    framework="custom",
    description="Delegates to researcher.",
    delegation_targets=["researcher"],
)


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Delegate the request and return the child result."""
    headers = {"X-Api-Key": os.environ["AGENTOMATIC_API_KEY"]}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=15) as client:
        response = await client.post(
            "/api/v1/researcher/invoke",
            headers=headers,
            json={"query": state.get("current_query", "")},
        )
    response.raise_for_status()
    return {"response": f"delegated:{response.json()['response']}", "agent_type": "coordinator"}
