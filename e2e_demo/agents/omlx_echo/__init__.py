"""A real oMLX-backed agent for the local container verification stack.

Unlike the deterministic ``greeter`` fixture, this module proves that a
containerised Agentomatic deployment can reach the host's OpenAI-compatible
oMLX server and return a model-generated response through the normal agent
router.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from agentomatic import AgentManifest

manifest = AgentManifest(
    name="omlx_echo",
    slug="omlx_echo",
    version="1.0.0",
    framework="openai-compatible",
    description="Calls the local oMLX OpenAI-compatible endpoint from Docker.",
    metadata={"provider": "omlx", "purpose": "container e2e verification"},
)


def _model_name() -> str:
    """Return the upstream model id without Agentomatic's provider prefix."""
    model = os.getenv(
        "AGENTOMATIC_LIVE_MODEL",
        os.getenv("AGENTOMATIC_TASK_MODEL", "omlx/Qwen3.5-9B-MLX-4bit"),
    )
    return model.removeprefix("omlx/")


async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the configured local model and return its assistant message."""
    query = str(state.get("current_query") or state.get("query") or "Hello from Agentomatic")
    client = AsyncOpenAI(
        base_url=os.getenv("OMLX_BASE_URL", "http://host.docker.internal:8000/v1"),
        api_key=os.getenv("OMLX_API_KEY", "local"),
        timeout=45.0,
    )
    try:
        completion = await client.chat.completions.create(
            model=_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user directly and concisely. Do not reveal chain-of-thought.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=96,
            # Qwen-family models served by oMLX otherwise emit their private
            # reasoning preamble as the visible answer. This OpenAI-compatible
            # extension is harmless on servers that do not implement it.
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "enable_thinking": False,
            },
        )
        response = completion.choices[0].message.content or ""
    finally:
        await client.close()

    return {
        "response": response,
        "agent_type": manifest.name,
        "metadata": {"provider": "omlx", "model": _model_name()},
    }
