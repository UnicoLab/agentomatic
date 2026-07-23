"""Helpers for Agentomatic ``AgentInvokeRequest`` payloads.

Agentomatic now preserves the full client payload and flattens ``context``
into the transform dict before ``input_to_state`` (see
``agentomatic.core.agent_invoke.build_invoke_state`` /
``_input_from_state``). This helper only aliases ``query`` /
``current_query`` onto ``question`` for chat-style agents.
"""

from __future__ import annotations

from typing import Any


def flatten_invoke_input(input_data: dict[str, Any] | None) -> dict[str, Any]:
    """Alias query fields onto ``question`` when absent.

    Context flattening is handled by Agentomatic; keep this thin alias so
    agents that read ``question`` continue to work for chat-style invokes.
    """
    data = dict(input_data or {})
    if not data.get("question"):
        q = data.get("current_query") or data.get("query")
        if q:
            data["question"] = q
    return data
