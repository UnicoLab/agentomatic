"""Structured AgentInvokeResponse coercion for class-agent state_to_output."""

from __future__ import annotations

import json

from agentomatic.core.router_factory import AgentInvokeResponse, coerce_agent_invoke_payload


def test_coerce_structured_dict_without_response_key() -> None:
    """Class agents return dicts without a response string."""
    result = {
        "content": "Bonjour",
        "next_action": "Continuer",
        "estimated_days": 12,
    }
    text, output, context = coerce_agent_invoke_payload(result)
    assert text == "Bonjour"
    assert output == result
    assert context == {}
    envelope = AgentInvokeResponse(
        response=text,
        output=output,
        agent_type="agent-assistant",
        duration_ms=1.0,
    )
    dumped = envelope.model_dump()
    assert dumped["output"]["estimated_days"] == 12
    assert dumped["response"] == "Bonjour"


def test_coerce_explicit_string_response() -> None:
    text, output, _ = coerce_agent_invoke_payload({"response": "plain", "agent_type": "x"})
    assert text == "plain"
    assert output is None


def test_coerce_json_fallback_when_no_human_text() -> None:
    payload = {"modules": [{"name": "A"}], "p50": 10}
    text, output, _ = coerce_agent_invoke_payload(payload)
    assert output == payload
    assert json.loads(text)["p50"] == 10
