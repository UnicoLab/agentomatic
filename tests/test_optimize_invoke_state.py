# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Ensure /optimize/invoke flattens context.document like /invoke."""

from __future__ import annotations

from agentomatic.core.agent_invoke import _input_from_state, build_invoke_state
from agentomatic.core.router_factory import OptimizeInvokeRequest


def test_optimize_request_builds_state_with_document_context() -> None:
    """Regression: document must not be buried only under metadata."""
    req = OptimizeInvokeRequest(
        query="Extract parties and MOA",
        context={
            "document": "### MOA\n**SCI Les Berges du Rhône**\n",
            "document_text": "### MOA\n**SCI Les Berges du Rhône**\n",
            "project_id": "optimize-smoke",
        },
        include_retrieval_context=True,
        include_steps=True,
    )
    state = build_invoke_state(req, default_thread_id="opt_test")
    assert state["current_query"] == "Extract parties and MOA"
    assert isinstance(state.get("context"), dict)
    assert "SCI Les Berges" in state["context"]["document"]
    # metadata must NOT be the only home for document
    meta = state.get("metadata") or {}
    assert meta.get("document") is None or meta.get("document") == state["context"]["document"]

    payload = _input_from_state(state)
    assert "SCI Les Berges" in str(payload.get("document") or "")
    assert payload.get("project_id") == "optimize-smoke"
