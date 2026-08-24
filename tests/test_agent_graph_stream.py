# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
import pytest

from agentomatic.agents.graph import AgentGraph, GraphNode


@pytest.mark.asyncio
async def test_agent_graph_astream():
    def node_a(state):
        state["path"].append("a")
        return state

    async def node_b(state):
        state["path"].append("b")
        return state

    graph = AgentGraph(
        nodes={"a": GraphNode(name="a", handler=node_a), "b": GraphNode(name="b", handler=node_b)},
        edges={"a": "b"},
        entrypoint="a",
        finish="b",
    )

    state = {"path": []}
    streamed_events = []

    async for event in graph.astream(state):
        streamed_events.append(event)

    assert len(streamed_events) == 2
    assert "a" in streamed_events[0]
    assert streamed_events[0]["a"]["path"] == ["a"]

    assert "b" in streamed_events[1]
    assert streamed_events[1]["b"]["path"] == ["a", "b"]


@pytest.mark.asyncio
async def test_agent_graph_astream_studio_events():
    def node_a(state):
        state["path"].append("a")
        return state

    graph = AgentGraph(
        nodes={
            "a": GraphNode(name="a", handler=node_a),
        },
        edges={},
        entrypoint="a",
        finish="a",
    )

    state = {"path": []}
    events = []

    async for event in graph.astream_studio_events(state, run_id="test_123"):
        events.append(event)

    event_types = [e["event"] for e in events]
    assert "run_start" in event_types
    assert "node_start" in event_types
    assert "node_end" in event_types
    assert "state_update" in event_types
    assert "run_complete" in event_types


@pytest.mark.asyncio
async def test_agent_graph_astream_serializes_langchain_messages():
    """State containing raw BaseMessage objects must stream as JSON-safe dicts.

    A class-agent node commonly does ``state.messages = [HumanMessage(...), ...]``.
    The streamed event payloads must be safe to ``json.dumps`` (no BaseMessage
    objects left over) so SSE/Studio consumers don't crash or get stringified reprs.
    """
    import json

    from langchain_core.messages import AIMessage, HumanMessage

    def node_a(state):
        state["messages"] = [HumanMessage(content="hi"), AIMessage(content="hello")]
        return state

    graph = AgentGraph(
        nodes={"a": GraphNode(name="a", handler=node_a)},
        edges={},
        entrypoint="a",
        finish="a",
    )

    events = []
    async for event in graph.astream_studio_events({"messages": []}, run_id="test_lc"):
        events.append(event)

    state_update = next(e for e in events if e["event"] == "state_update")
    messages = state_update["data"]["messages"]
    assert messages == [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "hello"},
    ]
    # Must be JSON-serializable without a `default=` fallback.
    json.dumps(state_update)
