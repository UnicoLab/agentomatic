# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""End-to-end tests for class-based agents built on LangChain abstractions.

These exercise a ``BaseGraphAgent`` subclass that uses the full LangChain
surface a real user would reach for — ``ChatPromptTemplate`` with
``MessagesPlaceholder``, an LCEL chain (``prompt | llm``), real
``HumanMessage``/``AIMessage``/``SystemMessage``/``ToolMessage`` objects in
state, ``@tool``-decorated tools, and an explicit ``RunnableConfig`` — served
through the *actual* platform HTTP stack (REST invoke / chat / SSE stream /
Studio debug API), not just called directly in-process.

The LLM is a real LangChain ``FakeListChatModel`` runnable, so the chain
composes and executes for real (no mocking of LangChain itself) while staying
deterministic and offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.agents import BaseGraphAgent
from agentomatic.langchain_adapter import dict_to_messages, make_config, serialize_messages

# =====================================================================
# A realistic LangChain-native class agent
# =====================================================================


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its id."""
    return f"order {order_id} is shipped"


@dataclass
class SupportState:
    """State carrying real LangChain message objects."""

    request: str = ""
    thread_id: str | None = None
    # NOTE: raw LangChain BaseMessage objects live here — the framework must
    # serialize these safely on every outbound path.
    messages: list[Any] = field(default_factory=list)
    response: str = ""
    used_config: dict[str, Any] = field(default_factory=dict)


class SupportAgent(BaseGraphAgent[SupportState]):
    """Class agent using ChatPromptTemplate + MessagesPlaceholder + LCEL."""

    agent_name = "support"
    agent_description = "LangChain-native support agent"
    agent_framework = "graph_agent"

    def __init__(self, *, llm: Any = None) -> None:
        super().__init__()
        self.llm = llm
        self.tools = [lookup_order]
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                MessagesPlaceholder("messages"),
            ]
        )
        self.chain = self.prompt_template | self.llm if self.llm is not None else None

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: SupportState) -> SupportState:
        lc_messages = (
            dict_to_messages(state.messages)
            if state.messages
            else dict_to_messages({"current_query": state.request})
        )
        config = make_config(thread_id=state.thread_id, tags=["support"])
        # Record the RunnableConfig so a test can assert it was threaded through.
        state.used_config = dict(config)

        result = self.chain.invoke(
            {"system_message": "You are a support agent.", "messages": lc_messages},
            config=config,
        )
        text = getattr(result, "content", None) or str(result)

        # Deliberately leave RAW BaseMessage objects in state — the framework
        # must make these JSON-safe on the REST/SSE/Studio paths.
        state.messages = [*lc_messages, AIMessage(content=text)]
        state.response = text
        return state

    def input_to_state(self, data: dict[str, Any]) -> SupportState:
        return SupportState(
            request=data.get("current_query", ""),
            messages=data.get("messages", []) or [],
            thread_id=data.get("thread_id"),
        )

    def state_to_output(self, state: SupportState) -> dict[str, Any]:
        return {
            "response": state.response,
            "messages": state.messages,  # raw BaseMessage objects on purpose
            "used_config": state.used_config,
        }


def _make_agent(replies: list[str] | None = None) -> SupportAgent:
    llm = FakeListChatModel(responses=replies or ["Hello from the fake LLM"])
    return SupportAgent(llm=llm)


@pytest.fixture
def client(tmp_path):
    agent = _make_agent()
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        title="LangChain E2E",
        enable_studio=True,
    )
    platform.register_agent(
        manifest=AgentManifest(
            name="support",
            slug="support",
            description="LangChain-native support agent",
        ),
        class_instance=agent,
    )
    with TestClient(platform.build()) as c:
        yield c


# =====================================================================
# In-process: the LangChain abstractions really execute
# =====================================================================


def test_lcel_chain_executes_and_threads_runnable_config() -> None:
    """prompt | llm composes and runs, and a RunnableConfig reaches the chain."""
    agent = _make_agent(["Order is on the way"])
    state = agent.input_to_state({"current_query": "where is my order?", "thread_id": "t-1"})
    out = agent.respond(state)

    assert out.response == "Order is on the way"
    # The RunnableConfig carried the thread_id through for tracing/checkpointing.
    assert out.used_config["configurable"]["thread_id"] == "t-1"
    assert "support" in out.used_config["tags"]
    # Real message objects, not dicts/strings.
    assert isinstance(out.messages[0], HumanMessage)
    assert isinstance(out.messages[-1], AIMessage)


def test_message_placeholder_receives_full_history() -> None:
    """MessagesPlaceholder("messages") must receive prior turns, not just the query."""
    agent = _make_agent(["ack"])
    state = agent.input_to_state(
        {
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        }
    )
    out = agent.respond(state)

    types = [type(m) for m in out.messages[:3]]
    assert types == [SystemMessage, HumanMessage, AIMessage]


def test_tool_messages_round_trip_with_tool_call_id() -> None:
    """ToolMessage/AIMessage tool-call metadata survives the state round-trip."""
    agent = _make_agent(["done"])
    state = agent.input_to_state(
        {
            "messages": [
                {"role": "user", "content": "check order 42"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "lookup_order", "args": {"order_id": "42"}, "id": "call_1"}
                    ],
                },
                {
                    "role": "tool",
                    "content": "order 42 is shipped",
                    "tool_call_id": "call_1",
                    "name": "lookup_order",
                },
            ]
        }
    )
    out = agent.respond(state)

    tool_msg = next(m for m in out.messages if isinstance(m, ToolMessage))
    assert tool_msg.tool_call_id == "call_1"
    ai_with_calls = next(
        m for m in out.messages if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )
    assert ai_with_calls.tool_calls[0]["id"] == "call_1"


# =====================================================================
# Over real HTTP: REST invoke / chat / SSE stream
# =====================================================================


def test_http_invoke_serializes_raw_langchain_messages(client) -> None:
    """state_to_output() returning raw BaseMessage objects must not break the
    JSON response — they become plain role/content dicts.
    """
    resp = client.post("/api/v1/support/invoke", json={"query": "hi"})
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["response"] == "Hello from the fake LLM"

    messages = body["output"]["messages"]
    assert messages == [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "Hello from the fake LLM"},
    ]
    # No Python reprs leaked into the payload.
    assert "HumanMessage(" not in resp.text
    assert "additional_kwargs" not in resp.text


def test_http_invoke_response_is_valid_json_end_to_end(client) -> None:
    """The whole envelope must be re-parseable (no stringified objects)."""
    resp = client.post("/api/v1/support/invoke", json={"query": "hi"})
    reparsed = json.loads(resp.content)
    assert isinstance(reparsed["output"]["messages"], list)
    assert all(isinstance(m, dict) for m in reparsed["output"]["messages"])


def test_http_chat_creates_thread_and_returns_text(client) -> None:
    resp = client.post("/api/v1/support/chat", json={"content": "hello"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["thread_id"]
    assert body["response"] == "Hello from the fake LLM"


def test_http_sse_stream_emits_jsonable_messages(client) -> None:
    """SSE frames must be valid JSON with serialized messages."""
    resp = client.post("/api/v1/support/invoke/stream", json={"query": "stream me"})
    assert resp.status_code == 200, resp.text

    frames = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line.strip() != "data: [DONE]"
    ]
    assert frames, f"no SSE frames parsed from: {resp.text[:400]}"

    node_frame = next(f for f in frames if "respond" in f)
    streamed = node_frame["respond"]["messages"]
    assert {"role": "human", "content": "stream me"} in streamed
    assert "[DONE]" in resp.text


def test_http_chat_preserves_tool_call_fidelity(client) -> None:
    """A caller-supplied tool-calling history on /chat must reach the agent
    (and come back) with ``tool_calls`` and ``tool_call_id`` intact.

    Losing either breaks the call/result pairing that OpenAI/Anthropic
    require on the following turn.
    """
    resp = client.post(
        "/api/v1/support/chat",
        json={
            "content": "check it",
            "messages": [
                {"role": "user", "content": "check order 42"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "lookup_order", "args": {"order_id": "42"}, "id": "call_1"}
                    ],
                },
                {
                    "role": "tool",
                    "content": "order 42 is shipped",
                    "tool_call_id": "call_1",
                    "name": "lookup_order",
                },
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    messages = resp.json()["output"]["messages"]
    tool_entry = next(m for m in messages if m["role"] == "tool")
    assert tool_entry["tool_call_id"] == "call_1"
    assert tool_entry["name"] == "lookup_order"

    ai_entry = next(m for m in messages if m.get("tool_calls"))
    assert ai_entry["tool_calls"][0]["id"] == "call_1"


def test_http_chat_forwards_history_and_thread_id_to_input_to_state(client) -> None:
    """``messages`` and ``thread_id`` must actually reach ``input_to_state``.

    They used to be stripped as "conversation bookkeeping", which made the
    scaffolded LangChain template's history/thread handling dead code on
    every HTTP path.
    """
    resp = client.post(
        "/api/v1/support/chat",
        json={
            "content": "second turn",
            "messages": [{"role": "user", "content": "first turn"}],
        },
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    # thread_id reached the agent and was threaded into the RunnableConfig.
    assert body["output"]["used_config"]["configurable"]["thread_id"] == body["thread_id"]
    # Prior turn was visible to the agent (so MessagesPlaceholder sees it).
    contents = [m["content"] for m in body["output"]["messages"]]
    assert "first turn" in contents
    assert "second turn" in contents


# =====================================================================
# Studio debug API works for a LangChain class agent
# =====================================================================


def test_studio_lists_and_introspects_langchain_class_agent(client) -> None:
    listing = client.get("/studio/agents")
    assert listing.status_code == 200, listing.text
    assert any(a["name"] == "support" for a in listing.json())

    graph = client.get("/studio/agents/support/graph")
    assert graph.status_code == 200, graph.text
    node_ids = {n["id"] for n in graph.json()["nodes"]}
    assert "respond" in node_ids

    schemas = client.get("/studio/agents/support/schemas")
    assert schemas.status_code == 200, schemas.text


def test_openapi_is_valid_with_langchain_class_agent(client) -> None:
    """A bad response_model would make /openapi.json 500 — guard against it."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200, resp.text
    spec = resp.json()
    assert "/api/v1/support/invoke" in spec["paths"]


# =====================================================================
# Serialization helper used by the scaffolded template
# =====================================================================


def test_serialize_messages_matches_rest_representation() -> None:
    """The helper the scaffold uses produces the same shape the REST layer emits."""
    msgs = [HumanMessage(content="hi"), AIMessage(content="yo")]
    assert serialize_messages(msgs) == [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "yo"},
    ]
