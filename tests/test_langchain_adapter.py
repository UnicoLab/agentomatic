"""Comprehensive tests for agentomatic.langchain_adapter — covers ALL agent patterns."""

from __future__ import annotations

import pytest

# =====================================================================
# AgentAdapter — core functionality
# =====================================================================


def test_agent_adapter_base_graph_pass_through() -> None:
    """BaseGraphAgent-compatible agents pass through unchanged."""
    from agentomatic.langchain_adapter import AgentAdapter

    class CompatibleAgent:
        agent_name = "compat"
        agent_description = "already compatible"

        def build_graph(self):
            pass

        async def atransform(self, data):
            return {**data, "response": "ok"}

        def transform(self, data):
            return {**data, "response": "ok"}

    adapted = AgentAdapter(CompatibleAgent())
    assert adapted.detect_pattern() == "base_graph_agent"
    assert adapted.is_compatible


@pytest.mark.asyncio
async def test_agent_adapter_state_graph() -> None:
    """LangGraph StateGraph agent with ainvoke."""
    from agentomatic.langchain_adapter import AgentAdapter

    class StateGraphAgent:
        agent_name = "stategraph_bot"

        def __init__(self):
            self.graph = self

        async def ainvoke(self, state, config=None):
            msgs = list(state.get("messages", []))
            msgs.append({"role": "ai", "content": "Hello back!"})
            return {"messages": msgs}

    adapted = AgentAdapter(StateGraphAgent())
    assert adapted.detect_pattern() == "state_graph_agent"
    assert adapted.graph is not None

    result = await adapted.atransform({"current_query": "Hello"})
    assert "messages" in result


@pytest.mark.asyncio
async def test_agent_adapter_callable_agent() -> None:
    """Agent with ainvoke but no graph."""
    from agentomatic.langchain_adapter import AgentAdapter

    class CallableBot:
        agent_name = "callbot"

        async def ainvoke(self, state, config=None):
            msgs = state.get("messages", [])
            q = ""
            for m in msgs:
                c = getattr(m, "content", "") or ""
                if c:
                    q = str(c)
            return {"response": f"Got: {q}"}

    adapted = AgentAdapter(CallableBot())
    assert adapted.detect_pattern() == "callable_agent"

    result = await adapted.atransform({"current_query": "test"})
    assert result["response"] == "Got: test"


@pytest.mark.asyncio
async def test_agent_adapter_sync_invoke() -> None:
    """Agent with only sync invoke."""
    from agentomatic.langchain_adapter import AgentAdapter

    class SyncBot:
        agent_name = "syncbot"

        def invoke(self, state):
            return {"response": f"Sync: {state.get('current_query', '')}"}

    adapted = AgentAdapter(SyncBot())
    assert adapted.is_compatible

    result = await adapted.atransform({"current_query": "hello"})
    assert "Sync: hello" in str(result.get("response", ""))


def test_agent_adapter_detect_unknown() -> None:
    """Unrecognized agent returns 'unknown'."""
    from agentomatic.langchain_adapter import AgentAdapter

    class Unknown:
        pass

    adapted = AgentAdapter(Unknown())
    assert adapted.detect_pattern() == "unknown"
    assert not adapted.is_compatible


def test_agent_adapter_properties() -> None:
    """AgentAdapter exposes agent metadata."""
    from agentomatic.langchain_adapter import AgentAdapter

    class Bot:
        agent_name = "mybot"
        agent_description = "A test bot"
        agent_framework = "custom"

        async def ainvoke(self, state, config=None):
            return state

    adapted = AgentAdapter(Bot())
    assert adapted.agent_name == "mybot"
    assert adapted.agent_description == "A test bot"
    assert adapted.agent_framework == "custom"


def test_agent_adapter_name_override() -> None:
    """agent_name can be overridden in constructor."""
    from agentomatic.langchain_adapter import AgentAdapter

    class Bot:
        async def ainvoke(self, state, config=None):
            return state

    adapted = AgentAdapter(Bot(), agent_name="override")
    assert adapted.agent_name == "override"


def test_agent_adapter_repr() -> None:
    """repr is informative."""
    from agentomatic.langchain_adapter import AgentAdapter

    class Bot:
        agent_name = "reprbot"
        graph = "fake_graph"

        async def ainvoke(self, state, config=None):
            return state

    adapted = AgentAdapter(Bot())
    r = repr(adapted)
    assert "reprbot" in r
    assert "state_graph_agent" in r


def test_agent_adapter_detect_pattern_names() -> None:
    """All known patterns are detected correctly."""
    from agentomatic.langchain_adapter import AgentAdapter

    class BaseGraph:
        agent_name = "bg"

        def build_graph(self):
            pass

        async def atransform(self, d):
            return d

        def transform(self, d):
            return d

    class StateGraph:
        agent_name = "sg"
        graph = "g"

        async def ainvoke(self, s, c=None):
            return s

    class CallableBot:
        agent_name = "cb"

        async def ainvoke(self, s, c=None):
            return s

    class Unknown:
        agent_name = "unk"

    assert AgentAdapter(BaseGraph()).detect_pattern() == "base_graph_agent"
    assert AgentAdapter(StateGraph()).detect_pattern() == "state_graph_agent"
    assert AgentAdapter(CallableBot()).detect_pattern() == "callable_agent"
    assert AgentAdapter(Unknown()).detect_pattern() == "unknown"


# =====================================================================
# Message conversion (dict ↔ LangChain)
# =====================================================================


def test_dict_to_messages_basic() -> None:
    from agentomatic.langchain_adapter import dict_to_messages

    msgs = dict_to_messages({"current_query": "Hello"})
    assert len(msgs) == 1
    content = getattr(msgs[0], "content", "") or ""
    assert "Hello" in str(content)


def test_dict_to_messages_with_history() -> None:
    from agentomatic.langchain_adapter import dict_to_messages

    state = {
        "current_query": "New",
        "messages": [
            {"role": "user", "content": "Old"},
            {"role": "ai", "content": "Old answer"},
        ],
    }
    msgs = dict_to_messages(state)
    assert len(msgs) >= 3


def test_messages_to_dict_basic() -> None:
    from agentomatic.langchain_adapter import dict_to_messages, messages_to_dict

    orig = {"current_query": "Hi", "thread_id": "t1"}
    msgs = dict_to_messages(orig)
    result = messages_to_dict(msgs, orig)
    assert "current_query" in result
    assert "messages" in result


def test_dict_to_messages_no_query() -> None:
    from agentomatic.langchain_adapter import dict_to_messages

    msgs = dict_to_messages({})
    assert len(msgs) == 0


def test_dict_to_messages_accepts_list() -> None:
    from agentomatic.langchain_adapter import dict_to_messages, serialize_messages

    msgs = dict_to_messages(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )
    assert len(msgs) == 2
    plain = serialize_messages(msgs)
    assert plain[0]["role"] in ("user", "human")
    assert plain[0]["content"] == "Hi"


# =====================================================================
# RunnableConfig
# =====================================================================


def test_make_config_defaults() -> None:
    from agentomatic.langchain_adapter import make_config

    cfg = make_config()
    assert cfg["recursion_limit"] == 25


def test_make_config_full() -> None:
    from agentomatic.langchain_adapter import make_config

    cfg = make_config(thread_id="t42", tags=["prod"], recursion_limit=50)
    assert cfg["configurable"]["thread_id"] == "t42"
    assert cfg["tags"] == ["prod"]
    assert cfg["recursion_limit"] == 50


def test_inject_config() -> None:
    from agentomatic.langchain_adapter import inject_config

    state = {"a": 1}
    result = inject_config(state, thread_id="abc")
    assert result["a"] == 1
    assert result["runnable_config"]["configurable"]["thread_id"] == "abc"


# =====================================================================
# Chain detection & wrapping
# =====================================================================


def test_is_chain_true() -> None:
    from agentomatic.langchain_adapter import is_chain

    class Chain:
        def invoke(self, x):
            return x

        async def ainvoke(self, x):
            return x

        def stream(self, x):
            yield x

    assert is_chain(Chain()) is True


def test_is_chain_false() -> None:
    from agentomatic.langchain_adapter import is_chain

    assert is_chain("string") is False
    assert is_chain(42) is False
    assert is_chain(None) is False


def test_wrap_chain_as_node() -> None:
    from agentomatic.langchain_adapter import wrap_chain_as_node

    class FakeChain:
        def invoke(self, x):
            return f"Result: {x.get('query', '')}"

    node = wrap_chain_as_node(FakeChain(), output_key="answer")
    result = node({"current_query": "test"})
    assert result["answer"] == "Result: test"


@pytest.mark.asyncio
async def test_wrap_chain_as_async_node() -> None:
    from agentomatic.langchain_adapter import wrap_chain_as_async_node

    class FakeChain:
        async def ainvoke(self, x):
            return f"Async: {x.get('query', '')}"

    node = wrap_chain_as_async_node(FakeChain())
    result = await node({"current_query": "hello"})
    assert "Async: hello" in result["response"]


def test_wrap_chain_with_custom_mapper() -> None:
    from agentomatic.langchain_adapter import wrap_chain_as_node

    class FakeChain:
        def invoke(self, x):
            return str(x)

    def my_mapper(state):
        return {"custom": state.get("current_query", "")}

    node = wrap_chain_as_node(FakeChain(), input_mapper=my_mapper, output_key="out")
    result = node({"current_query": "x"})
    assert "custom" in result["out"]


# =====================================================================
# Prompt template utilities
# =====================================================================


def test_extract_system_prompt_string() -> None:
    from agentomatic.langchain_adapter import extract_system_prompt

    assert extract_system_prompt("You are a bot.") == "You are a bot."
    assert extract_system_prompt(None, default="fallback") == "fallback"
    assert extract_system_prompt("", default="fallback") == ""
    assert extract_system_prompt(42, default="fallback") == "fallback"


def test_extract_system_prompt_attribute() -> None:
    from agentomatic.langchain_adapter import extract_system_prompt

    class A:
        system_prompt = "from attribute"

    assert extract_system_prompt(A()) == "from attribute"

    class B:
        system_message = "from msg attr"

    assert extract_system_prompt(B()) == "from msg attr"


def test_inject_system_prompt_no_messages() -> None:
    from agentomatic.langchain_adapter import inject_system_prompt

    # Object without .messages — returned as-is
    obj = object()
    assert inject_system_prompt(obj, "new") is obj


def test_resolve_prompt_attribute() -> None:
    from agentomatic.langchain_adapter import resolve_prompt

    class Agent:
        agent_name = "a"
        system_prompt = "direct"

    assert resolve_prompt(Agent()) == "direct"


def test_resolve_prompt_method() -> None:
    from agentomatic.langchain_adapter import resolve_prompt

    class Agent:
        agent_name = "a"

        def _system_prompt(self):
            return "from method"

    assert resolve_prompt(Agent()) == "from method"


def test_resolve_prompt_fallback() -> None:
    from agentomatic.langchain_adapter import resolve_prompt

    class Agent:
        agent_name = "a"

    assert resolve_prompt(Agent(), default="fallback") == "fallback"


# =====================================================================
# Tool support
# =====================================================================


def test_tools_to_names() -> None:
    from agentomatic.langchain_adapter import tools_to_names

    class T1:
        name = "search"

    class T2:
        name = "calculate"

    names = tools_to_names([T1(), T2()])
    assert names == ["search", "calculate"]


def test_tools_to_names_empty() -> None:
    from agentomatic.langchain_adapter import tools_to_names

    assert tools_to_names([]) == []


def test_bind_tools_available() -> None:
    from agentomatic.langchain_adapter import bind_tools

    class LLM:
        def bind_tools(self, tools):
            self._bound = tools
            return self

    llm = LLM()
    result = bind_tools(llm, ["t1"])
    assert hasattr(result, "_bound")


def test_bind_tools_unavailable() -> None:
    from agentomatic.langchain_adapter import bind_tools

    class NoBind:
        pass

    llm = NoBind()
    result = bind_tools(llm, ["t1"])
    assert result is llm


# =====================================================================
# adapt_langgraph_agent
# =====================================================================


@pytest.mark.asyncio
async def test_adapt_langgraph_agent_already_compatible() -> None:
    from agentomatic.langchain_adapter import adapt_langgraph_agent

    class CompatAgent:
        agent_name = "already"

        def transform(self, data):
            return {**data, "response": "ok"}

    adapted = adapt_langgraph_agent(CompatAgent())
    assert adapted is not None
    assert adapted.agent_name == "already"


@pytest.mark.asyncio
async def test_adapt_langgraph_agent_minimal() -> None:
    from agentomatic.langchain_adapter import adapt_langgraph_agent

    class MinimalAgent:
        agent_name = "minimal"

        async def ainvoke(self, state, config=None):
            return {"response": "done"}

    adapted = adapt_langgraph_agent(MinimalAgent())
    assert adapted.agent_name == "minimal"
    result = await adapted.atransform({"current_query": "q"})
    assert result["response"] == "done"


# =====================================================================
# collect_stream
# =====================================================================


@pytest.mark.asyncio
async def test_collect_stream() -> None:
    from agentomatic.langchain_adapter import collect_stream

    async def fake_stream():
        yield {"messages": [{"role": "user", "content": "hi"}]}
        yield {"messages": [{"role": "ai", "content": "hello"}]}
        yield {"response": "final"}

    result = await collect_stream(fake_stream())
    assert len(result.get("messages", [])) >= 1


# =====================================================================
# Integration: AgentAdapter with message conversion
# =====================================================================


@pytest.mark.asyncio
async def test_agent_adapter_full_flow() -> None:
    """Full flow: create agent → adapt → transform → get response."""
    from agentomatic.langchain_adapter import AgentAdapter

    class ProductionAgent:
        agent_name = "prod_bot"
        agent_description = "A production agent using LangGraph"

        def __init__(self):
            self.graph = self

        async def ainvoke(self, state, config=None):
            msgs = list(state.get("messages", []))
            return {"messages": msgs, "response": "Answer from production agent"}

    adapted = AgentAdapter(ProductionAgent())
    assert adapted.agent_name == "prod_bot"
    assert adapted.detect_pattern() == "state_graph_agent"
    assert adapted.is_compatible

    result = await adapted.atransform({"current_query": "What is Python?"})
    assert "messages" in result


@pytest.mark.asyncio
async def test_agent_adapter_system_prompt_management() -> None:
    """get_system_prompt and set_system_prompt work."""
    from agentomatic.langchain_adapter import AgentAdapter

    class Agent:
        agent_name = "spm"
        system_prompt = "original"

        async def ainvoke(self, s, c=None):
            return s

    adapted = AgentAdapter(Agent())
    assert adapted.get_system_prompt() == "original"

    adapted.set_system_prompt("updated")
    assert adapted._agent.system_prompt == "updated"


@pytest.mark.asyncio
async def test_agent_adapter_streaming() -> None:
    """AgentAdapter supports streaming."""
    from agentomatic.langchain_adapter import AgentAdapter

    class StreamingAgent:
        agent_name = "streamer"

        async def astream(self, state, config=None):
            for i in range(3):
                yield {"step": i}

    adapted = AgentAdapter(StreamingAgent())
    events = []
    async for event in adapted.astream({"current_query": "test"}):
        events.append(event)
    assert len(events) == 3
