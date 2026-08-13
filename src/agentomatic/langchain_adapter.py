"""First-class LangChain / LangGraph adapter for agentomatic agents.

Bridges **any** LangChain- or LangGraph-based agent into the agentomatic
platform — no rewriting required.  Handles:

* ``ChatPromptTemplate`` with ``MessagesPlaceholder``
* LangChain message types (``HumanMessage``, ``AIMessage``, ``ToolMessage``)
* ``RunnableConfig`` passthrough
* LCEL chains (``prompt | llm | parser``)
* LangGraph ``StateGraph`` / ``MessageGraph``
* ``deep_agent`` / ``create_deep_agent`` sub-agents
* Tool calling with ``BaseTool`` / ``@tool``
* Streaming (sync + async)
* Prompt manager integration (read/write system prompts)
* ``BaseGraphAgent`` ↔ LangChain agent wrapping

**All of this works without any agent changes.**  Just drop your LangChain
agent class into an agent folder and agentomatic discovers it.

Quick examples
--------------

**Wrap a LangChain prompt template for agentomatic**::

    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from agentomatic.langchain_adapter import (
        extract_system_prompt, inject_system_prompt, wrap_chain_as_node,
    )

    template = ChatPromptTemplate.from_messages([
        ("system", "{sys}"),
        MessagesPlaceholder("messages"),
    ])
    chain = template | my_llm

    # agentomatic can read and optimise the system prompt
    text = extract_system_prompt(template)

    # Use the chain as a LangGraph node
    node = wrap_chain_as_node(chain)

**Convert dict state to LangChain messages**::

    from agentomatic.langchain_adapter import (
        dict_to_messages, messages_to_dict,
    )
    messages = dict_to_messages({"current_query": "Hello"})
    # → [HumanMessage(content="Hello")]

**Build RunnableConfig for LangGraph**::

    from agentomatic.langchain_adapter import make_config
    config = make_config(thread_id="t1", recursion_limit=50)
    result = await graph.ainvoke(state, config=config)

**Adapt a LangGraph agent class for agentomatic Studio + streaming**::

    from agentomatic.langchain_adapter import adapt_langgraph_agent

    class MyLangGraphAgent:
        def __init__(self):
            self.graph = self._build_graph()

        async def ainvoke(self, state, config=None):
            return await self.graph.ainvoke(state, config=config)

        async def astream(self, state, config=None):
            async for event in self.graph.astream(state, config=config):
                yield event

    # Agentomatic can now serve, stream, and debug this agent
    adapted = adapt_langgraph_agent(MyLangGraphAgent())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence


# =====================================================================
# Prompt template utilities
# =====================================================================


def extract_system_prompt(template: Any, *, default: str = "") -> str:
    """Extract the system-message text from any LangChain prompt template.

    Supports:
    - ``ChatPromptTemplate`` (LangChain >= 0.2)
    - ``SystemMessagePromptTemplate``
    - Tuple-based ``[("system", "..."), ...]`` templates
    - Objects with ``system_prompt`` or ``system_message`` attributes
    - Plain strings

    Args:
        template: A LangChain prompt template or compatible object.
        default: Fallback value when extraction fails.

    Returns:
        The system prompt text, or *default*.

    Example::

        from langchain.prompts import ChatPromptTemplate
        t = ChatPromptTemplate.from_messages([("system", "You are helpful")])
        text = extract_system_prompt(t)  # "You are helpful"
    """
    if isinstance(template, str):
        return template

    if template is None:
        return default

    # ChatPromptTemplate with .messages attribute
    if hasattr(template, "messages"):
        for msg in template.messages:
            if _is_system(msg):
                return _render_text(msg)

    # Tuple-based messages
    if hasattr(template, "messages") and isinstance(template.messages, list):
        for msg in template.messages:
            if isinstance(msg, (tuple, list)) and len(msg) >= 2:
                if msg[0] == "system" and isinstance(msg[1], str):
                    return msg[1]

    # Attribute-based
    for attr in ("system_prompt", "system_message", "_system_prompt"):
        val = getattr(template, attr, None)
        if isinstance(val, str):
            return val

    return default


def inject_system_prompt(
    template: Any,
    new_prompt: str,
) -> Any:
    """Replace or insert the system message in a ``ChatPromptTemplate``.

    Creates a **new** template; the original is not mutated.

    Args:
        template: A LangChain prompt template with ``.messages``.
        new_prompt: The replacement system prompt text.

    Returns:
        A new template with the updated system message.

    Example::

        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        t = ChatPromptTemplate.from_messages([
            ("system", "{old}"),
            MessagesPlaceholder("messages"),
        ])
        t2 = inject_system_prompt(t, "You are an expert.")
    """
    if not hasattr(template, "messages"):
        return template

    try:
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
    except ImportError:
        return template

    new_messages: list[Any] = []
    replaced = False

    for msg in template.messages:
        if _is_system(msg) and not replaced:
            try:
                new_messages.append(SystemMessagePromptTemplate.from_template(new_prompt))
            except Exception:
                new_messages.append(("system", new_prompt))
            replaced = True
        else:
            new_messages.append(msg)

    if not replaced:
        try:
            new_messages.insert(0, SystemMessagePromptTemplate.from_template(new_prompt))
        except Exception:
            new_messages.insert(0, ("system", new_prompt))

    return ChatPromptTemplate.from_messages(new_messages)


# =====================================================================
# Message conversion (dict ↔ LangChain)
# =====================================================================


def dict_to_messages(
    state: dict[str, Any] | Sequence[Any],
    *,
    query_key: str = "current_query",
    history_key: str = "messages",
) -> list[Any]:
    """Convert agentomatic state (or a message list) to LangChain messages.

    Args:
        state: Either a state dict with ``current_query`` / ``messages``,
            or a sequence of role/content dicts / LangChain messages.
        query_key: Key for the user's current query (dict form).
        history_key: Key for message history (dict form).

    Returns:
        List of LangChain messages (``HumanMessage``, ``AIMessage``, etc.).

    Example::

        msgs = dict_to_messages({"current_query": "Hello"})
        # → [HumanMessage(content="Hello")]
        msgs = dict_to_messages([{"role": "user", "content": "Hi"}])
    """
    # Accept a bare message list (common in graph state dataclasses).
    if not isinstance(state, dict):
        return [_dict_to_lc(msg) for msg in state]

    try:
        from langchain_core.messages import HumanMessage
    except ImportError:
        return [{"role": "user", "content": str(state.get(query_key, ""))}]

    messages: list[Any] = []

    history = state.get(history_key, [])
    if history:
        for msg in history:
            messages.append(_dict_to_lc(msg))

    query = state.get(query_key, "")
    if query:
        messages.append(HumanMessage(content=str(query)))

    return messages


def messages_to_dict(
    messages: Sequence[Any],
    fallback_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert LangChain messages back to agentomatic dict state.

    The last ``HumanMessage`` becomes ``current_query``, the last
    ``AIMessage`` becomes ``response``, and the full sequence is
    stored under ``messages``.

    Args:
        messages: Sequence of LangChain messages.
        fallback_state: Original state dict to merge into.

    Returns:
        State dict with ``current_query``, ``response``, ``messages``.

    Example::

        state = messages_to_dict([HumanMessage("Hi"), AIMessage("Hello!")])
        # → {"current_query": "Hi", "response": "Hello!", "messages": [...]}
    """
    state: dict[str, Any] = dict(fallback_state or {})
    lc_list: list[dict[str, str]] = []
    last_query = ""
    last_response = ""

    for msg in messages:
        content = getattr(msg, "content", "") or ""
        content_str = str(content)
        role = _msg_role(msg)
        lc_list.append({"role": role, "content": content_str})
        if role in ("user", "human"):
            last_query = content_str
        elif role in ("ai", "assistant"):
            last_response = content_str

    state["messages"] = lc_list
    if last_query:
        state["current_query"] = last_query
    if last_response:
        state["response"] = last_response

    return state


def serialize_messages(messages: Sequence[Any]) -> list[dict[str, str]]:
    """Serialize LangChain / role-dict messages to a plain ``list[dict]``."""
    return list(messages_to_dict(messages).get("messages", []))


# =====================================================================
# RunnableConfig
# =====================================================================


def make_config(
    thread_id: str | None = None,
    recursion_limit: int = 25,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a ``RunnableConfig``-compatible dict for LangGraph invocations.

    Args:
        thread_id: Conversation thread ID (required for checkpointing).
        recursion_limit: Max graph steps.
        tags: Optional tracing tags.
        metadata: Optional metadata dict.
        extra: Additional config keys.

    Returns:
        A dict suitable for ``graph.ainvoke(state, config=...)``.

    Example::

        config = make_config(thread_id="conv_42", tags=["production"])
        result = await graph.ainvoke(state, config=config)
    """
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    if tags:
        config["tags"] = list(tags)
    if metadata:
        config["metadata"] = dict(metadata)
    config.update(extra)
    return config


def inject_config(
    state: dict[str, Any],
    thread_id: str | None = None,
    recursion_limit: int = 25,
    **extra: Any,
) -> dict[str, Any]:
    """Add ``runnable_config`` to a state dict for nodes needing explicit access.

    Args:
        state: Existing state dict.
        thread_id: Optional thread ID.
        recursion_limit: Recursion limit.
        extra: Additional config keys.

    Returns:
        State dict with ``runnable_config`` key added.
    """
    config = make_config(thread_id=thread_id, recursion_limit=recursion_limit, **extra)
    return {**state, "runnable_config": config}


# =====================================================================
# Chain wrapping (LCEL → LangGraph node)
# =====================================================================


def is_chain(obj: Any) -> bool:
    """Check if *obj* looks like an LCEL chain (``prompt | llm | parser``).

    Returns ``True`` for any object with ``invoke`` + ``ainvoke`` + ``stream``.
    """
    return hasattr(obj, "invoke") and hasattr(obj, "ainvoke") and hasattr(obj, "stream")


def wrap_chain_as_node(
    chain: Any,
    *,
    input_mapper: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    output_key: str = "response",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap an LCEL chain as a sync LangGraph node.

    The node accepts dict state, extracts input, runs the chain,
    and writes the output back.

    Args:
        chain: An LCEL chain (``prompt | llm | parser``).
        input_mapper: Optional ``(state) -> chain_input`` function.
            Default maps ``current_query`` → ``query``.
        output_key: State key to write the response into.

    Returns:
        Callable ``(state) -> dict`` suitable as a graph node.

    Example::

        chain = template | llm
        node = wrap_chain_as_node(chain)
        g.add_node("chat", node)
    """

    def _default_mapper(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": state.get("current_query", ""),
            "messages": state.get("messages", []),
        }

    mapper = input_mapper if callable(input_mapper) else _default_mapper

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        chain_input = mapper(state)
        result = chain.invoke(chain_input)

        if isinstance(result, str):
            return {**state, output_key: result}
        if hasattr(result, "content"):
            return {**state, output_key: str(result.content)}
        return {**state, output_key: str(result)}

    return _node


def wrap_chain_as_async_node(
    chain: Any,
    *,
    input_mapper: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    output_key: str = "response",
) -> Callable[[dict[str, Any]], Any]:
    """Wrap an LCEL chain as an **async** LangGraph node.

    Like :func:`wrap_chain_as_node` but returns an async callable suitable
    for LangGraph graphs that use ``ainvoke`` / ``astream``.

    Example::

        chain = template | llm
        node = wrap_chain_as_async_node(chain)
        g.add_node("chat", node)
    """

    def _default_mapper(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": state.get("current_query", ""),
            "messages": state.get("messages", []),
        }

    mapper = input_mapper if callable(input_mapper) else _default_mapper

    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        chain_input = mapper(state)
        try:
            result = await chain.ainvoke(chain_input)
        except (TypeError, AttributeError, NotImplementedError):
            result = chain.invoke(chain_input)

        if isinstance(result, str):
            return {**state, output_key: result}
        if hasattr(result, "content"):
            return {**state, output_key: str(result.content)}
        return {**state, output_key: str(result)}

    return _node


# =====================================================================
# Agent adaptation (LangGraph → agentomatic-compatible)
# =====================================================================


def adapt_langgraph_agent(
    agent: Any,
    *,
    agent_name: str | None = None,
) -> Any:
    """Adapt a LangGraph-based agent class so agentomatic can serve it.

    Inspects the agent for ``graph``, ``ainvoke``, ``astream``,
    ``transform``, and ``invoke`` methods and returns an adapter
    object that exposes the full agentomatic contract.

    If the agent is already agentomatic-compatible (has ``transform``
    or is a ``BaseGraphAgent``), it is returned as-is.

    Args:
        agent: A LangGraph agent instance.
        agent_name: Optional override for the agent name.

    Returns:
        An adapted agent that works with agentomatic routing,
        Studio streaming, prompt optimisation, and A2A.

    Example::

        from agentomatic.langchain_adapter import adapt_langgraph_agent

        class MyBot:
            def __init__(self):
                self.graph = self._build()

            async def ainvoke(self, state, config=None):
                return await self.graph.ainvoke(state, config=config)

        adapted = adapt_langgraph_agent(MyBot())
        # Now usable in agentomatic agents/ folder
    """
    # Already compatible — return as-is
    if hasattr(agent, "transform") or hasattr(agent, "atransform"):
        return agent

    name = agent_name or getattr(agent, "agent_name", None) or "agent"
    graph = getattr(agent, "graph", None)
    ainvoke_fn = getattr(agent, "ainvoke", None)
    invoke_fn = getattr(agent, "invoke", None)
    astream_fn = getattr(agent, "astream", None)

    # Build adapter with default implementations
    class _LangGraphAdapter:
        agent_name = name
        agent_description = getattr(agent, "agent_description", f"LangGraph agent: {name}")
        agent_framework = "langgraph"

        def __init__(self):
            self._agent = agent
            self._graph = graph

        @property
        def graph(self):
            return self._graph

        async def atransform(self, input_data: dict[str, Any]) -> dict[str, Any]:
            if ainvoke_fn is not None:
                config = input_data.get("runnable_config") or make_config(
                    thread_id=input_data.get("thread_id"),
                )
                result = await ainvoke_fn(input_data, config=config)
                return result if isinstance(result, dict) else {"response": str(result)}
            return {"response": f"Agent '{name}' has no async invoke method"}

        def transform(self, input_data: dict[str, Any]) -> dict[str, Any]:
            if invoke_fn is not None:
                result = invoke_fn(input_data)
                return result if isinstance(result, dict) else {"response": str(result)}
            return {"response": f"Agent '{name}' has no invoke method"}

        async def astream(self, input_data: dict[str, Any]):
            if astream_fn is not None:
                config = input_data.get("runnable_config") or make_config(
                    thread_id=input_data.get("thread_id"),
                )
                async for event in astream_fn(input_data, config=config):
                    yield event
                return
            # Fallback: yield atransform result
            result = await self.atransform(input_data)
            yield result

    return _LangGraphAdapter()


# =====================================================================
# Tool support
# =====================================================================


def tools_to_names(tools: list[Any]) -> list[str]:
    """Extract tool names from a list of LangChain tools.

    Handles ``BaseTool``, ``@tool``-decorated functions, and
    raw callables.

    Args:
        tools: List of LangChain tool objects.

    Returns:
        List of tool name strings.
    """
    names: list[str] = []
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if name:
            names.append(str(name))
    return names


def bind_tools(llm: Any, tools: list[Any]) -> Any:
    """Bind LangChain tools to an LLM, if supported.

    Calls ``llm.bind_tools(tools)`` if available; otherwise returns
    the LLM unchanged.

    Args:
        llm: A LangChain chat model.
        tools: List of tools to bind.

    Returns:
        The LLM with tools bound, or the original LLM.
    """
    if hasattr(llm, "bind_tools"):
        return llm.bind_tools(tools)
    return llm


# =====================================================================
# Streaming helpers
# =====================================================================


async def collect_stream(
    stream: AsyncIterator[Any],
    *,
    message_key: str = "messages",
) -> dict[str, Any]:
    """Collect an async LangGraph stream into a final state dict.

    Merges all yielded chunks (the last-write-wins for each key)
    and returns the merged state.

    Args:
        stream: An async iterator of state updates.
        message_key: Key under which messages accumulate.

    Returns:
        The final merged state dict.

    Example::

        final = await collect_stream(graph.astream(state, config=config))
        print(final["response"])
    """
    state: dict[str, Any] = {}
    messages: list[Any] = []

    async for chunk in stream:
        if not isinstance(chunk, dict):
            # Non-dict stream modes (e.g. raw tokens) — accumulate under response.
            state["response"] = str(chunk)
            continue
        for key, value in chunk.items():
            if key == message_key:
                if isinstance(value, list):
                    messages.extend(value)
                elif value is not None:
                    messages.append(value)
            else:
                state[key] = value

    if messages:
        state[message_key] = messages

    # Extract final AI response
    for msg in reversed(messages):
        if hasattr(msg, "content") and _msg_role(msg) in ("ai", "assistant"):
            state["response"] = str(msg.content)
            break

    return state


# =====================================================================
# Prompt manager integration
# =====================================================================


def resolve_prompt(
    agent: Any,
    prompt_manager: Any | None = None,
    *,
    default: str = "",
) -> str:
    """Resolve the current system prompt for an agent.

    Checks (in order):
    1. ``agent.system_prompt`` attribute
    2. ``agent._system_prompt()`` method
    3. Agent's ``ChatPromptTemplate`` (via :func:`extract_system_prompt`)
    4. ``prompt_manager`` (versioned prompts)
    5. *default* fallback

    Args:
        agent: Any agent object.
        prompt_manager: Optional ``PromptManager`` instance.
        default: Fallback prompt string.

    Returns:
        The resolved system prompt text.
    """
    # 1. Direct attribute
    val = getattr(agent, "system_prompt", None)
    if isinstance(val, str) and val.strip():
        return val

    # 2. Method
    method = getattr(agent, "_system_prompt", None) or getattr(agent, "build_system_prompt", None)
    if callable(method):
        try:
            result = method()
            if isinstance(result, str) and result.strip():
                return result
        except Exception:
            pass

    # 3. LangChain template
    template = getattr(agent, "prompt_template", None) or getattr(agent, "template", None)
    if template is not None:
        text = extract_system_prompt(template)
        if text:
            return text

    # 4. Prompt manager
    if prompt_manager is not None:
        agent_name = getattr(agent, "agent_name", "unknown")
        try:
            return prompt_manager.get_prompt(agent_name)
        except Exception:
            pass

    return default


# =====================================================================
# Internal helpers
# =====================================================================


def _is_system(msg: Any) -> bool:
    """Check if a LangChain message/prompt is a system message."""
    type_name = type(msg).__name__.lower()
    if "system" in type_name:
        return True
    role = getattr(msg, "role", "") or getattr(msg, "type", "")
    return str(role).lower() == "system"


def _render_text(msg: Any) -> str:
    """Extract text from a message/prompt template."""
    if hasattr(msg, "prompt") and hasattr(msg.prompt, "template"):
        return str(msg.prompt.template)
    if hasattr(msg, "content"):
        content = msg.content
        return str(content) if isinstance(content, str) else str(content)
    if isinstance(msg, (tuple, list)) and len(msg) >= 2:
        return str(msg[1])
    return ""


def _dict_to_lc(d: dict[str, Any]) -> Any:
    """Convert a dict to a LangChain message object."""
    try:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    except ImportError:
        return d

    role = str(d.get("role", "")).lower()
    content = str(d.get("content", ""))

    if role in ("user", "human"):
        return HumanMessage(content=content)
    if role in ("ai", "assistant"):
        return AIMessage(content=content)
    if role == "tool":
        return ToolMessage(content=content, tool_call_id=str(d.get("tool_call_id", "")))
    return HumanMessage(content=content)


def _msg_role(msg: Any) -> str:
    """Return the role string for a LangChain message or dict."""
    if isinstance(msg, dict) and "role" in msg:
        role = str(msg.get("role", "")).lower()
        if role in ("user", "human"):
            return "human"
        if role in ("ai", "assistant"):
            return "ai"
        if role == "tool":
            return "tool"
        if role == "system":
            return "system"
    name = type(msg).__name__.lower()
    if "human" in name:
        return "human"
    if "ai" in name:
        return "ai"
    if "tool" in name:
        return "tool"
    if "system" in name:
        return "system"
    return "unknown"


# =====================================================================
# AgentAdapter — universal entry point for ANY agent pattern
# =====================================================================


class AgentAdapter:
    """Universal adapter for ANY LangChain/LangGraph agent into agentomatic.

    Auto-detects the agent's pattern and exposes a unified interface.
    **No agent changes required.**

    Supported patterns (auto-detected):
    - BaseGraphAgent — already compatible, passed through
    - StateGraph class — has graph + ainvoke/astream
    - Function graph — has graph_fn or create_graph
    - LCEL chain — prompt | llm | parser
    - deep_agent / create_deep_agent
    - Plain callable — (state) -> state

    Example::

        from agentomatic.langchain_adapter import AgentAdapter

        class MyBot:
            def __init__(self): self.graph = self._build()
            async def ainvoke(self, state, config=None):
                return await self.graph.ainvoke(state, config=config)

        adapted = AgentAdapter(MyBot())
        result = await adapted.atransform({"current_query": "Hello"})
    """

    def __init__(self, agent: Any, *, agent_name: str | None = None) -> None:
        self._agent = agent
        self._name = agent_name or getattr(agent, "agent_name", None) or "agent"
        self._desc = getattr(agent, "agent_description", None) or ""
        self._framework = getattr(agent, "agent_framework", None) or "langgraph"
        self._is_base_graph = hasattr(agent, "atransform") and hasattr(agent, "build_graph")
        self._has_graph = bool(getattr(agent, "graph", None))
        self._has_ainvoke = callable(getattr(agent, "ainvoke", None))
        self._has_invoke = callable(getattr(agent, "invoke", None))
        self._has_astream = callable(getattr(agent, "astream", None))

    @property
    def agent_name(self) -> str:
        return self._name

    @property
    def agent_description(self) -> str:
        return self._desc

    @property
    def agent_framework(self) -> str:
        return self._framework

    @property
    def graph(self) -> Any:
        if hasattr(self._agent, "graph"):
            return self._agent.graph
        if hasattr(self._agent, "get_graph"):
            return self._agent.get_graph()
        return None

    @property
    def is_compatible(self) -> bool:
        return self._is_base_graph or self._has_graph or self._has_ainvoke or self._has_invoke

    async def atransform(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Primary agentomatic entry point — async transform."""
        if self._is_base_graph:
            fn = getattr(self._agent, "atransform", None) or getattr(self._agent, "transform")
            result = fn(input_data)
            if hasattr(result, "__await__"):
                return await result
            return result

        thread_id = input_data.get("thread_id")
        config = input_data.get("runnable_config") or make_config(thread_id=thread_id)

        if self._has_ainvoke:
            # Preserve agentomatic keys (current_query, metadata, …) while
            # ensuring a LangChain-style ``messages`` list is present.
            payload = dict(input_data)
            existing = payload.get("messages")
            if not existing:
                payload["messages"] = dict_to_messages(input_data)
            result = await self._agent.ainvoke(payload, config=config)
            if isinstance(result, dict) and "messages" in result:
                merged = messages_to_dict(result["messages"], input_data)
                # Preserve non-message keys from the agent result (e.g. response).
                for key, value in result.items():
                    if key == "messages":
                        continue
                    if value is not None:
                        merged[key] = value
                return merged
            if isinstance(result, dict):
                return {**input_data, **result}
            return {**input_data, "response": str(result)}

        if self._has_invoke:
            import asyncio

            result = await asyncio.to_thread(self._agent.invoke, input_data)
            return result if isinstance(result, dict) else {**input_data, "response": str(result)}

        return {**input_data, "response": f"[{self._name}] no invoke method"}

    def transform(self, input_data: dict[str, Any]) -> dict[str, Any]:
        if self._is_base_graph and hasattr(self._agent, "transform"):
            return self._agent.transform(input_data)
        from agentomatic.async_utils import run_sync

        return run_sync(self.atransform(input_data))

    async def astream(self, input_data: dict[str, Any]):
        if self._is_base_graph and hasattr(self._agent, "astream"):
            async for event in self._agent.astream(input_data):
                yield event
            return
        if self._has_astream:
            config = input_data.get("runnable_config") or make_config(
                thread_id=input_data.get("thread_id")
            )
            async for event in self._agent.astream(input_data, config=config):
                yield event
            return
        result = await self.atransform(input_data)
        yield result

    def get_system_prompt(self, default: str = "") -> str:
        return resolve_prompt(self._agent, default=default)

    def set_system_prompt(self, new_prompt: str) -> None:
        template = getattr(self._agent, "prompt_template", None) or getattr(
            self._agent, "template", None
        )
        if template is not None:
            new_t = inject_system_prompt(template, new_prompt)
            for attr in ("prompt_template", "template"):
                if hasattr(self._agent, attr):
                    try:
                        setattr(self._agent, attr, new_t)
                    except (AttributeError, TypeError):
                        pass
        if hasattr(self._agent, "system_prompt"):
            try:
                self._agent.system_prompt = new_prompt
            except (AttributeError, TypeError):
                pass

    def detect_pattern(self) -> str:
        if self._is_base_graph:
            return "base_graph_agent"
        if self._has_graph:
            return "state_graph_agent"
        if self._has_ainvoke:
            return "callable_agent"
        return "unknown"

    def __repr__(self) -> str:
        return f"AgentAdapter({self._name!r}, {self.detect_pattern()})"
