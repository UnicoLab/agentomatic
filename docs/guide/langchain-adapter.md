# LangChain / LangGraph Adapter

Agentomatic ships a library helper — `agentomatic.langchain_adapter` — that
normalises LangChain / LangGraph abstractions (messages, prompt templates,
LCEL chains, tools, `RunnableConfig`) into the platform's dict-based
invoke contract.

This is **distinct from** the Studio `LangChainAdapter`
(`agentomatic.studio.adapters.langchain`), which powers the debug UI for
`framework="langchain"` agents. Use the library adapter when packaging a
LangChain agent as a class agent or bridging message types.

---

## Quick start

```python
from agentomatic.langchain_adapter import (
    AgentAdapter, dict_to_messages, messages_to_dict, serialize_messages,
)

# Wrap a compiled LangGraph / LCEL runnable
adapted = AgentAdapter(my_graph_or_chain)
result = await adapted.atransform({"current_query": "Hello", "messages": []})
# Sync entrypoint (safe inside FastAPI / notebooks via run_sync):
result = adapted.transform({"current_query": "Hello"})

# State dict → LangChain messages
msgs = dict_to_messages({"current_query": "Hello", "messages": []})

# Or convert a bare message list (graph dataclass state)
msgs = dict_to_messages([{"role": "user", "content": "Hi"}])

# LangChain messages → state dict / plain list
state = messages_to_dict(msgs)
plain = serialize_messages(msgs)  # list[{"role", "content"}]
```

## Scaffold a LangChain agent

```bash
agentomatic init chatbot --template langchain
```

The template uses `dict_to_messages` / `serialize_messages` inside nodes so
REST `/api/v1/{name}/invoke` payloads work with LangChain message objects.

## Prompt helpers

```python
from agentomatic.langchain_adapter import (
    resolve_prompt, extract_system_prompt, inject_system_prompt,
)

# Resolve from agent attributes / ChatPromptTemplate / PromptManager
system = resolve_prompt(agent, default="You are helpful.")

# Read or replace the system message on a ChatPromptTemplate / chain
system = extract_system_prompt(prompt_template)
updated = inject_system_prompt(prompt_template, "You are helpful.")
```

## Config helpers

```python
from agentomatic.langchain_adapter import make_config, inject_config

config = make_config(thread_id="t-1", tags=["studio"])
result = await chain.ainvoke(payload, config=config)

# inject_config mutates a *state dict* with runnable_config / thread_id keys
state = inject_config({"current_query": "Hi"}, config)
```

## Adapting LangGraph agents

```python
from agentomatic.langchain_adapter import adapt_langgraph_agent

wrapped = adapt_langgraph_agent(compiled_graph, agent_name="chatbot")
```

## Studio vs library

| Surface | Module | Role |
|---|---|---|
| Studio debug UI | `studio.adapters.langchain.LangChainAdapter` | SSE streaming, LCEL graph, traces |
| Runtime bridge | `langchain_adapter.AgentAdapter` | Dict ↔ messages, prompt/tools helpers |

Both can be used together: wrap with `AgentAdapter` for REST/optimize, and
declare `framework="langchain"` (for LCEL `node_fn` agents) so Studio selects
its adapter automatically. Class agents (`BaseGraphAgent`) keep
`framework="graph_agent"` and still use `langchain_adapter` helpers inside nodes.
