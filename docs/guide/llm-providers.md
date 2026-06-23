# LLM Providers & Failovers

Agentomatic abstracts away the underlying differences between LLM providers using a unified Multi-Provider LLM Factory (`get_llm`). This allows your agents to seamlessly swap out the model provider, handle failover mechanisms, and stay robust during outages.

## Supported Providers

Agentomatic supports multiple LangChain-compatible model providers out of the box:
- `ollama`: For local models (e.g. `llama3`, `mistral`)
- `openai`: OpenAI GPT series
- `azure`: Azure OpenAI endpoints
- `vertex`: Google Vertex AI models
- `google_genai`: Google GenAI API
- `dummy`: For testing purposes

## Initializing an LLM

You can construct a singleton LLM instance using the `get_llm` function. The returned instance is cached to avoid unnecessary re-initialization.

```python
from agentomatic.providers.llm import get_llm

llm = get_llm(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
    api_key="your-api-key"
)
```

## Failover Chains

A key reliability feature of Agentomatic is the ability to automatically fallback to secondary or tertiary models if the primary provider goes down or rate-limits you.

You can define a failover chain using the `fallbacks` parameter:

```python
from agentomatic.providers.llm import get_llm

llm = get_llm(
    provider="openai",              # Primary provider
    model="gpt-4",
    api_key="...",
    fallbacks=["azure", "ollama"],  # Fallback chain
    # Shared or specific kwargs:
    base_url="http://localhost:11434",
    temperature=0.1
)
```

If OpenAI fails, the system will automatically route the request to `azure`. If `azure` fails, it will attempt `ollama`.

### Telemetry

Failovers are automatically tracked. You can inspect the total failover count programmatically:

```python
from agentomatic.providers.llm import get_failover_count

count = get_failover_count()
print(f"Total failovers occurred: {count}")
```

## Embedding Providers

Agentomatic also provides abstractions for embedding models:

```python
from agentomatic.providers.embeddings import get_embeddings

embeddings = get_embeddings(
    provider="ollama",
    model="nomic-embed-text"
)
```

This keeps both generation and vector search completely agnostic to the underlying provider.
