# Stacks & Configuration Management

Agentomatic allows you to bundle and manage multi-environment LLM configurations, databases, authentication, and feature flags into deployable units known as **Stacks**.

Stacks are defined using YAML files and managed using the `StackManager`. This allows you to smoothly transition from local development to staging and production without changing code.

## The StackManager

The `StackManager` reads YAML stack files, applies environment variables, and interpolates `${ENV_VAR}` references automatically.

### Loading a Stack

You can initialize the `StackManager` to load configurations dynamically:

```python
from agentomatic.stacks.manager import StackManager

# Load from a directory (looks for local.yaml, prod.yaml, etc.)
mgr = StackManager("stacks/")
stack = mgr.load("local")

# Or load directly from a file
mgr = StackManager.from_file("stacks/prod.yaml")
```

## Stack YAML Structure

A complete stack configuration is defined under a `StackConfig` structure. This includes:

- **llm**: Named LLM profiles (e.g. `default`, `fast`, `judge`).
- **embedding**: The embedding provider settings.
- **database**: Async database connection URL and pool settings.
- **features**: Feature flags to toggle components like metrics, streaming, and rate limiting.
- **auth**: API Key or JWT authentication settings.
- **agent_overrides**: Per-agent specific overrides.

### Example: `stacks/production.yaml`

```yaml
name: "Production Stack"
description: "High-availability production stack with failover support"

features:
  enable_streaming: true
  enable_metrics: true
  enable_rate_limit: true
  enable_db: true

database:
  url: "postgresql+asyncpg://${DB_USER}:${DB_PASS}@${DB_HOST}/agentomatic"
  pool_size: 20

auth:
  method: "jwt"
  jwks_url: "https://auth.example.com/.well-known/jwks.json"

llm:
  default:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.2
    api_key: "${OPENAI_API_KEY}"

  fast:
    provider: "ollama"
    model: "llama3"
    base_url: "http://internal-ollama:11434"

agent_overrides:
  coder_agent:
    provider: "openai"
    model: "o1-preview"
```

## Using Stack Configuration

Once loaded, you can access properties and pass them to agents or LLM factories seamlessly:

```python
# Access a specific LLM configuration profile
llm_cfg = mgr.get_llm_config("fast")

# Access the database URL (interpolated)
db_url = stack.database.url

# Check feature flags
if stack.features.enable_metrics:
    print("Prometheus metrics enabled")
```

The stack configuration provides a single source of truth for your agent's ecosystem.
