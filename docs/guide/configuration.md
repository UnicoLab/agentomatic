# Configuration

## Platform Configuration

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore

platform = AgentPlatform.from_folder(
    "agents/",
    # Identity
    title="My Platform",
    version="1.0.0",
    # API
    api_prefix="/api/v1",
    cors_origins=["*"],
    # Storage
    store=MemoryStore(),
    # Auth
    enable_auth=True,
    auth_api_key="secret",
    # Rate limiting
    enable_rate_limit=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    # Observability
    enable_metrics=True,
    enable_logging=True,
    log_level="INFO",
)
```

## Per-Agent Configuration

Create `config.py` in your agent folder:

```python
from pydantic import BaseModel, Field

class MyAgentConfig(BaseModel):
    prompt_version: str = Field("v1")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1)
    enable_memory: bool = Field(True)
```

## Environment Variables

All settings can be overridden via environment variables:

```bash
export AGENTOMATIC_LOG_LEVEL=DEBUG
export AGENTOMATIC_API_PREFIX=/api/v2
```
