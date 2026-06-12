"""Full Agent Example — demonstrates ALL agentomatic customization options.

This example shows:
1. Auto-discovery from folder structure
2. Custom agent config (config.py)
3. Custom schemas (schemas.py)
4. Custom API router (api.py) — overrides auto-generated endpoints
5. Versioned prompts (prompts.json)
6. LangChain tools (tools.py)
7. Multi-step graph with branching (graph.py)
8. Lifecycle hooks
9. Extra global routers

Usage:
    pip install agentomatic[langgraph]
    cd examples/full_agent
    uvicorn main:app --reload

    # Test:
    curl http://localhost:8000/api/v1/agents
    curl http://localhost:8000/api/v1/weather/locations
    curl -X POST http://localhost:8000/api/v1/weather/forecast \
      -H 'Content-Type: application/json' \
      -d '{"location": "Paris", "days": 3}'
"""
from __future__ import annotations

from fastapi import APIRouter

from agentomatic import AgentPlatform

# --- Platform Setup ---
platform = AgentPlatform.from_folder(
    "agents/",
    title="Full Demo Platform",
    description="Agentomatic full customization demo",
    version="2.0.0",
    api_prefix="/api/v1",
    package_prefix="agents",
    cors_origins=["*"],
    log_level="DEBUG",
)


# --- Lifecycle Hooks ---
@platform.on_startup
async def on_startup():
    """Called when the platform starts."""
    print("🚀 Full demo platform starting!")


@platform.on_shutdown
async def on_shutdown():
    """Called when the platform shuts down."""
    print("🛑 Full demo platform stopping!")


# --- Extra Global Router ---
admin_router = APIRouter(prefix="/admin", tags=["Admin"])


@admin_router.get("/stats")
async def get_stats():
    """Global admin stats endpoint."""
    return {
        "agents": platform.registry.count,
        "agent_names": platform.registry.list_names(),
    }


platform.include_router(admin_router)

# --- Build App ---
app = platform.build()

if __name__ == "__main__":
    platform.run(port=8001, reload=True)
