"""Simplified FastAPI router for Beta agent."""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from ...app.dependencies import agent_registry
from .schemas import BetaInput, BetaOutput

router = APIRouter()


def get_beta_agent():
    """Get Beta agent instance."""
    agent = agent_registry.get_agent("beta")
    if not agent:
        raise HTTPException(status_code=404, detail="Beta agent not found")
    return agent


@router.post("/invoke", response_model=BetaOutput)
async def invoke_beta_agent(
    input_data: BetaInput,
    agent = Depends(get_beta_agent)
) -> BetaOutput:
    """Invoke Beta agent."""
    try:
        result = await agent.run(input_data)
        return result
    except Exception as e:
        logger.error(f"Beta agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def beta_health(agent = Depends(get_beta_agent)):
    """Check Beta agent health."""
    try:
        health = await agent.health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def beta_capabilities():
    """Get Beta agent capabilities."""
    return {
        "agent": "beta",
        "specialization": "Reasoning and Analysis",
        "features": [
            "Input classification",
            "Multi-step processing",
            "Complex problem solving",
            "Simple query handling"
        ],
        "routing": "Conditional based on input complexity"
    }