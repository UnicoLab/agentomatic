"""Simplified FastAPI router for Alpha agent."""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from ...app.dependencies import agent_registry
from .schemas import AlphaInput, AlphaOutput

router = APIRouter()


def get_alpha_agent():
    """Get Alpha agent instance."""
    agent = agent_registry.get_agent("alpha")
    if not agent:
        raise HTTPException(status_code=404, detail="Alpha agent not found")
    return agent


@router.post("/invoke", response_model=AlphaOutput)
async def invoke_alpha_agent(
    input_data: AlphaInput,
    agent = Depends(get_alpha_agent)
) -> AlphaOutput:
    """Invoke Alpha agent."""
    try:
        result = await agent.run(input_data)
        return result
    except Exception as e:
        logger.error(f"Alpha agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def alpha_health(agent = Depends(get_alpha_agent)):
    """Check Alpha agent health."""
    try:
        health = await agent.health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))