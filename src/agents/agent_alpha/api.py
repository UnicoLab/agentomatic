"""FastAPI router for the Alpha agent."""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from ...app.dependencies import agent_registry
from .schemas import AlphaInput, AlphaOutput

router = APIRouter()


def get_alpha_agent():
    """Dependency to get the Alpha agent instance."""
    try:
        return agent_registry.get_agent("alpha")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/invoke", response_model=AlphaOutput)
async def invoke_alpha_agent(
    input_data: AlphaInput,
    agent = Depends(get_alpha_agent)
) -> AlphaOutput:
    """Invoke the Alpha agent with input data.

    Example:
        POST /api/v1/agents/alpha/invoke
        {
            "query": "What is machine learning?",
            "context": "Educational context for beginners"
        }
    """
    try:
        logger.info(f"Alpha agent invoked with query: {input_data.query[:50]}...")
        result = await agent.run(input_data)
        logger.info(f"Alpha agent completed successfully")
        return result

    except Exception as e:
        logger.error(f"Alpha agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def alpha_agent_health(agent = Depends(get_alpha_agent)):
    """Check Alpha agent health status."""
    try:
        # Check LLM health
        llm_healthy = await agent.llm.health_check()

        return {
            "agent": "alpha",
            "status": "healthy" if llm_healthy else "degraded",
            "llm_healthy": llm_healthy,
            "config": {
                "model_name": agent.config.model_name,
                "prompt_version": agent.config.prompt_version
            }
        }
    except Exception as e:
        logger.error(f"Alpha agent health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))