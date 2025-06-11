"""Simplified FastAPI router for Alpha agent."""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import ValidationError
from loguru import logger

from ...app.dependencies import agent_registry
from .schemas import AlphaInput, AlphaOutput

router = APIRouter()


def get_alpha_agent():
    """Get Alpha agent instance."""
    agent = agent_registry.get_agent("alpha")
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alpha agent not found")
    return agent


@router.post("/invoke", response_model=AlphaOutput)
async def invoke_alpha_agent(
    input_data: AlphaInput,
    agent = Depends(get_alpha_agent)
) -> AlphaOutput:
    """Invoke Alpha agent with comprehensive validation."""
    try:
        # Validate input data (Pydantic does this automatically)
        logger.info(f"Alpha agent invoked with query: {input_data.query[:100]}...")

        # Additional business logic validation
        if len(input_data.query.split()) < 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Query should contain at least 2 words for meaningful processing"
            )

        result = await agent.run(input_data)
        logger.info("Alpha agent completed successfully")
        return result

    except ValidationError as e:
        logger.error(f"Alpha agent validation error: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Validation error: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Alpha agent failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent execution failed: {str(e)}")


@router.get("/health")
async def alpha_health(agent = Depends(get_alpha_agent)):
    """Check Alpha agent health."""
    try:
        health = await agent.health_check()
        return health
    except Exception as e:
        logger.error(f"Alpha health check failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Health check failed: {str(e)}")