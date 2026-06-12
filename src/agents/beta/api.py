"""Simplified FastAPI router for Beta agent."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import ValidationError

from ...app.dependencies import agent_registry
from .schemas import BetaInput, BetaOutput

router = APIRouter()


def get_beta_agent():
    """Get Beta agent instance."""
    agent = agent_registry.get_agent("beta")
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Beta agent not found")
    return agent


@router.post("/invoke", response_model=BetaOutput)
async def invoke_beta_agent(input_data: BetaInput, agent=Depends(get_beta_agent)) -> BetaOutput:
    """Invoke Beta agent with comprehensive validation."""
    try:
        # Validate input data (Pydantic does this automatically)
        logger.info(f"Beta agent invoked with problem: {input_data.problem[:100]}...")

        # Additional business logic validation
        if len(input_data.problem.split()) < 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Problem description should contain at least 3 words for meaningful analysis",
            )

        # Validate requirements if provided
        if input_data.requirements and len(input_data.requirements) > 20:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Maximum 20 requirements allowed for processing efficiency",
            )

        result = await agent.run(input_data)
        logger.info("Beta agent completed analysis successfully")
        return result

    except ValidationError as e:
        logger.error(f"Beta agent validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Validation error: {str(e)}"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Beta agent failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


@router.get("/health")
async def beta_health(agent=Depends(get_beta_agent)):
    """Check Beta agent health."""
    try:
        health = await agent.health_check()
        return health
    except Exception as e:
        logger.error(f"Beta health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )
