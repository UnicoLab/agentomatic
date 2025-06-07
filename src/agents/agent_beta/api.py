"""FastAPI router for the Beta agent."""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from ...app.dependencies import agent_registry
from .schemas import BetaInput, BetaOutput

router = APIRouter()


def get_beta_agent():
    """Dependency to get the Beta agent instance."""
    try:
        return agent_registry.get_agent("beta")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/invoke", response_model=BetaOutput)
async def invoke_beta_agent(
    input_data: BetaInput,
    agent = Depends(get_beta_agent)
) -> BetaOutput:
    """Invoke the Beta agent for problem analysis and reasoning.
    
    Example:
        POST /api/v1/agents/beta/invoke
        {
            "problem": "How to optimize database queries?",
            "domain": "Software Engineering",
            "requirements": ["Low latency", "High throughput"],
            "constraints": "Limited memory budget"
        }
    """
    try:
        logger.info(f"Beta agent invoked for problem: {input_data.problem[:50]}...")
        result = await agent.run(input_data)
        logger.info(f"Beta agent completed analysis with {len(result.reasoning_steps)} reasoning steps")
        return result
        
    except Exception as e:
        logger.error(f"Beta agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def beta_agent_health(agent = Depends(get_beta_agent)):
    """Check Beta agent health status."""
    try:
        # Check LLM health
        llm_healthy = await agent.llm.health_check()
        
        return {
            "agent": "beta",
            "status": "healthy" if llm_healthy else "degraded",
            "llm_healthy": llm_healthy,
            "reasoning_enabled": agent.config.enable_reasoning,
            "config": {
                "model_name": agent.config.model_name,
                "prompt_version": agent.config.prompt_version,
                "max_tokens": agent.config.max_tokens
            }
        }
    except Exception as e:
        logger.error(f"Beta agent health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def beta_agent_capabilities():
    """Get Beta agent capabilities and features."""
    return {
        "agent": "beta",
        "specialization": "Reasoning and Analysis",
        "features": [
            "Problem decomposition",
            "Structured reasoning",
            "Risk assessment",
            "Solution design",
            "Multi-step analysis"
        ],
        "input_fields": [
            "problem",
            "domain", 
            "requirements",
            "constraints"
        ],
        "output_fields": [
            "analysis",
            "reasoning_steps",
            "solution_approach",
            "risk_assessment"
        ]
    }