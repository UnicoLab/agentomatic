"""Pydantic schemas for Beta agent input and output."""

from typing import Optional, List
from pydantic import BaseModel, Field


class BetaInput(BaseModel):
    """Input schema for the Beta agent.
    
    Example:
        input_data = BetaInput(
            problem="How to optimize database performance?",
            domain="Software Engineering",
            requirements=["Low latency", "High throughput"]
        )
    """
    
    problem: str = Field(..., description="The problem statement to analyze")
    domain: str = Field(..., description="The domain or context of the problem")
    requirements: List[str] = Field(
        default_factory=list,
        description="Specific requirements or constraints"
    )
    constraints: Optional[str] = Field(
        default="",
        description="Additional constraints or limitations"
    )
    max_tokens: Optional[int] = Field(
        default=1500,
        description="Maximum tokens for the response",
        ge=1,
        le=4000
    )
    temperature: Optional[float] = Field(
        default=0.5,
        description="Temperature for response generation",
        ge=0.0,
        le=2.0
    )


class ReasoningStep(BaseModel):
    """Individual reasoning step in the analysis."""
    
    step_number: int = Field(..., description="Step number in the reasoning process")
    description: str = Field(..., description="Description of this reasoning step")
    rationale: str = Field(..., description="Rationale behind this step")


class BetaOutput(BaseModel):
    """Output schema for the Beta agent.
    
    Example:
        output = BetaOutput(
            analysis="Detailed analysis...",
            reasoning_steps=[...],
            confidence=0.92
        )
    """
    
    analysis: str = Field(..., description="The complete analysis response")
    reasoning_steps: List[ReasoningStep] = Field(
        default_factory=list,
        description="Structured reasoning steps"
    )
    solution_approach: str = Field(
        default="",
        description="Recommended solution approach"
    )
    risk_assessment: str = Field(
        default="",
        description="Risk assessment and mitigation strategies"
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score for the analysis",
        ge=0.0,
        le=1.0
    )
    tokens_used: int = Field(
        default=0,
        description="Number of tokens used in generation",
        ge=0
    )
    processing_time: float = Field(
        default=0.0,
        description="Processing time in seconds",
        ge=0.0
    )
    prompt_version: str = Field(
        default="v1",
        description="Version of the prompt template used"
    )