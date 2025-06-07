"""Simplified schemas for Beta agent."""

from typing import List, Optional
from pydantic import BaseModel, Field


class BetaInput(BaseModel):
    """Input for Beta agent."""
    problem: str = Field(..., description="The problem to analyze")
    domain: Optional[str] = Field(default="", description="Problem domain")
    requirements: Optional[List[str]] = Field(default_factory=list)
    constraints: Optional[str] = Field(default="")


class BetaOutput(BaseModel):
    """Output from Beta agent."""
    analysis: str = Field(..., description="The analysis result")
    reasoning_steps: List[str] = Field(default_factory=list)
    solution_approach: str = Field(default="")
    risk_assessment: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    processing_time: float = Field(default=0.0, ge=0.0)