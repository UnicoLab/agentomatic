"""Simplified schemas for Beta agent."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class BetaInput(BaseModel):
    """Input for Beta agent."""
    problem: str = Field(..., min_length=1, max_length=15000, description="The problem to analyze")
    domain: Optional[str] = Field(default="", max_length=500, description="Problem domain")
    requirements: Optional[List[str]] = Field(default_factory=list, description="List of requirements")
    constraints: Optional[str] = Field(default="", max_length=2000, description="Constraints to consider")

    @field_validator('problem')
    @classmethod
    def validate_problem(cls, v):
        if not v or not v.strip():
            raise ValueError('Problem description cannot be empty')
        return v.strip()

    @field_validator('domain')
    @classmethod
    def validate_domain(cls, v):
        if v:
            return v.strip()
        return v

    @field_validator('requirements')
    @classmethod
    def validate_requirements(cls, v):
        if v:
            # Clean up requirements list
            return [req.strip() for req in v if req and req.strip()]
        return []

    @field_validator('constraints')
    @classmethod
    def validate_constraints(cls, v):
        if v:
            return v.strip()
        return v


class BetaOutput(BaseModel):
    """Output from Beta agent."""
    analysis: str = Field(..., description="The analysis result")
    reasoning_steps: List[str] = Field(default_factory=list)
    solution_approach: str = Field(default="")
    risk_assessment: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    processing_time: float = Field(default=0.0, ge=0.0)