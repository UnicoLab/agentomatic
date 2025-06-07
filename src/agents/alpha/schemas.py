"""Simplified schemas for Alpha agent."""

from typing import Optional
from pydantic import BaseModel, Field


class AlphaInput(BaseModel):
    """Input for Alpha agent."""
    query: str = Field(..., description="The question or task to process")
    context: Optional[str] = Field(default="", description="Additional context")


class AlphaOutput(BaseModel):
    """Output from Alpha agent."""
    response: str = Field(..., description="The generated response")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    processing_time: float = Field(default=0.0, ge=0.0)
    prompt_version: str = Field(default="v1")