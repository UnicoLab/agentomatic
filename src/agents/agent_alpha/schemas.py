"""Pydantic schemas for Alpha agent input and output."""

from typing import Optional
from pydantic import BaseModel, Field


class AlphaInput(BaseModel):
    """Input schema for the Alpha agent.

    Example:
        input_data = AlphaInput(
            query="What is machine learning?",
            context="Educational context for beginners"
        )
    """

    query: str = Field(..., description="The main question or task to process")
    context: Optional[str] = Field(
        default="",
        description="Additional context or background information"
    )
    max_tokens: Optional[int] = Field(
        default=1000,
        description="Maximum tokens for the response",
        ge=1,
        le=4000
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="Temperature for response generation",
        ge=0.0,
        le=2.0
    )


class AlphaOutput(BaseModel):
    """Output schema for the Alpha agent.

    Example:
        output = AlphaOutput(
            response="Machine learning is...",
            confidence=0.95,
            tokens_used=150
        )
    """

    response: str = Field(..., description="The generated response to the query")
    confidence: float = Field(
        default=0.0,
        description="Confidence score for the response",
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