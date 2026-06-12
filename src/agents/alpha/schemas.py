"""Simplified schemas for Alpha agent."""


from pydantic import BaseModel, Field, field_validator


class AlphaInput(BaseModel):
    """Input for Alpha agent."""

    query: str = Field(
        ..., min_length=1, max_length=10000, description="The question or task to process"
    )
    context: str | None = Field(default="", max_length=5000, description="Additional context")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @field_validator("context")
    @classmethod
    def validate_context(cls, v):
        if v:
            return v.strip()
        return v


class AlphaOutput(BaseModel):
    """Output from Alpha agent."""

    response: str = Field(..., description="The generated response")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    processing_time: float = Field(default=0.0, ge=0.0)
    prompt_version: str = Field(default="v1")
