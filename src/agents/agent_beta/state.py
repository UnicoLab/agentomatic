"""State management for the Beta agent."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from .schemas import BetaInput, BetaOutput


class BetaState(BaseModel):
    """State model for Beta agent workflow focused on reasoning and analysis.
    
    Example:
        state = BetaState(
            input=BetaInput(problem="Database optimization", domain="Software"),
            run_count=1
        )
    """
    
    # Input/Output
    input: Optional[BetaInput] = Field(default=None, description="Input data")
    output: Optional[BetaOutput] = Field(default=None, description="Output data")
    
    # Processing state
    last_prompt: Optional[str] = Field(default=None, description="Last formatted prompt")
    last_llm_response: Optional[str] = Field(default=None, description="Last LLM response")
    run_count: int = Field(default=0, description="Number of processing runs")
    
    # Reasoning state
    reasoning_enabled: bool = Field(default=True, description="Whether reasoning is enabled")
    analysis_depth: str = Field(default="standard", description="Depth of analysis")
    
    # Validation state
    input_valid: bool = Field(default=False, description="Input validation status")
    output_valid: bool = Field(default=False, description="Output validation status")
    validation_error: Optional[str] = Field(default=None, description="Validation error message")
    
    # Error handling
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True