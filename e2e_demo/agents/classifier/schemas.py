"""Published schemas for the structured classifier fixture."""

from pydantic import BaseModel, Field


class CustomInvokeRequest(BaseModel):
    label: str = Field(..., description="Classification label")
    priority: int = Field(0, ge=0, description="Queue priority")


class CustomInvokeResponse(BaseModel):
    label: str
    priority: int
    response: str
