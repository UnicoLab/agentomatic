"""A typed custom endpoint for pipeline verification."""

from pydantic import BaseModel, Field

from agentomatic.endpoints import BaseEndpoint


class EnrichInput(BaseModel):
    text: str
    payload: dict[str, str] = Field(default_factory=dict)


class EnrichOutput(BaseModel):
    text: str


class Enricher(BaseEndpoint[EnrichInput, EnrichOutput]):
    endpoint_name = "enricher"
    endpoint_description = "Adds deterministic deployment context."
    # Keep a browser-safe read contract in the production-shaped fixture as
    # well as the ordinary JSON POST used by the pipeline verification.
    methods = ["GET", "POST"]

    async def handle(self, request: EnrichInput) -> EnrichOutput:
        return EnrichOutput(text=f"enriched:{request.text or request.payload.get('text', '')}")
