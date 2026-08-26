"""A GET-only endpoint proving query-bound schema contracts in Docker."""

from pydantic import BaseModel

from agentomatic.endpoints import BaseEndpoint


class LookupInput(BaseModel):
    text: str


class LookupOutput(BaseModel):
    echoed: str


class LookupEndpoint(BaseEndpoint[LookupInput, LookupOutput]):
    endpoint_name = "lookup"
    endpoint_description = "A deterministic GET-only endpoint for Studio contract verification."
    path = "/search"
    methods = ["GET"]

    async def handle(self, request: LookupInput) -> LookupOutput:
        return LookupOutput(echoed=f"lookup:{request.text}")
