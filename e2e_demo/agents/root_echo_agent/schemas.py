"""Live root request schema for the Docker production fixture."""

from pydantic import BaseModel, RootModel


class CustomInvokeRequest(RootModel[str]):
    """The public invoke body is one JSON string, not an object envelope."""


class CustomInvokeResponse(BaseModel):
    """Stable response contract for root-input deployment checks."""

    response: str
    root: str
