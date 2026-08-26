"""A real scalar-body plugin for the Docker/oMLX production fixture."""

from pydantic import BaseModel, RootModel

from agentomatic.plugins import BaseMLPlugin


class RootText(RootModel[str]):
    """A deliberately non-object request schema exposed through OpenAPI."""


class RootEchoOutput(BaseModel):
    text: str


class RootEcho(BaseMLPlugin[RootText, RootEchoOutput]):
    """Returns the scalar request body unchanged."""

    plugin_name = "root_echo"
    plugin_description = "Verifies scalar RootModel request bodies end to end."

    async def predict(self, inputs: RootText) -> RootEchoOutput:
        return RootEchoOutput(text=inputs.root)
