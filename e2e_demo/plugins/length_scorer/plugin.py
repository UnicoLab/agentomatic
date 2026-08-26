"""A deterministic plugin for end-to-end routing verification."""

from pydantic import BaseModel

from agentomatic.plugins import BaseMLPlugin


class ScoreInput(BaseModel):
    text: str


class ScoreOutput(BaseModel):
    score: int


class LengthScorer(BaseMLPlugin[ScoreInput, ScoreOutput]):
    plugin_name = "length_scorer"
    plugin_description = "Returns the character count."

    async def predict(self, inputs: ScoreInput) -> ScoreOutput:
        return ScoreOutput(score=len(inputs.text))
