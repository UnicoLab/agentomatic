import asyncio

from pydantic import BaseModel, Field

from agentomatic.plugins import BaseMLPlugin


class TextPayload(BaseModel):
    text: str = Field(..., description="The text to classify")

class ClassificationResult(BaseModel):
    category: str = Field(..., description="The classified category")
    score: float = Field(..., description="Confidence score")

class MyClassifier(BaseMLPlugin[TextPayload, ClassificationResult]):
    plugin_name = "my_classifier"
    plugin_description = "A dummy text classifier"
    plugin_version = "1.0.0"

    async def load_model(self) -> None:
        """Simulate loading a PyTorch/TensorFlow model."""
        await asyncio.sleep(0.1)
        self.model = {"spam": 0.9, "ham": 0.1}  # dummy weights
        await super().load_model()

    async def predict(self, inputs: TextPayload) -> ClassificationResult:
        """Run dummy inference."""
        if "buy" in inputs.text.lower():
            return ClassificationResult(category="spam", score=self.model["spam"])
        return ClassificationResult(category="ham", score=self.model["ham"])

    def model_card(self) -> dict:
        card = super().model_card()
        card["architecture"] = "DummyNet"
        return card
