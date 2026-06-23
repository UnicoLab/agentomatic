"""Tests for the ML Model Plugins system."""

from fastapi.testclient import TestClient
from pydantic import BaseModel

from agentomatic import AgentPlatform
from agentomatic.plugins import BaseMLPlugin


class SentimentInput(BaseModel):
    text: str


class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float


class DummySentimentPlugin(BaseMLPlugin[SentimentInput, SentimentOutput]):
    plugin_name = "sentiment_analyzer"
    plugin_description = "A dummy sentiment analyzer plugin"
    plugin_version = "0.1.0"

    async def load_model(self) -> None:
        # Simulate loading
        await super().load_model()
        self.model = "dummy_model_weights"

    async def predict(self, inputs: SentimentInput) -> SentimentOutput:
        # Dummy inference
        text = inputs.text.lower()
        if "good" in text:
            return SentimentOutput(sentiment="positive", confidence=0.99)
        elif "bad" in text:
            return SentimentOutput(sentiment="negative", confidence=0.95)
        return SentimentOutput(sentiment="neutral", confidence=0.50)


def test_plugin_registry_and_router(tmp_path):
    # Setup temporary plugins directory
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    # We can programmatically register the plugin instead of file discovery for the test,
    # but the platform discovers automatically from the folder.
    # To test integration, we create the platform and inject the plugin into the registry.
    platform = AgentPlatform(agents_dir=tmp_path / "agents", plugins_dir=plugins_dir)

    # Manually register the dummy plugin
    platform._plugin_registry._plugins["sentiment_analyzer"] = DummySentimentPlugin()

    app = platform.build()

    with TestClient(app) as client:
        # Lifespan should load the model
        assert platform._plugin_registry.get_plugin("sentiment_analyzer").is_loaded

        # Test Health
        resp = client.get("/api/v1/plugins/sentiment_analyzer/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["plugin_name"] == "sentiment_analyzer"

        # Test Model Card
        resp = client.get("/api/v1/plugins/sentiment_analyzer/model_card")
        assert resp.status_code == 200
        assert resp.json()["name"] == "sentiment_analyzer"

        # Test Inference
        resp = client.post(
            "/api/v1/plugins/sentiment_analyzer/predict", json={"text": "This is a good day!"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.99
