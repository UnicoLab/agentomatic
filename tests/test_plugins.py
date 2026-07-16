"""Tests for the ML Model Plugins system."""

from __future__ import annotations

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

    def __init__(self) -> None:
        super().__init__()
        self.load_count = 0
        self.model: str | None = None

    async def load_model(self) -> None:
        # Simulate loading
        self.load_count += 1
        self.model = f"dummy_model_weights_v{self.load_count}"
        await super().load_model()

    async def predict(self, inputs: SentimentInput) -> SentimentOutput:
        # Dummy inference
        text = inputs.text.lower()
        if "good" in text:
            return SentimentOutput(sentiment="positive", confidence=0.99)
        elif "bad" in text:
            return SentimentOutput(sentiment="negative", confidence=0.95)
        return SentimentOutput(sentiment="neutral", confidence=0.50)

    def model_card(self) -> dict:
        card = super().model_card()
        card["weights"] = self.model
        return card


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
        assert resp.json()["loaded_at"] is not None

        # Test Model Card
        resp = client.get("/api/v1/plugins/sentiment_analyzer/model_card")
        assert resp.status_code == 200
        assert resp.json()["name"] == "sentiment_analyzer"

        # Test list includes loaded_at
        resp = client.get("/api/v1/plugins")
        assert resp.status_code == 200
        listed = resp.json()
        assert any(p["name"] == "sentiment_analyzer" and p.get("loaded_at") for p in listed)

        # Test Inference
        resp = client.post(
            "/api/v1/plugins/sentiment_analyzer/predict", json={"text": "This is a good day!"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.99


def test_plugin_reload_happy_path(tmp_path):
    """POST /plugins/{name}/reload re-calls load_model and returns status."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    platform = AgentPlatform(agents_dir=tmp_path / "agents", plugins_dir=plugins_dir)
    plugin = DummySentimentPlugin()
    platform._plugin_registry._plugins["sentiment_analyzer"] = plugin
    app = platform.build()

    with TestClient(app) as client:
        assert plugin.load_count == 1  # lifespan
        first_loaded_at = plugin.loaded_at

        resp = client.post("/api/v1/plugins/sentiment_analyzer/reload")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "sentiment_analyzer"
        assert body["is_loaded"] is True
        assert body["version"] == "0.1.0"
        assert body["loaded_at"] is not None
        assert body["model_card"]["weights"] == "dummy_model_weights_v2"
        assert plugin.load_count == 2
        assert plugin.loaded_at is not None
        assert first_loaded_at is not None

        # Reload-all endpoint
        resp = client.post("/api/v1/plugins/reload")
        assert resp.status_code == 200, resp.text
        all_body = resp.json()
        assert all_body["reloaded"] == 1
        assert all_body["failed"] == 0
        assert plugin.load_count == 3


def test_plugin_reload_missing_returns_404(tmp_path):
    """Reload on an unknown plugin name returns HTTP 404."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    platform = AgentPlatform(agents_dir=tmp_path / "agents", plugins_dir=plugins_dir)
    app = platform.build()

    with TestClient(app) as client:
        resp = client.post("/api/v1/plugins/does_not_exist/reload")
        assert resp.status_code == 404
