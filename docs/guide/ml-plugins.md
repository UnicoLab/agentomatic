# ML Model Plugins

Agentomatic is more than just a Generative AI orchestration platform. It is a full-featured API generator that can also expose your **Classical Machine Learning models** (e.g. Scikit-learn, PyTorch, TensorFlow, PyMC) as production-ready REST endpoints.

By defining an ML model as a "Plugin", Agentomatic provides:

- Auto-discovery (from the `plugins/` directory)
- Automatic generation of `predict()`, `health()`, and `model_card()` APIs
- Strict Pydantic-based payload validation (request and response)
- Swagger / OpenAPI documentation auto-generation
- Lifespan management (loading bulky weights into memory upon server startup)

## How it works

1. Place your models in the `plugins/` directory (each plugin in a separate folder or `.py` file).
2. Subclass `BaseMLPlugin[InputT, OutputT]`.
3. Provide the Pydantic schemas for the generic types `InputT` and `OutputT`.

## Example: Sentiment Analyzer

Here is an example of wrapping a dummy ML model (or a scikit-learn model, or a PyTorch network).

```python
# plugins/sentiment/plugin.py
import asyncio
from pydantic import BaseModel, Field
from agentomatic.plugins import BaseMLPlugin

# 1. Define Request Schema
class SentimentInput(BaseModel):
    text: str = Field(..., description="The text to analyze")

# 2. Define Response Schema
class SentimentOutput(BaseModel):
    sentiment: str = Field(..., description="The classified sentiment")
    confidence: float = Field(..., description="The model's confidence score")

# 3. Create the Plugin
class SentimentPlugin(BaseMLPlugin[SentimentInput, SentimentOutput]):
    plugin_name = "sentiment_analyzer"
    plugin_description = "A classical ML sentiment analyzer."
    plugin_version = "1.0.0"

    async def load_model(self) -> None:
        """
        Executed exactly once during the platform startup.
        Load your heavy PyTorch / TF weights here!
        """
        await asyncio.sleep(0.1)  # Simulate slow weight loading
        self.model_weights = {"positive": 0.99, "neutral": 0.5}

        # You MUST call super().load_model() to mark the plugin as loaded
        await super().load_model()

    async def predict(self, inputs: SentimentInput) -> SentimentOutput:
        """
        Executed on every POST request to /api/v1/plugins/sentiment_analyzer/predict.
        Agentomatic guarantees that `inputs` is a valid SentimentInput object.
        """
        text = inputs.text.lower()
        if "good" in text:
            return SentimentOutput(sentiment="positive", confidence=self.model_weights["positive"])
        return SentimentOutput(sentiment="neutral", confidence=self.model_weights["neutral"])

    def model_card(self) -> dict:
        """
        Optional: Return metadata about the model.
        """
        card = super().model_card()
        card["architecture"] = "DummyNet v1"
        return card
```

## Generated API Endpoints

Once you run `agentomatic run` (or `platform.build()`), Agentomatic will dynamically generate the following endpoints for the `sentiment_analyzer`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/plugins/sentiment_analyzer/predict` | Executes `predict()`. Requires `SentimentInput` body, returns `SentimentOutput`. |
| `GET`  | `/api/v1/plugins/sentiment_analyzer/health` | Returns `{"status": "ok"}` if `load_model()` has finished running. |
| `GET`  | `/api/v1/plugins/sentiment_analyzer/model_card` | Returns the output of `model_card()`. |

You can view these directly in the Swagger UI at `http://localhost:8000/docs`.

## Scaffolding

Generate a ready-to-use plugin with the CLI:

```bash
agentomatic init my_plugin --template plugin
```

This creates:

```text
plugins/my_plugin/
├── plugin.py     # BaseMLPlugin subclass
└── README.md     # Documentation
```

## Framework Agnosticism

Agentomatic has absolutely zero opinion on what you put inside `load_model()` and `predict()`. You can run blocking code like `sklearn.predict()` or async code like HTTP calls to external APIs. Agentomatic uses FastAPI, which automatically manages thread-pooling for synchronous code.
