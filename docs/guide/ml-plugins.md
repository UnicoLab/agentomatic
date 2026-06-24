# ML Model Plugins

Agentomatic is more than just a Generative AI orchestration platform. It is a full-featured API generator that can also expose your **Classical Machine Learning models** (e.g. Scikit-learn, PyTorch, TensorFlow) as production-ready REST endpoints.

By defining an ML model as a "Plugin", Agentomatic provides:

- Auto-discovery (from the `plugins/` directory)
- Automatic generation of `/predict`, `/health`, and `/model_card` APIs
- Strict Pydantic-based payload validation (request and response)
- A complete, standardized ML Lifecycle (`train` -> `eval` -> `optimize` -> `predict`)
- Lifespan management (loading bulky weights into memory upon server startup)

## How it works

1. Place your models in the `plugins/` directory (each plugin in a separate folder).
2. Subclass `BaseMLPlugin[InputT, OutputT]`.
3. Provide the Pydantic schemas for the generic types `InputT` and `OutputT`.
4. Train and evaluate your model using the generated lifecycle scripts.

## The ML Lifecycle

Agentomatic encourages treating ML models as first-class citizens. When you generate a plugin, you get a full suite of scripts to manage the model's lifecycle:

=== "Training (`train.py`)"
    Loads the dataset, trains the model (e.g. fitting a scikit-learn pipeline), and saves the weights to disk (`.pkl`, `.pt`, etc.).

=== "Evaluation (`eval.py`)"
    Loads the saved weights and evaluates the model against a test dataset, calculating metrics like Accuracy, F1, or MSE.

=== "Optimization (`optimize.py`)"
    Runs hyperparameter tuning (e.g. GridSearchCV) to find the best configuration before retraining.

=== "Inference (`predict.py`)"
    Local CLI script to quickly test the model locally before deploying it behind the API.

## Scaffolding a Plugin

Generate a ready-to-use plugin with the CLI:

```bash
agentomatic init my_plugin --template plugin
```

This creates a complete structured repository:

```text
plugins/my_plugin/
├── __init__.py       # Package marker
├── plugin.py         # BaseMLPlugin subclass (load_model & predict)
├── train.py          # Classical ML training loop
├── eval.py           # Evaluation script
├── optimize.py       # Hyperparameter tuning
├── predict.py        # Local CLI inference
├── dataset.jsonl     # Sample dataset
├── Makefile          # Workflow commands
└── README.md         # Documentation
```

### Next Steps Workflow

After scaffolding, you can navigate into your plugin directory and use standard `make` commands:

1. `make train` — Fits your model using `dataset.jsonl` and saves the weights.
2. `make eval` — Tests the model.
3. `make optimize` — Tunes hyperparameters.
4. `make predict` — Runs a quick local test.

## Example: Sentiment Analyzer

Here is an example of wrapping a classical ML model.

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
        Load your heavy PyTorch / TF / Joblib weights here!
        """
        # Example: self.model = joblib.load("model_weights.pkl")
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
        card["architecture"] = "Logistic Regression"
        return card
```

## Generated API Endpoints

Once you run `agentomatic run`, Agentomatic will dynamically generate the following endpoints for the `sentiment_analyzer`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/plugins` | Lists all registered plugins across the platform. |
| `POST` | `/api/v1/plugins/sentiment_analyzer/predict` | Executes `predict()`. Requires `SentimentInput` body, returns `SentimentOutput`. |
| `GET`  | `/api/v1/plugins/sentiment_analyzer/health` | Returns `{"status": "ok"}` if `load_model()` has finished running. |
| `GET`  | `/api/v1/plugins/sentiment_analyzer/model_card` | Returns the output of `model_card()`. |

You can view these directly in the Swagger UI at `http://localhost:8000/docs`.

## Framework Agnosticism

Agentomatic has absolutely zero opinion on what you put inside `load_model()` and `predict()`. You can run blocking code like `sklearn.predict()` or async code like HTTP calls to external APIs. Agentomatic uses FastAPI, which automatically manages thread-pooling for synchronous code.
