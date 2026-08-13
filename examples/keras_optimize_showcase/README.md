# Keras-style optimization showcase (local omlx)

Proves that the Agentomatic optimization engine works **end-to-end** the
Keras way: define a class-based agent with a deliberately-bad prompt and
bad hyper-parameters, then `compile → fit → evaluate → save` and watch a
Keras-like `loss` get smaller **with every epoch** while the prompt keeps
evolving.

```
from agentomatic import BaseGraphAgent, EarlyStopping, MetricLoss, PromptFitterBridge

agent = MarkerAgent()                              # bad prompt + hot temperature
agent.compile(dataset, metrics=metrics, loss=loss, optimizer=bridge)
history = agent.fit(
    dataset,
    epochs=10,
    verbose=1,                                     # Keras-like per-epoch log lines
    validation_data=dataset.validation,            # adds val_* metrics
    callbacks=[curriculum, EarlyStopping(monitor="val_loss", patience=3)],
)
print(history.best("val_loss", mode="min"))        # (epoch, best_val_loss)
report = agent.evaluate(dataset.test, metrics)
agent.save("compiled/v1")
```

## The staged curriculum

The agent's output quality depends on **seven** signals in curriculum order:
four prompt markers (`banana`, `strawberry`, `blueberry`, `kiwi`) followed by
three temperature rungs (`<=0.5`, `<=0.3`, `<=0.15`). The demo's
`CurriculumCallback` raises one new requirement per epoch, so the optimizer
must keep discovering new markers — the loss descends step-by-step:

- `banana` is injectable by deterministic paths (expected-tips / gold few-shot)
- `strawberry` / `blueberry` / `kiwi` only by the LLM rewrite (judge guidance)
- the rungs only by the parameter optimizers

The agent itself is fully deterministic (no LLM calls inside `transform()`),
so **every point of loss movement is caused by the optimizer changing the
agent's prompt / parameters** — the omlx server is only used for prompt
rewriting, failure clustering and candidate proposal.

## Real results (local omlx, 10 epochs, `max_trials=6`)

```
📈 LOSS PER EPOCH PER METHOD (lower is better)
   rewrite              1.000 → 0.857 → 0.857 → 0.571 → 0.429 → 0.429 → …
   gepa_like            1.000 → 1.000 → 0.857 → 0.714 → 0.571 → 0.571 → …
   mipro_like           1.000 → 0.857 → 0.857 → 0.571 → 0.429 → 0.286 → 0.143 → 0.000 → …
   few_shot_bootstrap   1.000 → 0.857 → 0.857 → …
   param_search         1.000 → 0.857 → 0.714 → 0.571 → …
```

- `rewrite` discovers one prompt marker per epoch (banana → strawberry →
  blueberry → kiwi) and stops at the prompt-only ceiling.
- `mipro_like` additionally descends the temperature rungs (`0.7 → 0.02`) and
  converges to **loss 0.000**.
- `param_search` (no LLM calls for proposals) tunes temperature rung by rung.
- Every curve is non-increasing — the fitter never accepts a regression, and
  each epoch compounds from the previous best config.

## Run it

Requires a local omlx (or any OpenAI-compatible) server:

```bash
export OMLX_API_KEY=… 
export OMLX_BASE_URL=http://127.0.0.1:8000/v1

# single mode
uv run python examples/keras_optimize_showcase/train.py --mode rewrite
uv run python examples/keras_optimize_showcase/train.py --mode param_search

# all five registered fitter optimizers, loss curves side by side
uv run python examples/keras_optimize_showcase/train.py --mode all \
  --epochs 10 --max-trials 6 --report-dir docs/assets/showcase

# LLM data augmentation (20 seed → 30 examples), then fit
uv run python examples/keras_optimize_showcase/train.py --mode rewrite \
  --epochs 10 --max-trials 6 --augment --n-examples 30

# faster local model
uv run python examples/keras_optimize_showcase/train.py --mode all \
  --model omlx/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
```

Each mode writes:

- an interactive HolySheet report (`fit_<mode>.html`) — loss charts,
  per-epoch metric table, prompt evolution accordion, parameter changes,
  failure clusters;
- a JSON summary (`summary_<mode>.json`) with the full history, compiled
  config and prompt evolution;
- the compiled agent state under `optimization_results/showcase/<mode>/`.

## Files

- `agent.py` — `MarkerAgent` (class-based `BaseGraphAgent`) + curriculum
- `dataset.py` / `dataset.jsonl` — fake training dataset (10 train / 6 val / 4 test)
- `train.py` — the Keras-style workflow + curriculum + report generation
- `make_chart.py` — renders the loss-curve SVG for the docs
