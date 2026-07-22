# Bug Report — Fixed in v1.8.3

Diagnosed by tracing the full `train_next.py` optimization pipeline end-to-end.
All five bugs interact: together they cause the optimizer to waste compute,
produce zero signal, and leave omlx generating after the script exits.

---

## BUG-1 · `RewriteOptimizer` ignores `rewrite_model`, uses `task_model` instead

**File:** `src/agentomatic/optimize/fitter_optimizers.py` — `resolve_fitter_optimizer()`

**Symptom:** The dedicated rewrite LLM (configured via `PromptFitter(rewrite_model=...)`)
is silently ignored. The task-execution model is used for prompt rewriting, which is
slower, more expensive, and produces lower-quality rewrites when a smaller/specialised
rewrite LLM is intended.

**Root cause:**
```python
# BEFORE (wrong):
if cls is RewriteOptimizer:
    return RewriteOptimizer(model=model, **rewrite_only, **kwargs)
#                           ↑ task_model is used even when rewrite_model is set
```

**Fix:**
```python
# AFTER:
if cls is RewriteOptimizer:
    return RewriteOptimizer(model=rewrite_model or model, **rewrite_only, **kwargs)
```

---

## BUG-2 · Minibatch size clamps to `_MINIBATCH_MIN=5` even with 1 validation point

**File:** `src/agentomatic/optimize/fitter.py` — `PromptFitter.fit()`

**Symptom:** With a small validation set (e.g. 1 point), the log prints:

```
Minibatch: 5 / 1 points (500%)
```

The clamped `minibatch_size=5` is passed to `val_points[:5]`, which silently clips
to 1 element anyway. This is a no-op but the misleading log obscures the real problem:
with only 1 validation point there is essentially no statistical signal.

**Root cause:**
```python
# BEFORE:
minibatch_size = max(_MINIBATCH_MIN, int(len(val_points) * _MINIBATCH_FRACTION))
# No upper bound → size can exceed the available data.
```

**Fix:**
```python
# AFTER:
minibatch_size = min(
    len(val_points),
    max(_MINIBATCH_MIN, int(len(val_points) * _MINIBATCH_FRACTION)),
)
minibatch_size = max(1, minibatch_size)
```

---

## BUG-3 · Early-stopping patience never fires for small trial budgets

**File:** `src/agentomatic/optimize/fitter.py` — `PromptFitter.fit()`

**Symptom:** With `max_trials=8` and `_CANDIDATES_PER_ROUND=4`, `max_rounds=2`.
The global constant `_EARLY_STOP_PATIENCE=3` is *greater* than `max_rounds`,
so the early-stop condition `no_improvement_rounds >= 3` is never reached.
Every round runs to completion even when there is zero improvement signal,
wasting ~2× the compute and keeping omlx busy generating outputs after the
Python script has logically finished.

**Root cause:**
```python
_EARLY_STOP_PATIENCE: int = 3   # constant, never adapted to trial budget

for round_idx in range(max_rounds):   # max_rounds = 2 for max_trials=8
    ...
    if no_improvement_rounds >= _EARLY_STOP_PATIENCE:  # 3 > 2 → never true
        break
```

**Fix:** Compute an effective patience that is always ≤ `max_rounds`:
```python
effective_patience = min(_EARLY_STOP_PATIENCE, max(1, max_rounds))
# All _EARLY_STOP_PATIENCE references in the loop replaced with effective_patience.
```

**Side-effect fixed:** This also prevents omlx from being queried for every
round when there is no improvement signal — reducing the number of lingering
in-flight LLM requests after the script exits.

---

## BUG-4 · `AgentExample.to_datapoint()` skips the `current_query` input field

**File:** `src/agentomatic/agents/types.py` — `AgentExample.to_datapoint()`

**Symptom:** Agents that use `current_query` as their primary input field (the
standard agentomatic convention set by `_wrap_local_agent`) produce a `DataPoint`
whose `query` field is the entire `json.dumps(input)` blob instead of the actual
user question. The optimizer's briefing, failure-analysis prompts, and LLM-as-judge
evaluations all receive a JSON dump rather than the human-readable query, degrading
candidate generation and scoring quality.

**Root cause:**
```python
# BEFORE:
query = (
    self.input.get("query")
    or self.input.get("request")
    or self.input.get("question")
    or json.dumps(self.input)  # ← fallback fires for current_query agents
)
```

**Fix:**
```python
# AFTER:
query = (
    self.input.get("query")
    or self.input.get("current_query")   # ← added
    or self.input.get("request")
    or self.input.get("question")
    or json.dumps(self.input)
)
```

---

## BUG-5 · `PromptFitterBridge.optimizer` buried in `**kwargs` — wrong default, silent misconfiguration

**File:** `src/agentomatic/agents/optimizers.py` — `PromptFitterBridge`

**Symptom:** The `optimizer` strategy (e.g. `"rewrite"`, `"mipro_like"`) is passed
as an opaque `**kwargs` entry. If the user misspells the key (e.g. `optimiser=`) the
error is silent and the fitter silently falls back to the `PromptFitter` default
(`"gepa_like"`), a complex multi-LLM strategy that is rarely what the user wants.

In practice, the PromptFitter default `"gepa_like"` was observed firing instead of
the requested `"rewrite"` in some configurations because the kwarg was not seeded
into the kwargs dict before overrides were applied.

**Root cause:**
```python
# BEFORE: optimizer not named → no default, no validation, buried in **kwargs
def __init__(self, agent_name="", task_model=..., ..., **kwargs):
    self.kwargs = kwargs  # optimizer hidden here
```

**Fix:** Promote `optimizer` to an explicit named parameter with a sensible default:
```python
# AFTER:
def __init__(self, ..., optimizer: str = "rewrite", ..., **kwargs):
    self.optimizer = optimizer
    self.kwargs = kwargs

# In _build_fitter(), seed kwargs before applying overrides:
kwargs.setdefault("optimizer", self.optimizer)
kwargs.update(overrides)  # per-fit() overrides still win
```

---

## Related fix in `train_next.py` (user-side, not library)

The fitter's optimization metric was set to `CustomMetric(fn=_opt_composite)`.
`_opt_composite` returns `1.0` for **any** response containing valid JSON with the
required keys — which the agent always produces. Consequently:

- Baseline score = 1.0
- All candidate scores = 1.0
- No candidate ever beats the baseline → 0 rounds complete → 0 prompt improvement
- The optimizer analyzes "0 failures" and generates meaningless candidates

**Fix:** Pass the `LocalJudgeMetric` (which scores ~0.3 for the baseline) as the
fitter's `metric=`. The judge measures actual answer quality (pertinence,
groundedness, actionability), giving the optimizer real failure signal to work with.

```python
# BEFORE:
fitter = PromptFitterBridge(..., metric=CustomMetric(fn=_opt_composite, name="composite"))

# AFTER:
fitter = PromptFitterBridge(..., metric=judge)   # judge = LocalJudgeMetric(...)
```

---

## How to reinstall the fixed library

From `SCOOPER_NEW/ai_platform/`:

```bash
uv pip install -e /path/to/agentomatic
# or
uv pip install /path/to/agentomatic
```
