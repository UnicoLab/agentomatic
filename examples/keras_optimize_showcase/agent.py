"""MarkerAgent — a deliberately-bad agent used to prove the optimization engine.

The agent's output quality depends on SEVEN optimizable signals, staged so
that a multi-epoch ``fit()`` improves *progressively* (a real Keras-like
loss curve that descends epoch after epoch and then plateaus):

1. **``banana``** — injectable by deterministic paths (expected-tips / gold
   few-shot). Usually found in epoch 1.
2. **``strawberry`` / ``blueberry`` / ``kiwi``** — only discoverable by the
   LLM rewrite from the judge guidance in the expected answers. One or two
   are typically found per epoch, so the prompt keeps evolving.
3. **temperature rungs** — the ``param_search`` optimizer descends a grid;
   the quality signal rewards each rung (``<=0.5``, ``<=0.3``, ``<=0.15``)
   so parameter tuning lowers the loss step by step too.

Every epoch of ``agent.fit()`` therefore produces a Keras-style ``loss``
that gets smaller with each iteration and a prompt that keeps improving —
exactly what we want to showcase. The agent itself is fully deterministic
(no LLM calls inside ``transform()``), so **all** loss movement is caused
by the optimizer changing the prompt / parameters.

Usage (all imports from the public package — no internals)::

    from agentomatic import BaseGraphAgent
    from examples.keras_optimize_showcase.agent import MarkerAgent

    agent = MarkerAgent()
    agent.transform({"query": "What is the budget?"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentomatic import BaseGraphAgent

# Deliberately-bad baseline: no grounding marker at all.
BAD_PROMPT = "You are a vague assistant."
# Success markers the fit optimizers must inject into the prompt.
# ``banana`` is injected by deterministic paths (expected-tips / gold
# few-shot); the others can only be discovered by the LLM rewrite from the
# judge guidance, so prompt-only fits improve progressively across epochs.
MARKER = "banana"
MARKER2 = "strawberry"
MARKER3 = "blueberry"
MARKER4 = "kiwi"
LLM_MARKERS = (MARKER2, MARKER3, MARKER4)
ALL_MARKERS = (MARKER, MARKER2, MARKER3, MARKER4)
# Few-shot header the agent detects when few-shot examples are appended.
FEW_SHOT_HEADER = "## Few-shot examples"
# Temperature rungs: each met rung contributes one quality step.
TEMP_RUNGS = (0.5, 0.3, 0.15)


@dataclass
class MarkerState:
    """Graph state for MarkerAgent."""

    request: str = ""
    temperature: float | None = None  # injected by the fitter during candidate evals
    output: dict[str, Any] = field(default_factory=dict)


class MarkerAgent(BaseGraphAgent[MarkerState]):
    """Deterministic agent whose quality depends on prompt + params.

    ``transform()`` is fully deterministic — no LLM calls — so the *only*
    source of score movement across epochs is the optimizer changing the
    agent's prompt / temperature / few-shot config. This makes the
    Keras-style loss curve a faithful measurement of the optimization
    engine (with the local omlx server used for prompt rewriting /
    failure analysis).

    The ``difficulty`` attribute (stepped by the demo's curriculum
    callback between epochs) controls how many quality signals the agent
    currently requires: epoch N demands the first N signals, so the
    optimizer must keep discovering new markers — producing a loss curve
    that descends every epoch.
    """

    agent_name = "marker_agent"
    agent_description = "Deliberately-bad agent used to prove prompt/param optimization."

    # --- the three optimizable knobs (bad by design) -------------------
    system_prompt = BAD_PROMPT
    temperature = 0.7  # too hot — param_search should descend the rungs
    top_p = 0.9
    few_shot_examples: list[dict[str, Any]] = []

    # Curriculum ladder: how many of the seven quality signals are required.
    difficulty: int = 1

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_temperature(self, state: MarkerState) -> float:
        """Prefer the fitter-injected temperature, else the agent attribute.

        Also honours ``compiled_config`` values (the path used when a fit
        result is applied without a matching instance attribute).
        """
        if state.temperature is not None:
            return float(state.temperature)
        compiled = getattr(self, "compiled_config", None) or {}
        if compiled.get("temperature") is not None:
            try:
                return float(compiled["temperature"])
            except (TypeError, ValueError):
                pass
        try:
            return float(getattr(self, "temperature", 0.7))
        except (TypeError, ValueError):
            return 0.7

    def _prompt_with_few_shot(self, prompt: str) -> str:
        """Merge ``self.few_shot_examples`` into the prompt (agent-side)."""
        examples = list(getattr(self, "few_shot_examples", None) or [])
        if not examples:
            return prompt
        blocks = [FEW_SHOT_HEADER]
        for idx, ex in enumerate(examples[:6], 1):
            q = str(ex.get("query") or ex.get("input") or "").strip()
            r = str(ex.get("response") or ex.get("output") or "").strip()
            if not q and not r:
                continue
            blocks.append(f"Example {idx}\nQ: {q[:400]}\nA: {r[:600]}")
        return (prompt + "\n\n" + "\n\n".join(blocks)).strip()

    # ------------------------------------------------------------------
    # graph node
    # ------------------------------------------------------------------

    def _signal_states(self, prompt_lower: str, temperature: float) -> list[bool]:
        """Return the seven quality signals, in curriculum order.

        Order: ``banana``, ``strawberry``, ``blueberry``, ``kiwi``, then the
        three temperature rungs (``<=0.5`` / ``<=0.3`` / ``<=0.15``).
        """
        few_shot_block = FEW_SHOT_HEADER.lower() in prompt_lower
        return [
            MARKER in prompt_lower or few_shot_block,
            MARKER2 in prompt_lower,
            MARKER3 in prompt_lower,
            MARKER4 in prompt_lower,
            temperature <= TEMP_RUNGS[0],
            temperature <= TEMP_RUNGS[1],
            temperature <= TEMP_RUNGS[2],
        ]

    def respond(self, state: MarkerState) -> MarkerState:
        prompt = self._prompt_with_few_shot(self.resolve_system_prompt(default=self.system_prompt))
        temperature = self._resolve_temperature(state)
        prompt_lower = prompt.lower()

        signals = self._signal_states(prompt_lower, temperature)
        difficulty = max(0, int(getattr(self, "difficulty", 1)))
        required = signals[:difficulty]
        satisfied = [sig for sig in required if sig]
        n_satisfied = len(satisfied)
        rungs_met = sum(1 for rung in TEMP_RUNGS if temperature <= rung)

        # Tokens are emitted for the *required* signals that are met only —
        # all seven signals (prompt markers AND temperature rungs) are
        # curriculum-gated, so the fit objective and epoch metrics move
        # exactly one step per epoch.
        token_map = [MARKER, MARKER2, MARKER3, MARKER4, "r1", "r2", "r3"]
        tokens = [token_map[i] for i in range(difficulty) if signals[i]]
        if tokens:
            tag = "OPT" if n_satisfied == difficulty else "PARTIAL"
            response = f"{tag}: answer to {state.request} {' '.join(tokens)}"
        else:
            tag = "OPT" if difficulty == 0 else "BASE"
            response = f"{tag}: answer to {state.request}"

        state.output = {
            "response": response,
            "used_prompt": prompt,
            "temperature": temperature,
            "temp_ok": temperature <= TEMP_RUNGS[1],
            "temp_rungs": rungs_met,
            "banana_ok": signals[0],
            "strawberry_ok": signals[1],
            "blueberry_ok": signals[2],
            "kiwi_ok": signals[3],
            "n_satisfied": n_satisfied,
            "difficulty": difficulty,
        }
        return state

    # ------------------------------------------------------------------
    # BaseGraphAgent protocol
    # ------------------------------------------------------------------

    def input_to_state(self, input_data: dict[str, Any]) -> MarkerState:
        state = MarkerState(
            request=str(input_data.get("current_query") or input_data.get("query") or "")
        )
        # The fitter injects candidate params via ``model_params`` /
        # ``temperature`` during candidate evaluation.
        model_params = input_data.get("model_params")
        if isinstance(model_params, dict) and model_params.get("temperature") is not None:
            state.temperature = model_params["temperature"]
        elif input_data.get("temperature") is not None:
            state.temperature = input_data["temperature"]
        return state

    def state_to_output(self, state: MarkerState) -> dict[str, Any]:
        return state.output
