"""Search-space definition for prompt fitting.

Defines *what* is allowed to change during optimization — model
parameters, RAG knobs, tool settings, and few-shot configuration.

The :class:`PromptSearchSpace` dataclass acts as a declarative grid:
each boolean flag toggles an entire category on/off, while the
corresponding ``*_param_space`` dict maps parameter names to lists of
candidate values.

Usage::

    from agentomatic.optimize.search_space import PromptSearchSpace

    space = PromptSearchSpace(
        optimize_model_params=True,
        model_param_space={
            "temperature": [0.0, 0.3, 0.7],
            "top_p": [0.9, 1.0],
        },
    )
    print(space.n_combinations("model"))   # 6
    print(space.total_search_size())       # 6

    combos = space.param_combinations("model")
    # [{"temperature": 0.0, "top_p": 0.9}, ...]

    sampled = space.sample_params(3, "model")
    # 3 random combinations from the grid
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Valid few-shot selection strategies.
_VALID_FEW_SHOT_STRATEGIES = frozenset(
    {"top_k", "diversity_weighted", "random_search"},
)


# =====================================================================
# Search Space
# =====================================================================


@dataclass(slots=True)
class PromptSearchSpace:
    """Declarative search space for prompt optimisation.

    Each boolean flag enables / disables an optimisation category.
    The ``*_param_space`` dicts define candidate values — keys are
    parameter names, values are lists of candidates.

    Examples
    --------
    >>> space = PromptSearchSpace()
    >>> space.n_combinations("model")
    45
    >>> space.active_spaces()
    ['model']
    """

    # -- boolean flags ---------------------------------------------------
    optimize_system_prompt: bool = True
    optimize_user_template: bool = False
    optimize_few_shot: bool = True
    optimize_model_params: bool = True
    optimize_rag_params: bool = False
    optimize_tool_params: bool = False
    optimize_model_choice: bool = False

    # -- parameter grids -------------------------------------------------
    model_param_space: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "temperature": [0.0, 0.1, 0.2, 0.4, 0.7],
            "top_p": [0.7, 0.9, 1.0],
            "max_tokens": [800, 1200, 2000],
        },
    )
    rag_param_space: dict[str, list[Any]] = field(default_factory=dict)
    tool_param_space: dict[str, list[Any]] = field(default_factory=dict)

    # -- deployment routing ------------------------------------------------
    model_choices: list[str] = field(
        default_factory=list
    )  # e.g. ["ollama/qwen2.5:7b", "openai/gpt-4.1"]
    fallback_models: list[str] = field(default_factory=list)  # fallback candidates
    routing_weight_space: dict[str, list[float]] = field(default_factory=dict)  # A/B weights

    # -- few-shot config -------------------------------------------------
    max_few_shot_examples: int = 5
    few_shot_selection_strategy: str = "diversity_weighted"

    # -- internal helpers ------------------------------------------------

    def __post_init__(self) -> None:
        if self.few_shot_selection_strategy not in _VALID_FEW_SHOT_STRATEGIES:
            msg = (
                f"Invalid few_shot_selection_strategy "
                f"'{self.few_shot_selection_strategy}'. "
                f"Choose from {sorted(_VALID_FEW_SHOT_STRATEGIES)}."
            )
            raise ValueError(msg)

    def _resolve_space(self, space_name: str) -> dict[str, list[Any]]:
        """Return the param-space dict for *space_name*."""
        mapping: dict[str, dict[str, list[Any]]] = {
            "model": self.model_param_space,
            "rag": self.rag_param_space,
            "tool": self.tool_param_space,
            "routing": self.routing_weight_space,
        }
        if space_name not in mapping:
            msg = f"Unknown space '{space_name}'. Choose from {sorted(mapping)}."
            raise ValueError(msg)
        return mapping[space_name]

    # -- public API ------------------------------------------------------

    def param_combinations(self, space_name: str = "model") -> list[dict[str, Any]]:
        """Generate every combination for the requested parameter space.

        Uses :func:`itertools.product` to build the full Cartesian product.

        Examples
        --------
        >>> s = PromptSearchSpace(
        ...     model_param_space={"temperature": [0.0, 0.1], "top_p": [0.9, 1.0]},
        ... )
        >>> s.param_combinations("model")
        [{'temperature': 0.0, 'top_p': 0.9}, {'temperature': 0.0, 'top_p': 1.0}, {'temperature': 0.1, 'top_p': 0.9}, {'temperature': 0.1, 'top_p': 1.0}]
        """
        space = self._resolve_space(space_name)
        if not space:
            return [{}]
        keys = list(space.keys())
        values = [space[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def sample_params(
        self,
        n: int,
        space_name: str = "model",
    ) -> list[dict[str, Any]]:
        """Return a random sample of *n* combinations from *space_name*.

        If *n* >= total combinations the full grid is returned (shuffled).
        """
        all_combos = self.param_combinations(space_name)
        if n >= len(all_combos):
            logger.debug(
                "Requested {} samples but only {} available — returning all.",
                n,
                len(all_combos),
            )
            shuffled = list(all_combos)
            random.shuffle(shuffled)
            return shuffled
        return random.sample(all_combos, n)

    def n_combinations(self, space_name: str = "model") -> int:
        """Return the total number of combinations for *space_name*.

        Examples
        --------
        >>> PromptSearchSpace().n_combinations("model")
        45
        """
        space = self._resolve_space(space_name)
        if not space:
            return 1
        total = 1
        for values in space.values():
            total *= len(values)
        return total

    def active_spaces(self) -> list[str]:
        """Return names of parameter spaces that are enabled.

        Returns a subset of ``["model", "rag", "tool", "routing"]``.
        """
        active: list[str] = []
        if self.optimize_model_params:
            active.append("model")
        if self.optimize_rag_params:
            active.append("rag")
        if self.optimize_tool_params:
            active.append("tool")
        if self.routing_weight_space:
            active.append("routing")
        return active

    def total_search_size(self) -> int:
        """Total combinations across **all** active parameter spaces.

        Includes model choices and fallback models if ``optimize_model_choice``
        is enabled.
        """
        total = 0
        for name in self.active_spaces():
            total += self.n_combinations(name)
        if self.optimize_model_choice and self.model_choices:
            total += len(self.model_choices)
        if total == 0:
            return 1
        return total

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the search space to a plain dictionary."""
        return {
            "optimize_system_prompt": self.optimize_system_prompt,
            "optimize_user_template": self.optimize_user_template,
            "optimize_few_shot": self.optimize_few_shot,
            "optimize_model_params": self.optimize_model_params,
            "optimize_rag_params": self.optimize_rag_params,
            "optimize_tool_params": self.optimize_tool_params,
            "model_param_space": self.model_param_space,
            "rag_param_space": self.rag_param_space,
            "tool_param_space": self.tool_param_space,
            "max_few_shot_examples": self.max_few_shot_examples,
            "few_shot_selection_strategy": self.few_shot_selection_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptSearchSpace:
        """Deserialise a search space from a dictionary.

        Unknown keys are silently ignored for forward-compatibility.

        Examples
        --------
        >>> d = PromptSearchSpace().to_dict()
        >>> PromptSearchSpace.from_dict(d).n_combinations("model")
        45
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
