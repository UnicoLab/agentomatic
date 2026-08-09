"""Environment-driven optimisation settings with ``.env`` support.

All settings are prefixed with ``OPTIMIZER_`` and can be set via
``.env`` files.  Includes presets for common scenarios and a
``display()`` method that prints a formatted configuration table.

Example::

    # .env
    OPTIMIZER_TRAINER_MODEL=ollama/mistral:7b
    OPTIMIZER_MAX_ITERATIONS=10
    OPTIMIZER_STRATEGY=combined

    from agentomatic.optimize.settings import OptimizerSettings
    settings = OptimizerSettings()
    settings.display()
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any

# =====================================================================
# Enums
# =====================================================================


class OptimizationStrategy(StrEnum):
    ITERATIVE_REFINEMENT = "iterative_refinement"
    BOOTSTRAP_FEW_SHOT = "bootstrap_few_shot"
    MIPRO = "mipro"
    SIGNATURE_OPTIMIZATION = "signature_optimization"
    COMBINED = "combined"


class AgentTypeEnum(StrEnum):
    AUTO = "auto"
    STATELESS = "stateless"
    RAG = "rag"
    TOOL_USING = "tool_using"
    CONVERSATIONAL = "conversational"
    DEEP_AGENT = "deep_agent"


class AugmentationMethod(StrEnum):
    PARAPHRASE = "paraphrase"
    ADD_NOISE = "add_noise"
    SIMPLIFY = "simplify"
    COMPLICATE = "complicate"
    EDGE_CASE = "edge_case"
    ADVERSARIAL = "adversarial"

    @classmethod
    def default_set(cls) -> list[AugmentationMethod]:
        return [cls.PARAPHRASE, cls.EDGE_CASE]


class CallbackType(StrEnum):
    EARLY_STOPPING = "early_stopping"
    CHECKPOINT = "checkpoint"
    NAN_STOPPING = "nan_stopping"
    TEMPERATURE_SCHEDULER = "temperature_scheduler"
    PROGRESS = "progress"
    PLATEAU_STOPPING = "plateau_stopping"
    SCORE_THRESHOLD = "score_threshold"

    @classmethod
    def default_set(cls) -> list[CallbackType]:
        return [cls.EARLY_STOPPING, cls.CHECKPOINT, cls.PROGRESS]


class EvalMetric(StrEnum):
    ANSWER_RELEVANCY = "answer_relevancy"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    GEVAL = "geval"
    RAGAS_FAITHFULNESS = "ragas_faithfulness"
    RAGAS_ANSWER_RELEVANCY = "ragas_answer_relevancy"
    RAGAS_CONTEXT_PRECISION = "ragas_context_precision"
    RAGAS_CONTEXT_RECALL = "ragas_context_recall"
    TOOL_CALL_ACCURACY = "tool_call_accuracy"
    TOOL_SELECTION = "tool_selection"
    TASK_COMPLETION = "task_completion"
    STEP_EFFICIENCY = "step_efficiency"
    GOAL_ACCURACY = "goal_accuracy"


# =====================================================================
# Helpers
# =====================================================================


def _env(key: str, default: str = "") -> str:
    return os.getenv(f"OPTIMIZER_{key}", default)


def _env_int(key: str, default: str = "0") -> int:
    return int(_env(key, default))


def _env_float(key: str, default: str = "0.0") -> float:
    return float(_env(key, default))


def _env_bool(key: str, default: str = "false") -> bool:
    return _env(key, default).lower() in ("true", "1", "yes")


def _env_list(key: str, default: str = "") -> list[str]:
    val = _env(key, default)
    if not val:
        return []
    return [item.strip() for item in val.split(",") if item.strip()]


# =====================================================================
# OptimizerSettings
# =====================================================================


class OptimizerSettings:
    """Configuration for prompt optimisation, driven by environment.

    All fields can be set via ``OPTIMIZER_*`` environment variables or
    passed directly to the constructor.  Use classmethod presets for
    one-liner configuration.

    Example::

        settings = OptimizerSettings(
            trainer_model="ollama/mistral:7b",
            max_iterations=5,
        )
        settings = OptimizerSettings.for_local()     # free Ollama
        settings = OptimizerSettings.for_quality()   # GPT-4o
        settings = OptimizerSettings.for_quick()     # 2-round bootstrap
    """

    # -- Models --------------------------------------------------------
    trainer_model: str
    eval_model: str
    gen_model: str

    # -- Training ------------------------------------------------------
    max_iterations: int
    target_score: float
    strategy: OptimizationStrategy
    patience: int

    # -- Evaluation ----------------------------------------------------
    agent_type: AgentTypeEnum
    eval_metrics: list[EvalMetric]
    eval_threshold: float
    custom_eval_criteria: str

    # -- Augmentation --------------------------------------------------
    augmentation_enabled: bool
    augmentation_count: int
    augmentation_methods: list[AugmentationMethod]

    # -- Callbacks -----------------------------------------------------
    callbacks: list[CallbackType]

    # -- Experiment tracking -------------------------------------------
    track_experiments: bool
    project_name: str
    output_dir: str

    # -- Output ---------------------------------------------------------
    verbose: bool
    auto_report: bool

    def __init__(
        self,
        *,
        trainer_model: str | None = None,
        eval_model: str | None = None,
        gen_model: str | None = None,
        max_iterations: int | None = None,
        target_score: float | None = None,
        strategy: OptimizationStrategy | str | None = None,
        patience: int | None = None,
        agent_type: AgentTypeEnum | str | None = None,
        eval_metrics: list[EvalMetric | str] | None = None,
        eval_threshold: float | None = None,
        custom_eval_criteria: str | None = None,
        augmentation_enabled: bool | None = None,
        augmentation_count: int | None = None,
        augmentation_methods: list[AugmentationMethod | str] | None = None,
        callbacks: list[CallbackType | str] | None = None,
        track_experiments: bool | None = None,
        project_name: str | None = None,
        output_dir: str | None = None,
        verbose: bool | None = None,
        auto_report: bool | None = None,
    ) -> None:
        # Best-effort .env load so OPTIMIZER_* vars work without exporting.
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except Exception:
            pass

        # Models
        self.trainer_model = trainer_model or _env("TRAINER_MODEL", "ollama/mistral:7b")
        self.eval_model = eval_model or _env("EVAL_MODEL", self.trainer_model)
        self.gen_model = gen_model or _env("GEN_MODEL", self.trainer_model)

        # Training
        self.max_iterations = (
            max_iterations if max_iterations is not None else _env_int("MAX_ITERATIONS", "5")
        )
        self.target_score = (
            target_score if target_score is not None else _env_float("TARGET_SCORE", "0.85")
        )
        strategy_str = (
            strategy if isinstance(strategy, str) else (strategy.value if strategy else None)
        )
        strategy_str = strategy_str or _env("STRATEGY", "")
        self.strategy = (
            OptimizationStrategy(strategy_str)
            if strategy_str
            else OptimizationStrategy.ITERATIVE_REFINEMENT
        )
        self.patience = patience if patience is not None else _env_int("PATIENCE", "3")

        # Evaluation
        at_str = (
            agent_type
            if isinstance(agent_type, str)
            else (agent_type.value if agent_type else None)
        )
        at_str = at_str or _env("AGENT_TYPE", "")
        self.agent_type = AgentTypeEnum(at_str) if at_str else AgentTypeEnum.AUTO
        self.eval_metrics = self._parse_metrics(eval_metrics)
        self.eval_threshold = (
            eval_threshold if eval_threshold is not None else _env_float("EVAL_THRESHOLD", "0.7")
        )
        self.custom_eval_criteria = custom_eval_criteria or _env("CUSTOM_EVAL_CRITERIA", "")

        # Augmentation
        self.augmentation_enabled = (
            augmentation_enabled
            if augmentation_enabled is not None
            else _env_bool("AUGMENTATION_ENABLED")
        )
        self.augmentation_count = (
            augmentation_count
            if augmentation_count is not None
            else _env_int("AUGMENTATION_COUNT", "10")
        )
        self.augmentation_methods = self._parse_augmentation(augmentation_methods)

        # Callbacks
        self.callbacks = self._parse_callbacks(callbacks)

        # Experiment tracking
        self.track_experiments = (
            track_experiments
            if track_experiments is not None
            else _env_bool("TRACK_EXPERIMENTS", "true")
        )
        self.project_name = project_name or _env("PROJECT_NAME", "optimization")
        self.output_dir = output_dir or _env("OUTPUT_DIR", "optimization_results")

        # Output
        self.verbose = verbose if verbose is not None else _env_bool("VERBOSE", "true")
        self.auto_report = (
            auto_report if auto_report is not None else _env_bool("AUTO_REPORT", "true")
        )

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_metrics(self, raw: list[EvalMetric | str] | None) -> list[EvalMetric]:
        if raw:
            return [EvalMetric(r) if isinstance(r, str) else r for r in raw]
        env_val = _env_list("EVAL_METRICS")
        if env_val:
            return [EvalMetric(m) for m in env_val]
        return [EvalMetric.ANSWER_RELEVANCY, EvalMetric.GEVAL]

    def _parse_augmentation(
        self, raw: list[AugmentationMethod | str] | None
    ) -> list[AugmentationMethod]:
        if raw:
            return [AugmentationMethod(r) if isinstance(r, str) else r for r in raw]
        env_val = _env_list("AUGMENTATION_METHODS")
        if env_val:
            return [AugmentationMethod(m) for m in env_val]
        return AugmentationMethod.default_set()

    def _parse_callbacks(self, raw: list[CallbackType | str] | None) -> list[CallbackType]:
        if raw:
            return [CallbackType(r) if isinstance(r, str) else r for r in raw]
        env_val = _env_list("CALLBACKS")
        if env_val:
            return [CallbackType(c) for c in env_val]
        return CallbackType.default_set()

    # ------------------------------------------------------------------
    # Computed
    # ------------------------------------------------------------------

    @property
    def effective_eval_model(self) -> str:
        return self.eval_model or self.trainer_model

    @property
    def effective_gen_model(self) -> str:
        return self.gen_model or self.trainer_model

    @property
    def trainer_provider(self) -> str:
        return self.trainer_model.split("/")[0] if "/" in self.trainer_model else "ollama"

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def for_local(cls) -> OptimizerSettings:
        """Free, fast setup — local Ollama, 5 iterations."""
        return cls(
            trainer_model="ollama/mistral:7b",
            max_iterations=5,
            strategy=OptimizationStrategy.ITERATIVE_REFINEMENT,
            agent_type=AgentTypeEnum.AUTO,
            eval_metrics=[EvalMetric.ANSWER_RELEVANCY, EvalMetric.GEVAL],
            augmentation_enabled=False,
        )

    @classmethod
    def for_quality(cls) -> OptimizerSettings:
        """Best quality — GPT-4o, 10 iterations, combined strategy."""
        return cls(
            trainer_model="openai/gpt-4o",
            max_iterations=10,
            strategy=OptimizationStrategy.COMBINED,
            target_score=0.90,
            agent_type=AgentTypeEnum.AUTO,
            eval_metrics=[EvalMetric.ANSWER_RELEVANCY, EvalMetric.GEVAL, EvalMetric.FAITHFULNESS],
        )

    @classmethod
    def for_quick(cls) -> OptimizerSettings:
        """Ultra-fast — 2 bootstrap rounds, minimal overhead."""
        return cls(
            trainer_model="ollama/mistral:7b",
            max_iterations=2,
            strategy=OptimizationStrategy.BOOTSTRAP_FEW_SHOT,
            agent_type=AgentTypeEnum.AUTO,
            eval_metrics=[EvalMetric.ANSWER_RELEVANCY],
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Print a formatted configuration table."""
        rows = [
            ("Trainer Model", self.trainer_model),
            ("Eval Model", self.eval_model),
            ("Max Iterations", str(self.max_iterations)),
            ("Target Score", f"{self.target_score:.2f}"),
            ("Strategy", self.strategy.value),
            ("Patience", str(self.patience)),
            ("Agent Type", self.agent_type.value),
            ("Eval Metrics", ", ".join(m.value for m in self.eval_metrics)),
            ("Eval Threshold", f"{self.eval_threshold:.2f}"),
            ("Augmentation", "on" if self.augmentation_enabled else "off"),
            ("Callbacks", ", ".join(c.value for c in self.callbacks)),
            ("Experiment DB", f"{self.output_dir}/experiments.db"),
            ("Verbose", "yes" if self.verbose else "no"),
        ]
        max_key = max(len(k) for k, _ in rows)
        sep = "─" * (max_key + 40)
        print(f"\n{sep}")
        print("  ⚙  OptimizerSettings")
        print(sep)
        for key, val in rows:
            print(f"  {key:<{max_key}} │ {val}")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainer_model": self.trainer_model,
            "eval_model": self.eval_model,
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "strategy": self.strategy.value,
            "agent_type": self.agent_type.value,
            "eval_metrics": [m.value for m in self.eval_metrics],
            "eval_threshold": self.eval_threshold,
            "augmentation_enabled": self.augmentation_enabled,
            "project_name": self.project_name,
        }


# =====================================================================
# Helpers
# =====================================================================


def model_provider(model: str) -> str:
    """Return the provider prefix from a model string."""
    for prefix in ("ollama", "openai", "gemini", "litellm", "omlx", "anthropic"):
        if model.startswith(f"{prefix}/"):
            return prefix
    return "ollama"


def validate_model_string(model: str) -> bool:
    """Check whether a model string looks valid."""
    return "/" in model and len(model.split("/")) >= 2


def show_available_options() -> None:
    """Print all available enum values for inspection."""
    for name, enum_cls in [
        ("Strategy", OptimizationStrategy),
        ("AgentType", AgentTypeEnum),
        ("AugmentationMethod", AugmentationMethod),
        ("CallbackType", CallbackType),
        ("EvalMetric", EvalMetric),
    ]:
        print(f"\n{name}:")
        for val in enum_cls:
            print(f"  - {val.value}")


# =====================================================================
# Pydantic BaseSettings integration (replaces hand-rolled env parsing)
# =====================================================================

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class OptimizerPydanticSettings(BaseSettings):
        """Pydantic :class:`BaseSettings` for full ``.env`` + validation.

        Reads from ``OPTIMIZER_*`` environment variables and ``.env``
        files automatically.  Uses the same enums defined above for
        IDE autocomplete and validation.

        Example::

            # .env
            OPTIMIZER_TRAINER_MODEL=ollama/llama3:70b
            OPTIMIZER_MAX_ITERATIONS=10

            from agentomatic.optimize.settings import OptimizerPydanticSettings
            s = OptimizerPydanticSettings()  # reads .env automatically
            print(s.trainer_model)  # "ollama/llama3:70b"
        """

        model_config = SettingsConfigDict(
            env_prefix="OPTIMIZER_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # ── Models ────────────────────────────────────────────
        trainer_model: str = "ollama/mistral:7b"
        eval_model: str = "ollama/mistral:7b"

        # ── Training ──────────────────────────────────────────
        max_iterations: int = 5
        target_score: float = 0.85
        strategy: str = "iterative_refinement"
        patience: int = 3

        # ── Evaluation ────────────────────────────────────────
        agent_type: str = "auto"
        eval_metrics: str = "answer_relevancy,geval"
        eval_threshold: float = 0.7
        custom_eval_criteria: str = ""

        # ── Augmentation ──────────────────────────────────────
        augmentation_enabled: bool = False
        augmentation_count: int = 10
        augmentation_methods: str = "paraphrase,edge_case"

        # ── Callbacks ─────────────────────────────────────────
        callbacks: str = "early_stopping,checkpoint,progress"

        # ── Experiment tracking ───────────────────────────────
        track_experiments: bool = True
        output_dir: str = "optimization_results"
        project_name: str = "optimization"

        # ── Output ────────────────────────────────────────────
        verbose: bool = True
        auto_report: bool = True

        # ── API Keys (from global env, no prefix) ─────────────
        openai_api_key: str = ""
        google_api_key: str = ""
        anthropic_api_key: str = ""

        @property
        def eval_metrics_list(self) -> list[str]:
            return [m.strip() for m in self.eval_metrics.split(",") if m.strip()]

        @property
        def callbacks_list(self) -> list[str]:
            return [c.strip() for c in self.callbacks.split(",") if c.strip()]

        def display(self) -> None:
            """Print formatted configuration table."""
            rows = [
                ("Trainer Model", self.trainer_model),
                ("Eval Model", self.eval_model),
                ("Max Iterations", str(self.max_iterations)),
                ("Target Score", f"{self.target_score:.2f}"),
                ("Strategy", self.strategy),
                ("Patience", str(self.patience)),
                ("Agent Type", self.agent_type),
                ("Eval Metrics", self.eval_metrics),
                ("Eval Threshold", f"{self.eval_threshold:.2f}"),
                ("Augmentation", "on" if self.augmentation_enabled else "off"),
                ("Callbacks", self.callbacks),
                ("Output Dir", self.output_dir),
                ("Verbose", "yes" if self.verbose else "no"),
            ]
            max_key = max(len(k) for k, _ in rows)
            sep = "─" * (max_key + 40)
            print(f"\n{sep}")
            print("  ⚙  OptimizerPydanticSettings")
            print(sep)
            for key, val in rows:
                print(f"  {key:<{max_key}} │ {val}")
            print(f"{sep}\n")

        @classmethod
        def for_local(cls) -> OptimizerPydanticSettings:
            return cls(trainer_model="ollama/mistral:7b", max_iterations=5)

        @classmethod
        def for_quality(cls) -> OptimizerPydanticSettings:
            return cls(
                trainer_model="openai/gpt-4o",
                max_iterations=10,
                target_score=0.90,
                strategy="combined",
            )

        @classmethod
        def for_quick(cls) -> OptimizerPydanticSettings:
            return cls(
                trainer_model="ollama/mistral:7b",
                max_iterations=2,
                strategy="bootstrap_few_shot",
                verbose=False,
            )

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()


except ImportError:
    OptimizerPydanticSettings = None  # type: ignore[assignment,misc]


def generate_env_template() -> str:
    """Generate a commented ``.env`` template for all settings."""
    lines = [
        "# ── Models ──────────────────────────────────────",
        "# OPTIMIZER_TRAINER_MODEL=ollama/mistral:7b",
        "# OPTIMIZER_EVAL_MODEL=ollama/mistral:7b",
        "",
        "# ── Training ────────────────────────────────────",
        "# OPTIMIZER_MAX_ITERATIONS=5",
        "# OPTIMIZER_TARGET_SCORE=0.85",
        "# OPTIMIZER_STRATEGY=iterative_refinement",
        "# OPTIMIZER_PATIENCE=3",
        "",
        "# ── Evaluation ──────────────────────────────────",
        "# OPTIMIZER_AGENT_TYPE=auto",
        "# OPTIMIZER_EVAL_METRICS=answer_relevancy,geval",
        "# OPTIMIZER_EVAL_THRESHOLD=0.7",
        "# OPTIMIZER_CUSTOM_EVAL_CRITERIA=",
        "",
        "# ── Augmentation ────────────────────────────────",
        "# OPTIMIZER_AUGMENTATION_ENABLED=false",
        "# OPTIMIZER_AUGMENTATION_COUNT=10",
        "# OPTIMIZER_AUGMENTATION_METHODS=paraphrase,edge_case",
        "",
        "# ── Callbacks ───────────────────────────────────",
        "# OPTIMIZER_CALLBACKS=early_stopping,checkpoint,progress",
        "",
        "# ── Experiment tracking ─────────────────────────",
        "# OPTIMIZER_TRACK_EXPERIMENTS=true",
        "# OPTIMIZER_OUTPUT_DIR=optimization_results",
        "# OPTIMIZER_PROJECT_NAME=optimization",
        "",
        "# ── API Keys ────────────────────────────────────",
        "# OPENAI_API_KEY=",
        "# ANTHROPIC_API_KEY=",
        "# GOOGLE_API_KEY=",
    ]
    return "\n".join(lines)


# =====================================================================
# Integration with PlatformSettings
# =====================================================================
#
# The platform already has a comprehensive Pydantic BaseSettings system
# in :mod:`agentomatic.config.settings` (:class:`PlatformSettings`).
#
# ``OptimizerPydanticSettings`` uses ``OPTIMIZER_`` prefix while
# ``PlatformSettings`` uses nested ``__`` delimiter and ``AGENTOMATIC_``
# aliases.  They coexist — the optimiser reads from its own env vars,
# the platform reads from its own.  For a unified setup, set both in
# your ``.env``::
#
#     # Platform (agentomatic.config.settings.PlatformSettings)
#     LLM__PROVIDER=ollama
#     LLM__MODEL=mistral:7b
#     FEATURES__ENABLE_METRICS=true
#
#     # Optimiser (agentomatic.optimize.settings.OptimizerPydanticSettings)
#     OPTIMIZER_TRAINER_MODEL=ollama/mistral:7b
#     OPTIMIZER_MAX_ITERATIONS=10
#     OPTIMIZER_STRATEGY=combined
#
# Both are Pydantic :class:`~pydantic_settings.BaseSettings` subclasses
# with automatic ``.env`` loading, validation, and IDE autocomplete.
