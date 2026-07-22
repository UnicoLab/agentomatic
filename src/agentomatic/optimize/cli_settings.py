"""Pydantic settings for thin train/eval scripts (env + CLI overrides).

Keeps scaffolded ``train.py`` / ``eval.py`` flat: load settings → build agent
→ ``train_and_report`` / ``evaluate_and_report``. Knobs come from
``AGENTOMATIC_*`` env vars and optional kebab-case CLI flags (backward
compatible with the former argparse surface).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentomatic.optimize.eval_api import EvalConfig
from agentomatic.optimize.train_api import TrainConfig

OptimizerName = Literal[
    "rewrite",
    "gepa_like",
    "mipro_like",
    "few_shot_bootstrap",
    "param_search",
]

EvalSplitName = Literal["test", "validation", "train", "eval", "all"]


class TrainCliSettings(BaseSettings):
    """Env + CLI knobs for ``train_and_report`` scripts.

    Environment variables use the ``AGENTOMATIC_`` prefix (e.g.
    ``AGENTOMATIC_STACK``, ``AGENTOMATIC_EPOCHS``). CLI flags are kebab-case
    without the prefix (``--stack``, ``--n-examples``, ``--persist-fit-store``).

    Example::

        cli = TrainCliSettings.parse()  # sys.argv[1:] + env
        config = cli.to_train_config(
            agent_name="assistant",
            agent_dir=HERE,
            stacks_dir=ROOT / "stacks",
            env_path=ROOT / ".env",
            required_keys=["content", "next_action"],
        )
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTOMATIC_",
        cli_prefix="",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_exit_on_error=True,
        extra="ignore",
        populate_by_name=True,
    )

    stack: str = Field(
        default="local",
        description="Stack name (env: AGENTOMATIC_STACK)",
    )
    epochs: int = Field(default=2, description="Outer Keras-style epochs")
    trials: int = Field(
        default=12,
        description="Inner PromptFitter trial budget → max_trials",
    )
    patience: int = Field(
        default=2,
        description="EarlyStopping patience on val_loss",
    )
    optimizer: OptimizerName = Field(
        default="rewrite",
        description=(
            "Fitter optimizer: rewrite | gepa_like | mipro_like | "
            "few_shot_bootstrap | param_search"
        ),
    )
    dataset: Path | None = Field(
        default=None,
        description="JSONL path (default datasets/all.jsonl)",
    )
    augment: bool = Field(
        default=False,
        description="LLM-augment seed data before fit",
    )
    n_examples: int | None = Field(
        default=None,
        validation_alias=AliasChoices("n_examples", "nr_examples"),
        description="Target example count after augment (default: 3× seed)",
    )
    persist: bool = Field(
        default=False,
        description="Write datasets/all.augmented.jsonl",
    )
    apply: bool = Field(
        default=False,
        description="Persist best prompt if improved",
    )
    apply_as: str | None = Field(
        default=None,
        description="prompts.json version key",
    )
    min_improvement: float | None = Field(
        default=None,
        description="Min absolute score lift (default TrainConfig 0.001)",
    )
    persist_fit_store: bool = Field(
        default=False,
        description=(
            "Also write retrain artefacts to DATABASE_URL / "
            "AGENTOMATIC_FIT_STORE_URL"
        ),
    )

    @classmethod
    def parse(cls, argv: list[str] | None = None) -> Self:
        """Load from env + optional CLI overrides (``sys.argv[1:]`` by default).

        Args:
            argv: CLI argument list (without program name). ``--help`` prints
                usage and raises ``SystemExit``.

        Returns:
            Parsed settings instance.
        """
        args = list(sys.argv[1:] if argv is None else argv)
        return cls(_cli_parse_args=args)

    def to_train_config(
        self,
        *,
        agent_name: str,
        agent_dir: Path,
        stacks_dir: Path | None = None,
        env_path: Path | None = None,
        required_keys: list[str] | None = None,
        judge_criteria: str | None = None,
        judge_dimensions: list[str] | None = None,
        **overrides: Any,
    ) -> TrainConfig:
        """Build a :class:`TrainConfig` from CLI/env knobs + agent metadata.

        Args:
            agent_name: Agent name for artefacts / reports.
            agent_dir: Agent package directory.
            stacks_dir: Optional stacks directory.
            env_path: Optional ``.env`` path.
            required_keys: Structured output keys (TrainConfig default if omitted).
            judge_criteria: LLM-as-judge criteria (TrainConfig default if omitted).
            judge_dimensions: Judge dimension names (TrainConfig default if omitted).
            **overrides: Extra :class:`TrainConfig` fields (win over CLI knobs).

        Returns:
            Ready-to-use :class:`TrainConfig`.
        """
        kwargs: dict[str, Any] = {
            "agent_name": agent_name,
            "agent_dir": Path(agent_dir),
            "stacks_dir": stacks_dir,
            "env_path": env_path,
            "stack": self.stack,
            "dataset_path": self.dataset,
            "epochs": self.epochs,
            "max_trials": self.trials,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "augment": self.augment,
            "n_examples": self.n_examples,
            "persist": self.persist,
            "apply": self.apply,
            "apply_as": self.apply_as,
            "persist_fit_store": self.persist_fit_store,
        }
        if required_keys is not None:
            kwargs["required_keys"] = list(required_keys)
        if judge_criteria is not None:
            kwargs["judge_criteria"] = judge_criteria
        if judge_dimensions is not None:
            kwargs["judge_dimensions"] = list(judge_dimensions)
        if self.min_improvement is not None:
            kwargs["min_absolute_improvement"] = self.min_improvement
        kwargs.update(overrides)
        return TrainConfig(**kwargs)


class EvalCliSettings(BaseSettings):
    """Env + CLI knobs for ``evaluate_and_report`` scripts.

    Environment variables use the ``AGENTOMATIC_`` prefix (e.g.
    ``AGENTOMATIC_STACK``). CLI flags are kebab-case without the prefix
    (``--split``, ``--prefer-augmented``, ``--judge`` / ``--no-judge``).
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTOMATIC_",
        cli_prefix="",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_exit_on_error=True,
        extra="ignore",
        populate_by_name=True,
    )

    stack: str = Field(
        default="local",
        description="Stack name (env: AGENTOMATIC_STACK)",
    )
    dataset: Path | None = Field(default=None, description="JSONL path")
    split: EvalSplitName = Field(
        default="test",
        description="Dataset split: test | validation | train | eval | all",
    )
    limit: int | None = Field(
        default=None,
        description="Optional cap on the number of examples",
    )
    judge: bool = Field(
        default=True,
        description="Enable LLM-as-judge (use --no-judge to skip)",
    )
    prefer_augmented: bool = Field(
        default=False,
        description="Use datasets/all.augmented.jsonl when present",
    )
    json_out: Path | None = Field(
        default=None,
        description="Optional JSON summary path beside the HTML report",
    )
    report: Path | None = Field(
        default=None,
        description="Explicit HTML report path (default: timestamped under reports/)",
    )
    compiled: str | None = Field(
        default=None,
        description="Optional compiled agent directory to load before eval",
    )

    @classmethod
    def parse(cls, argv: list[str] | None = None) -> Self:
        """Load from env + optional CLI overrides (``sys.argv[1:]`` by default).

        Args:
            argv: CLI argument list (without program name). ``--help`` prints
                usage and raises ``SystemExit``.

        Returns:
            Parsed settings instance.
        """
        args = list(sys.argv[1:] if argv is None else argv)
        return cls(_cli_parse_args=args)

    def to_eval_config(
        self,
        *,
        agent_name: str,
        agent_dir: Path,
        stacks_dir: Path | None = None,
        env_path: Path | None = None,
        required_keys: list[str] | None = None,
        judge_criteria: str | None = None,
        judge_dimensions: list[str] | None = None,
        **overrides: Any,
    ) -> EvalConfig:
        """Build an :class:`EvalConfig` from CLI/env knobs + agent metadata.

        Args:
            agent_name: Agent name for artefacts / reports.
            agent_dir: Agent package directory.
            stacks_dir: Optional stacks directory.
            env_path: Optional ``.env`` path.
            required_keys: Structured output keys (EvalConfig default if omitted).
            judge_criteria: LLM-as-judge criteria (EvalConfig default if omitted).
            judge_dimensions: Judge dimension names (EvalConfig default if omitted).
            **overrides: Extra :class:`EvalConfig` fields (win over CLI knobs).

        Returns:
            Ready-to-use :class:`EvalConfig`.
        """
        kwargs: dict[str, Any] = {
            "agent_name": agent_name,
            "agent_dir": Path(agent_dir),
            "stacks_dir": stacks_dir,
            "env_path": env_path,
            "stack": self.stack,
            "dataset_path": self.dataset,
            "report_path": self.report,
            "split": self.split,
            "limit": self.limit,
            "use_judge": self.judge,
            "prefer_augmented": self.prefer_augmented,
            "json_out": self.json_out,
        }
        if required_keys is not None:
            kwargs["required_keys"] = list(required_keys)
        if judge_criteria is not None:
            kwargs["judge_criteria"] = judge_criteria
        if judge_dimensions is not None:
            kwargs["judge_dimensions"] = list(judge_dimensions)
        kwargs.update(overrides)
        return EvalConfig(**kwargs)


__all__ = [
    "EvalCliSettings",
    "EvalSplitName",
    "OptimizerName",
    "TrainCliSettings",
]
