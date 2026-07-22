"""Tests for TrainCliSettings / EvalCliSettings (env + CLI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentomatic.optimize import EvalCliSettings, TrainCliSettings
from agentomatic.optimize.train_api import TrainConfig


def test_train_cli_parse_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI flags override defaults (kebab-case, n-examples alias)."""
    monkeypatch.delenv("AGENTOMATIC_STACK", raising=False)
    cli = TrainCliSettings.parse(
        [
            "--stack",
            "gemini",
            "--epochs",
            "3",
            "--trials",
            "8",
            "--patience",
            "1",
            "--optimizer",
            "gepa_like",
            "--augment",
            "--n-examples",
            "40",
            "--persist",
            "--apply",
            "--apply-as",
            "v2_fit",
            "--min-improvement",
            "0.01",
            "--persist-fit-store",
        ]
    )
    assert cli.stack == "gemini"
    assert cli.epochs == 3
    assert cli.trials == 8
    assert cli.patience == 1
    assert cli.optimizer == "gepa_like"
    assert cli.augment is True
    assert cli.n_examples == 40
    assert cli.persist is True
    assert cli.apply is True
    assert cli.apply_as == "v2_fit"
    assert cli.min_improvement == 0.01
    assert cli.persist_fit_store is True


def test_train_cli_nr_examples_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """--nr-examples is accepted as an alias for n_examples."""
    monkeypatch.delenv("AGENTOMATIC_STACK", raising=False)
    cli = TrainCliSettings.parse(["--nr-examples", "12"])
    assert cli.n_examples == 12


def test_train_cli_env_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENTOMATIC_STACK is picked up when CLI omits --stack."""
    monkeypatch.setenv("AGENTOMATIC_STACK", "from_env")
    cli = TrainCliSettings.parse(["--epochs", "1"])
    assert cli.stack == "from_env"
    assert cli.epochs == 1


def test_to_train_config_maps_trials_and_min_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """to_train_config maps trials→max_trials and optional min_improvement."""
    monkeypatch.delenv("AGENTOMATIC_STACK", raising=False)
    cli = TrainCliSettings.parse(
        ["--stack", "local", "--trials", "9", "--min-improvement", "0.05"]
    )
    cfg = cli.to_train_config(
        agent_name="assistant",
        agent_dir=tmp_path,
        required_keys=["content", "next_action"],
        judge_dimensions=["pertinence"],
    )
    assert isinstance(cfg, TrainConfig)
    assert cfg.agent_name == "assistant"
    assert cfg.agent_dir == tmp_path
    assert cfg.stack == "local"
    assert cfg.max_trials == 9
    assert cfg.min_absolute_improvement == 0.05
    assert cfg.required_keys == ["content", "next_action"]
    assert cfg.judge_dimensions == ["pertinence"]


def test_to_train_config_default_min_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting --min-improvement keeps TrainConfig default."""
    monkeypatch.delenv("AGENTOMATIC_STACK", raising=False)
    cli = TrainCliSettings.parse([])
    cfg = cli.to_train_config(agent_name="x", agent_dir=tmp_path)
    assert cfg.min_absolute_improvement == 0.001


def test_eval_cli_parse_and_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EvalCliSettings maps --no-judge → use_judge=False."""
    monkeypatch.delenv("AGENTOMATIC_STACK", raising=False)
    cli = EvalCliSettings.parse(
        [
            "--stack",
            "gemini",
            "--split",
            "validation",
            "--limit",
            "3",
            "--no-judge",
            "--prefer-augmented",
            "--json-out",
            str(tmp_path / "out.json"),
            "--report",
            str(tmp_path / "r.html"),
            "--compiled",
            "compiled/dir",
        ]
    )
    assert cli.stack == "gemini"
    assert cli.split == "validation"
    assert cli.limit == 3
    assert cli.judge is False
    assert cli.prefer_augmented is True
    assert cli.compiled == "compiled/dir"
    cfg = cli.to_eval_config(
        agent_name="assistant",
        agent_dir=tmp_path,
        required_keys=["content"],
    )
    assert cfg.use_judge is False
    assert cfg.prefer_augmented is True
    assert cfg.split == "validation"
    assert cfg.limit == 3
    assert cfg.json_out == tmp_path / "out.json"
    assert cfg.report_path == tmp_path / "r.html"
    assert cfg.required_keys == ["content"]


def test_train_cli_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """--help prints usage and exits 0."""
    monkeypatch.delenv("AGENTOMATIC_STACK", raising=False)
    with pytest.raises(SystemExit) as exc:
        TrainCliSettings.parse(["--help"])
    assert exc.value.code == 0
