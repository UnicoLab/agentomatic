"""Tests for Phase 7 additions: search-space YAML loader, sync WeightedMetric.

Covers:
- ``PromptSearchSpace.from_yaml`` / ``to_yaml`` roundtrip and error handling.
- ``load_search_space`` convenience helper.
- ``agentomatic.agents.metrics.WeightedMetric`` weighted aggregation, error
  handling, and (name, metric, weight) triple / (metric, weight) pair inputs.
- CLI ``--param`` parsing helper.
- CLI ``optimize`` command surfaces the new ``--mode`` / ``--search-space`` /
  ``--optimize-prompt`` / ``--optimize-params`` / ``--param`` options.
- Full template ships ``search_space.yaml`` and enriched train/eval scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
    WeightedMetric,
)
from agentomatic.agents.types import AgentExample
from agentomatic.cli.commands import _coerce_param_value, _parse_param_overrides, cli
from agentomatic.cli.templates import get_template_files
from agentomatic.optimize.search_space import PromptSearchSpace, load_search_space

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _example(expected: dict[str, Any] | None = None) -> AgentExample:
    return AgentExample(id="ex", input={"query": "hi"}, expected_output=expected)


# ===========================================================================
# PromptSearchSpace.from_yaml / to_yaml / load_search_space
# ===========================================================================


class TestSearchSpaceYaml:
    """Tests for the YAML loader / dumper on ``PromptSearchSpace``."""

    def test_from_yaml_roundtrip(self, tmp_path: Path) -> None:
        original = PromptSearchSpace(
            optimize_model_params=True,
            optimize_rag_params=True,
            model_param_space={"temperature": [0.0, 0.5], "top_p": [0.9, 1.0]},
            rag_param_space={"top_k": [3, 5, 10]},
            max_few_shot_examples=8,
            few_shot_selection_strategy="top_k",
        )
        yaml_path = tmp_path / "space.yaml"
        original.to_yaml(yaml_path)

        assert yaml_path.exists()
        restored = PromptSearchSpace.from_yaml(yaml_path)
        assert restored.optimize_model_params is True
        assert restored.optimize_rag_params is True
        assert restored.model_param_space == {
            "temperature": [0.0, 0.5],
            "top_p": [0.9, 1.0],
        }
        assert restored.rag_param_space == {"top_k": [3, 5, 10]}
        assert restored.max_few_shot_examples == 8
        assert restored.few_shot_selection_strategy == "top_k"

    def test_from_yaml_accepts_string_path(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "space.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "optimize_model_params": True,
                    "model_param_space": {"temperature": [0.0, 0.7]},
                },
            ),
        )
        space = PromptSearchSpace.from_yaml(str(yaml_path))
        assert space.model_param_space == {"temperature": [0.0, 0.7]}

    def test_from_yaml_ignores_unknown_keys(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "space.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "optimize_model_params": False,
                    "future_key": {"nested": [1, 2, 3]},
                },
            ),
        )
        space = PromptSearchSpace.from_yaml(yaml_path)
        assert space.optimize_model_params is False

    def test_from_yaml_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PromptSearchSpace.from_yaml(tmp_path / "does-not-exist.yaml")

    def test_from_yaml_top_level_must_be_mapping(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "space.yaml"
        yaml_path.write_text("- one\n- two\n")
        with pytest.raises(ValueError, match="mapping at the top level"):
            PromptSearchSpace.from_yaml(yaml_path)

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "space.yaml"
        yaml_path.write_text("")
        # Empty YAML => defaults
        space = PromptSearchSpace.from_yaml(yaml_path)
        assert space.optimize_model_params is True  # default

    def test_load_search_space_helper(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "space.yaml"
        PromptSearchSpace(
            model_param_space={"temperature": [0.0, 0.3, 0.7]},
        ).to_yaml(yaml_path)
        space = load_search_space(yaml_path)
        assert space.model_param_space == {"temperature": [0.0, 0.3, 0.7]}


# ===========================================================================
# WeightedMetric (sync agents metric)
# ===========================================================================


class TestWeightedMetric:
    """Tests for the sync-protocol WeightedMetric composite."""

    def test_triple_form(self) -> None:
        metric = WeightedMetric(
            [
                ("keys", ExactKeyMatchMetric(["response"]), 0.5),
                ("terms", ContainsTermsMetric(["result"]), 0.5),
            ],
        )
        score = metric.score(
            _example(),
            {"response": "we found the Result quickly"},
        )
        # Both components score 1.0 → weighted average = 1.0
        assert score == pytest.approx(1.0)
        assert metric.last_component_scores == {"keys": 1.0, "terms": 1.0}

    def test_pair_form_uses_metric_name(self) -> None:
        metric = WeightedMetric(
            [
                (ExactKeyMatchMetric(["response"]), 0.8),
                (
                    CallableMetric("truthy", lambda ex, pred: 1.0 if pred else 0.0),
                    0.2,
                ),
            ],
        )
        score = metric.score(_example(), {"response": "ok"})
        assert score == pytest.approx(1.0)
        assert set(metric.last_component_scores) == {"exact_key_match", "truthy"}

    def test_weights_are_normalised(self) -> None:
        # Weights that don't sum to 1 — result should still be in [0, 1].
        metric = WeightedMetric(
            [
                ("a", CallableMetric("a", lambda ex, pred: 1.0), 3.0),
                ("b", CallableMetric("b", lambda ex, pred: 0.0), 1.0),
            ],
        )
        score = metric.score(_example(), {})
        # (1*3 + 0*1) / (3 + 1) = 0.75
        assert score == pytest.approx(0.75)

    def test_component_error_yields_zero(self) -> None:
        def boom(example: AgentExample, prediction: dict[str, Any]) -> float:
            raise RuntimeError("fail")

        metric = WeightedMetric(
            [
                ("ok", CallableMetric("ok", lambda ex, pred: 1.0), 0.7),
                ("boom", CallableMetric("boom", boom), 0.3),
            ],
        )
        score = metric.score(_example(), {})
        assert score == pytest.approx(0.7)
        assert metric.last_component_scores["boom"] == 0.0

    def test_empty_metrics_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            WeightedMetric([])

    def test_zero_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            WeightedMetric(
                [
                    ("a", CallableMetric("a", lambda ex, pred: 1.0), 0.0),
                ],
            )

    def test_bad_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="tuples"):
            WeightedMetric([(1, 2, 3, 4)])  # type: ignore[list-item]

    def test_component_missing_score_raises(self) -> None:
        with pytest.raises(TypeError, match="Metric.score protocol"):
            WeightedMetric([("bad", object(), 1.0)])

    def test_bare_metric_defaults_to_weight_one(self) -> None:
        metric = WeightedMetric(
            [
                ExactKeyMatchMetric(["response"]),
                CallableMetric("zero", lambda ex, pred: 0.0),
            ],
        )
        score = metric.score(_example(), {"response": "ok"})
        # (1.0 + 0.0) / (1 + 1) = 0.5
        assert score == pytest.approx(0.5)


# ===========================================================================
# CLI --param parser helpers
# ===========================================================================


class TestParamOverrideParser:
    """Tests for ``_parse_param_overrides`` / ``_coerce_param_value``."""

    def test_float_values(self) -> None:
        parsed = _parse_param_overrides(("temperature=0.0,0.2,0.7",))
        assert parsed == {"temperature": [0.0, 0.2, 0.7]}

    def test_int_values(self) -> None:
        parsed = _parse_param_overrides(("top_k=3,5,10",))
        assert parsed == {"top_k": [3, 5, 10]}

    def test_mixed_types(self) -> None:
        parsed = _parse_param_overrides(("mixed=1,2.5,hello,true,none",))
        assert parsed == {"mixed": [1, 2.5, "hello", True, None]}

    def test_multiple_overrides(self) -> None:
        parsed = _parse_param_overrides(
            (
                "temperature=0.0,0.5",
                "top_p=0.9,1.0",
            ),
        )
        assert parsed == {
            "temperature": [0.0, 0.5],
            "top_p": [0.9, 1.0],
        }

    def test_missing_equals_raises(self) -> None:
        from click import BadParameter

        with pytest.raises(BadParameter):
            _parse_param_overrides(("temperature",))

    def test_empty_string_name_raises(self) -> None:
        from click import BadParameter

        with pytest.raises(BadParameter):
            _parse_param_overrides(("=1,2,3",))

    def test_empty_input_ignored(self) -> None:
        assert _parse_param_overrides(("", "temperature=0.0")) == {
            "temperature": [0.0],
        }

    def test_coerce_bool_and_none(self) -> None:
        assert _coerce_param_value("true") is True
        assert _coerce_param_value("FALSE") is False
        assert _coerce_param_value("null") is None
        assert _coerce_param_value("None") is None

    def test_coerce_negative_int(self) -> None:
        assert _coerce_param_value("-3") == -3

    def test_coerce_string_fallback(self) -> None:
        assert _coerce_param_value("hello") == "hello"


# ===========================================================================
# CLI ``optimize`` command surface
# ===========================================================================


class TestOptimizeCliSurface:
    """The ``agentomatic optimize`` command should expose the new options."""

    def test_help_lists_new_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        for opt in (
            "--mode",
            "--search-space",
            "--optimize-prompt",
            "--optimize-params",
            "--param",
        ):
            assert opt in result.output, f"missing --{opt} in --help output"

    def test_help_lists_backcompat_options(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        # Backwards-compatible options must still exist.
        for opt in ("--dataset", "--strategy", "--metrics", "--prompt", "--apply"):
            assert opt in result.output


# ===========================================================================
# Full template ships enriched files
# ===========================================================================


class TestFullTemplateEnrichments:
    """Verify the ``full`` template now ships weighted metrics + search space."""

    def test_full_ships_search_space_yaml(self) -> None:
        files = get_template_files("full", "demo_agent")
        assert "search_space.yaml" in files
        content = files["search_space.yaml"]
        # Sanity-check the YAML is valid + loadable.
        data = yaml.safe_load(content)
        assert "model_param_space" in data
        assert isinstance(data["model_param_space"], dict)

    def test_full_search_space_is_loadable(self, tmp_path: Path) -> None:
        files = get_template_files("full", "demo_agent")
        yaml_path = tmp_path / "search_space.yaml"
        yaml_path.write_text(files["search_space.yaml"])
        space = load_search_space(yaml_path)
        assert space.model_param_space
        # Total combinations should be >= 1 by construction.
        assert space.total_search_size() >= 1

    def test_full_train_uses_weighted_metric(self) -> None:
        files = get_template_files("full", "demo_agent")
        train = files["train.py"]
        assert "WeightedMetric" in train
        assert "METRICS = [" in train
        assert '"exact_response"' in train
        assert '"contains_terms"' in train
        assert '"has_output"' in train

    def test_full_eval_uses_weighted_metric(self) -> None:
        files = get_template_files("full", "demo_agent")
        eval_py = files["eval.py"]
        assert "WeightedMetric" in eval_py
        assert "METRICS = [" in eval_py

    def test_full_optimize_mentions_new_modes(self) -> None:
        files = get_template_files("full", "demo_agent")
        optimize_py = files["optimize.py"]
        assert "search_space.yaml" in optimize_py
        assert "param_search" in optimize_py
        assert "prompt_only" in optimize_py

    def test_full_makefile_targets(self) -> None:
        files = get_template_files("full", "demo_agent")
        mk = files["Makefile"]
        assert "optimize-params" in mk
        assert "optimize-gepa" in mk


class TestExtractionTemplateConsistency:
    """Extraction template should ship the standard manifest / README bundle."""

    def test_extraction_has_readme_and_prompts(self) -> None:
        files = get_template_files("extraction", "docs_extractor")
        assert "__init__.py" in files
        assert "agent.py" in files
        assert "pipeline.yaml" in files
        assert "prompts.json" in files
        assert ".env.example" in files
        assert "README.md" in files
