"""Tests for run_train helpers, structured fit metrics, and HolySheet reports."""

from __future__ import annotations

import json
from pathlib import Path

from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.structured_metrics import (
    make_structured_fit_metric,
    structured_composite_score,
)
from agentomatic.optimize.train_api import TrainConfig, resolve_reports_dir


class TestStructuredCompositeMetric:
    def test_does_not_saturate_on_schema_only(self) -> None:
        expected = {
            "content": "Portail Sinistres modernise la déclaration, status framing.",
            "next_action": "Confirmer le périmètre avec Direction Sinistres.",
            "must_include": ["Portail Sinistres", "framing"],
            "must_not_include": ["million"],
        }
        weak = json.dumps({"content": "ok", "next_action": "go"})
        strong = json.dumps(
            {
                "content": "Portail Sinistres modernise la déclaration en framing.",
                "next_action": "Confirmer le périmètre avec Direction Sinistres.",
            }
        )
        w = structured_composite_score("De quoi parle ce projet ?", weak, expected)
        s = structured_composite_score("De quoi parle ce projet ?", strong, expected)
        assert w < 0.5
        assert s > w
        assert s > 0.7

    def test_custom_metric_factory(self) -> None:
        metric = make_structured_fit_metric(["content", "next_action"])
        assert metric.name == "composite"


class TestResolveReportsDir:
    def test_prefers_writable_agent_reports(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "assistant"
        agent_dir.mkdir()
        path = resolve_reports_dir(agent_dir, "assistant")
        assert path == agent_dir / "reports"
        assert path.is_dir()


class TestTrainConfigDefaults:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = TrainConfig(agent_name="a", agent_dir=tmp_path)
        assert cfg.epochs == 2
        assert cfg.optimizer == "rewrite"
        assert cfg.apply is False


class TestFitHolySheetReport:
    def test_generate_fit_report_writes_html(self, tmp_path: Path) -> None:
        from agentomatic.optimize.report import generate_fit_report

        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="BEST PROMPT with tips"),
            baseline_config=PromptRuntimeConfig(system_prompt="BASE PROMPT"),
            best_score=0.42,
            baseline_score=0.35,
            suggestions=["Improved grounding tips."],
            trials=[
                {
                    "round": 1,
                    "name": "tips_000",
                    "phase": "full_val",
                    "score": 0.42,
                    "mutation_notes": "expected tips",
                }
            ],
            score_history=[0.35, 0.40, 0.42],
            prompt_history=[
                {
                    "round_idx": 0,
                    "score": 0.40,
                    "accepted": True,
                    "next_focus": ["ground facts"],
                    "candidate_name": "tips_000",
                    "prompt_snapshot": "BEST PROMPT with tips",
                }
            ],
            duration_seconds=12.5,
            experiment_id="abc123",
            agent="assistant",
        )
        out = tmp_path / "train_assistant.html"
        path = generate_fit_report(
            result,
            output_path=out,
            keras_history={"loss": [0.4, 0.3], "val_loss": [0.45, 0.28]},
            eval_scores={"judge": 0.7, "f1": 0.5},
            dataset_sizes={"train": 4, "validation": 2, "test": 3},
            optimizer_name="rewrite",
            stack_name="gemini",
            model_name="openai/gemini-flash",
        )
        assert Path(path).exists()
        html = Path(path).read_text(encoding="utf-8")
        assert "0.42" in html or "Best" in html
        assert len(html) > 500
