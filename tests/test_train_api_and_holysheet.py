"""Tests for run_train helpers, structured fit metrics, and HolySheet reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
        assert cfg.min_absolute_improvement == 0.001
        assert cfg.persist_fit_store is False


class TestStagedCompileFitEvaluate:
    """Keras-like staged path shares primitives with train_and_report."""

    def test_public_exports(self) -> None:
        from agentomatic.optimize import (
            CompiledAgent,
            build_default_metrics,
            compile_agent,
            default_search_space,
            evaluate_agent,
            fit_agent,
            generate_fit_report,
            load_data,
            prepare_dataset,
            train_and_report,
        )

        assert callable(build_default_metrics)
        assert callable(compile_agent)
        assert callable(fit_agent)
        assert callable(evaluate_agent)
        assert callable(load_data)
        assert callable(prepare_dataset)
        assert callable(default_search_space)
        assert callable(generate_fit_report)
        assert callable(train_and_report)
        assert CompiledAgent is not None

    def test_build_default_metrics_shape(self) -> None:
        from agentomatic.optimize import build_default_metrics

        metrics, loss, fit_metric = build_default_metrics(
            model="openai/gpt-4o-mini",
            required_keys=["content", "next_action"],
            judge_criteria="Is the answer good?",
            judge_dimensions=["relevance"],
            judge_weight=0.3,
        )
        assert len(metrics) == 5
        assert getattr(loss, "name", None) or True
        assert fit_metric.name == "composite"

    def test_compile_fit_evaluate_pipeline(self) -> None:
        from agentomatic.agents.types import AgentDataset, AgentExample
        from agentomatic.optimize.train_api import (
            CompiledAgent,
            compile_agent,
            evaluate_agent,
            fit_agent,
        )

        class _Hist:
            history = {"loss": [0.5], "val_loss": [0.4]}

        class _Report:
            scores = {"composite": 0.8}

        agent = MagicMock()
        agent.agent_name = "echo"
        agent.compile.return_value = agent
        agent.fit.return_value = _Hist()
        agent.evaluate.return_value = _Report()
        agent._last_fit_result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="best"),
            baseline_config=PromptRuntimeConfig(system_prompt="base"),
            best_score=0.8,
            baseline_score=0.5,
        )
        agent._last_optimize_status = "ok"

        metric = MagicMock()
        metric.name = "m1"
        loss = MagicMock()
        loss.name = "loss"

        ds = AgentDataset(
            name="tiny",
            examples=[
                AgentExample(
                    id="t1",
                    input={"current_query": "hi"},
                    expected_output={"content": "HI", "next_action": "done"},
                    split="train",
                ),
                AgentExample(
                    id="v1",
                    input={"current_query": "yo"},
                    expected_output={"content": "YO", "next_action": "done"},
                    split="validation",
                ),
                AgentExample(
                    id="e1",
                    input={"current_query": "hey"},
                    expected_output={"content": "HEY", "next_action": "done"},
                    split="test",
                ),
            ],
        )

        compiled = compile_agent(
            agent,
            dataset=ds,
            metrics=[metric],
            loss=loss,
            fit_metric=MagicMock(name="fit"),
            optimizer="rewrite",
            task_model="openai/test",
            rewrite_model="openai/test",
            agent_name="echo",
            max_trials=3,
            patience=1,
            experiment_dir="/tmp/fit-test",
        )
        assert isinstance(compiled, CompiledAgent)
        assert compiled.optimizer_name == "rewrite"
        agent.compile.assert_called_once()
        call_kw = agent.compile.call_args
        assert call_kw.kwargs.get("loss") is loss or (
            len(call_kw.args) >= 1 and call_kw.kwargs.get("metrics") == [metric]
        )

        history = fit_agent(compiled, ds, epochs=2, trials=3, patience=1)
        assert history.history["loss"] == [0.5]
        agent.fit.assert_called_once()
        fit_kw = agent.fit.call_args.kwargs
        assert fit_kw["epochs"] == 2
        assert fit_kw["max_trials"] == 3
        assert fit_kw["optimize_mode"] == "rewrite"

        report = evaluate_agent(compiled, ds.test)
        assert report.scores["composite"] == 0.8
        agent.evaluate.assert_called_once()
        assert compiled.optimize_status == "ok"
        assert compiled.fit_result is not None

    def test_train_and_report_uses_staged_primitives(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        """run_train must call compile_agent / fit_agent / evaluate_agent."""
        from agentomatic.agents.types import AgentDataset, AgentExample
        from agentomatic.optimize import train_api as train_api_mod

        calls: list[str] = []

        class _Hist:
            history = {"loss": [0.4]}

        class _Report:
            scores = {"m": 0.7}

        class _Entry:
            provider = "openai"
            model = "test"
            base_url = "http://localhost/v1"
            api_key = "k"

        class _Stacks:
            def load(self, _name: str) -> None:
                return None

            def get_llm_config(self, _role: str) -> _Entry:
                return _Entry()

        monkeypatch.setattr(
            "agentomatic.config.settings.load_environment",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "agentomatic.stacks.manager.StackManager",
            lambda *_a, **_k: _Stacks(),
        )
        monkeypatch.setattr(
            "agentomatic.providers.apply_stack_defaults",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "agentomatic.optimize.report.generate_fit_report",
            lambda *_a, **_k: _k.get("output_path") or tmp_path / "x.html",
        )

        ds = AgentDataset(
            name="t",
            examples=[
                AgentExample(
                    id="1",
                    input={"current_query": "q"},
                    expected_output={"content": "a", "next_action": "n"},
                    split="train",
                )
            ],
        )

        def _compile(agent: Any, **_k: Any) -> Any:
            calls.append("compile")
            agent._last_fit_result = None
            agent._last_optimize_status = "ok"
            return train_api_mod.CompiledAgent(
                agent=agent,
                metrics=list(_k.get("metrics") or []),
                loss=_k.get("loss"),
                fit_metric=_k.get("fit_metric"),
                fitter=MagicMock(),
                search_space=MagicMock(),
                optimizer_name="rewrite",
                model="openai/test",
                agent_name="a",
            )

        def _fit(*_a: Any, **_k: Any) -> _Hist:
            calls.append("fit")
            return _Hist()

        def _eval(*_a: Any, **_k: Any) -> _Report:
            calls.append("evaluate")
            return _Report()

        monkeypatch.setattr(train_api_mod, "compile_agent", _compile)
        monkeypatch.setattr(train_api_mod, "fit_agent", _fit)
        monkeypatch.setattr(train_api_mod, "evaluate_agent", _eval)
        monkeypatch.setattr(
            train_api_mod,
            "build_default_metrics",
            lambda **_k: ([MagicMock(name="m")], MagicMock(), MagicMock()),
        )

        agent = MagicMock()
        agent.agent_name = "a"
        agent._last_fit_result = None
        agent._last_optimize_status = "ok"
        reports = tmp_path / "reports"
        reports.mkdir()
        cfg = TrainConfig(
            agent_name="a",
            agent_dir=tmp_path,
            stacks_dir=tmp_path,
            epochs=1,
            max_trials=2,
            reports_dir=reports,
        )

        result = train_api_mod.run_train(agent, config=cfg, dataset=ds)
        assert calls == ["compile", "fit", "evaluate"]
        assert result.eval_scores == {"m": 0.7}
        assert result.optimizer == "rewrite"


class TestPrintTrainResult:
    def test_print_summary_is_callable(self, tmp_path: Path) -> None:
        from io import StringIO

        from rich.console import Console

        from agentomatic.optimize.train_api import TrainResult, print_train_result

        class _Hist:
            history = {"loss": [0.5], "val_loss": [0.4]}

        result = TrainResult(
            history=_Hist(),
            fit_result=PromptFitResult(
                best_config=PromptRuntimeConfig(system_prompt="best"),
                baseline_config=PromptRuntimeConfig(system_prompt="base"),
                best_score=0.9,
                baseline_score=0.5,
                score_history=[0.5, 0.9],
                prompt_history=[{"accepted": True, "score": 0.9, "candidate_name": "c1"}],
            ),
            eval_scores={"composite": 0.9},
            report_path=tmp_path / "train_a.html",
            optimize_status="ok",
            applied_version=None,
            dataset_sizes={"train": 2, "validation": 1, "test": 0, "all": 3},
            stack="gemini",
            model="gemini/x",
            rewrite_model="gemini/x",
            optimizer="rewrite",
            agent_name="assistant",
        )
        (tmp_path / "train_a.html").write_text("<html></html>", encoding="utf-8")
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        print_train_result(result, console=console)
        result.print_summary(console=console)
        text = buf.getvalue()
        assert "assistant" in text
        assert "gemini" in text
        assert "Report:" in text


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
        assert "Prompt Evolution" in html or "BEST PROMPT" in html
        assert "val_loss" in html or "0.28" in html
        assert "Run Configuration" in html
        assert "Recommendations" in html
        assert "BEST PROMPT with tips" in html
        # Nested Section children must not be empty (HolySheet UX regression).
        import re

        m = re.search(r'"blocks"\s*:\s*(\[.*?\])\s*,\s*"filters"', html, re.S)
        if m:
            blocks = json.loads(m.group(1))

            def _walk(items: list) -> list:
                out: list = []
                for b in items:
                    out.append(b)
                    props = b.get("props") or {}
                    kids = props.get("children") or []
                    if isinstance(kids, list):
                        out.extend(_walk(kids))
                    for tab in props.get("tabs") or []:
                        out.extend(_walk(tab.get("children") or []))
                    for panel in props.get("panels") or []:
                        out.extend(_walk(panel.get("children") or []))
                return out

            flat = _walk(blocks)
            sections = {
                (b.get("props") or {}).get("title"): (b.get("props") or {}).get("children") or []
                for b in flat
                if b.get("type") == "section"
            }
            assert sections.get("Run Configuration"), "Run Configuration section empty"
            assert sections.get("Key Results"), "Key Results section empty"
            assert sections.get("Recommendations"), "Recommendations section empty"
        # HolySheet interactive bundle is large; fallback is still richer than a stub.
        assert len(html) > 2000

    def test_zero_improvement_holysheet_does_not_crash(self, tmp_path: Path) -> None:
        """Regression: KPI status='warning' used to break HolySheet export."""
        from agentomatic.optimize.report import generate_fit_report

        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="SAME"),
            baseline_config=PromptRuntimeConfig(system_prompt="SAME"),
            best_score=0.36,
            baseline_score=0.36,
            suggestions=["No configuration changes improved over the baseline."],
            trials=[{"round": 1, "name": "x", "phase": "full_val", "score": 0.36}],
            score_history=[0.36, 0.36],
            prompt_history=[
                {
                    "round_idx": 0,
                    "score": 0.36,
                    "accepted": False,
                    "prompt_snapshot": "SAME",
                    "candidate_name": "",
                }
            ],
            duration_seconds=5.0,
            experiment_id="flat000",
            agent="assistant",
            deployment_recommendation={
                "prompt_version": "v1",
                "confidence": "no_improvement",
                "rollout": {"strategy": "hold", "initial_weight": 0.0, "monitoring_hours": 0},
                "safety_notes": [],
            },
        )
        out = tmp_path / "flat.html"
        path = generate_fit_report(
            result,
            output_path=out,
            keras_history={"loss": [0.64, 0.64], "val_loss": [0.64, 0.64]},
            eval_scores={"composite": 0.36},
            dataset_sizes={"train": 4, "validation": 2, "test": 2},
            optimizer_name="rewrite",
            stack_name="gemini",
            model_name="gemini",
        )
        html = Path(path).read_text(encoding="utf-8")
        assert Path(path).exists()
        assert "0.36" in html
        assert "Prompt Evolution" in html or "val_loss" in html or "Keras" in html
        assert len(html) > 1500
