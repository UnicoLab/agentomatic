"""Tests for EvalConfig / evaluate_and_report and eval HolySheet reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from agentomatic.agents.types import AgentDataset, AgentExample, EvaluationReport, ExampleResult
from agentomatic.optimize.eval_api import (
    EvalConfig,
    resolve_eval_dataset_path,
    select_examples,
)
from agentomatic.optimize.report import generate_eval_report


def _example(eid: str, split: str = "test") -> AgentExample:
    return AgentExample(
        id=eid,
        input={
            "query": f"meta-{eid}",
            "question": f"Real question for {eid}?",
            "context": {"name": "Demo", "status": "framing"},
        },
        expected_output={
            "content": "Demo is in framing.",
            "next_action": "Confirm scope with sponsor.",
            "must_include": ["Demo", "framing"],
        },
        metadata={"judge_expected": "Ground in snapshot; propose one next step."},
        split=split,
    )


class TestSelectExamples:
    def test_split_fallbacks(self) -> None:
        ds = AgentDataset(
            name="t",
            examples=[
                _example("tr", "train"),
                _example("va", "validation"),
                _example("te", "test"),
            ],
        )
        assert [e.id for e in select_examples(ds, split="test")] == ["te"]
        assert [e.id for e in select_examples(ds, split="eval")] == ["va", "te"]
        assert [e.id for e in select_examples(ds, split="all")] == ["tr", "va", "te"]
        assert len(select_examples(ds, split="all", limit=2)) == 2

    def test_test_falls_back_to_validation(self) -> None:
        ds = AgentDataset(name="t", examples=[_example("va", "validation")])
        assert [e.id for e in select_examples(ds, split="test")] == ["va"]


class TestResolveEvalDatasetPath:
    def test_prefer_augmented(self, tmp_path: Path) -> None:
        datasets = tmp_path / "datasets"
        datasets.mkdir()
        (datasets / "all.jsonl").write_text("{}\n", encoding="utf-8")
        aug = datasets / "all.augmented.jsonl"
        aug.write_text("{}\n", encoding="utf-8")
        assert resolve_eval_dataset_path(tmp_path, prefer_augmented=True) == aug
        assert resolve_eval_dataset_path(tmp_path, prefer_augmented=False).name == "all.jsonl"


class TestEvalHolySheetReport:
    def test_generate_eval_report_writes_html(self, tmp_path: Path) -> None:
        report = EvaluationReport(
            agent_name="assistant",
            dataset_name="assistant",
            scores={"judge": 0.72, "f1": 0.55, "keywords": 0.8},
            example_results=[
                ExampleResult(
                    example_id="as_1",
                    prediction={"content": "Demo is framing.", "next_action": "Confirm"},
                    scores={"judge": 0.7, "f1": 0.5},
                    duration_ms=120.0,
                    metadata={
                        "judge": {
                            "score": 0.7,
                            "reason": "Mostly grounded but thin next step.",
                            "metadata": {
                                "motivation": "Needs a clearer actionable next step.",
                                "what_failed": ["weak next_action"],
                            },
                        }
                    },
                ),
                ExampleResult(
                    example_id="as_2",
                    scores={"judge": 0.74, "f1": 0.6},
                    error="boom",
                    duration_ms=90.0,
                ),
            ],
        )
        out = tmp_path / "eval_assistant.html"
        path = generate_eval_report(
            report,
            output_path=out,
            stack_name="gemini",
            model_name="openai/gemini-flash",
            split="test",
            dataset_sizes={"train": 4, "validation": 2, "test": 3, "all": 9},
            run_config={
                "required_keys": ["content", "next_action"],
                "judge_dimensions": ["pertinence", "groundedness"],
                "judge_criteria": "Score groundedness thoroughly.",
                "use_judge": True,
            },
        )
        assert Path(path).exists()
        html = Path(path).read_text(encoding="utf-8")
        assert "assistant" in html
        assert "0.72" in html or "judge" in html
        assert "Run Configuration" in html
        assert "Per-example" in html
        assert "Demo is framing." in html  # full prediction, not truncated snap
        assert "Needs a clearer actionable next step." in html
        assert "pertinence" in html or "required_keys" in html or "content" in html
        assert len(html) > 500


class TestOptimizeMetricAdapterQuery:
    def test_prefers_question_and_passes_context(self) -> None:
        from agentomatic.agents import OptimizeMetricAdapter

        captured: dict[str, Any] = {}

        class _FakeJudge:
            name = "judge"

            async def evaluate(self, query, response, expected=None, context=None):
                captured["query"] = query
                captured["expected"] = expected
                captured["context"] = context
                from agentomatic.optimize.metrics import EvalResult

                return EvalResult(metric_name="judge", score=0.9)

        adapter = OptimizeMetricAdapter(_FakeJudge(), name="judge")
        ex = _example("x")
        score = adapter.score(ex, {"content": "Demo framing", "next_action": "Confirm"})
        assert score == 0.9
        assert captured["query"] == "Real question for x?"
        assert captured["context"]
        assert "Demo" in str(captured["context"])
        assert "Judge guidance" in str(captured["expected"] or "")


class TestEvalConfigDefaults:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = EvalConfig(agent_name="a", agent_dir=tmp_path)
        assert cfg.split == "test"
        assert cfg.use_judge is True
        assert cfg.prefer_augmented is False


class TestRunEvalOffline:
    def test_run_eval_with_mocks(self, tmp_path: Path, monkeypatch: Any) -> None:
        import agentomatic.providers as providers_mod
        from agentomatic.optimize import eval_api

        agent_dir = tmp_path / "assistant"
        datasets = agent_dir / "datasets"
        reports = agent_dir / "reports"
        datasets.mkdir(parents=True)
        reports.mkdir()
        stacks = tmp_path / "stacks"
        stacks.mkdir()
        (stacks / "local.yaml").write_text(
            "name: local\nllm:\n  default:\n    provider: openai\n"
            "    model: fake\n    api_key: k\n    base_url: http://localhost\n",
            encoding="utf-8",
        )
        row = {
            "id": "e1",
            "split": "test",
            "input": {"question": "Hi?", "query": "meta", "context": {"name": "X"}},
            "expected_output": {
                "content": "Hello",
                "next_action": "Go",
                "must_include": ["Hello"],
            },
        }
        (datasets / "all.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

        class _Agent:
            agent_name = "assistant"

            def transform(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"content": "Hello X", "next_action": "Go next"}

            def evaluate(self, examples, metrics):
                assert len(examples) == 1
                assert metrics  # built by build_default_metrics (no judge)
                return EvaluationReport(
                    agent_name="assistant",
                    scores={"keywords": 1.0, "exact_key_match": 1.0},
                    example_results=[
                        ExampleResult(example_id="e1", scores={"keywords": 1.0}, duration_ms=1)
                    ],
                )

        monkeypatch.setattr(
            providers_mod, "apply_stack_defaults", lambda *_a, **_k: None
        )

        result = eval_api.run_eval(
            _Agent(),
            config=EvalConfig(
                agent_name="assistant",
                agent_dir=agent_dir,
                stacks_dir=stacks,
                stack="local",
                use_judge=False,
                split="test",
                report_path=reports / "eval.html",
            ),
        )
        assert result.n_examples == 1
        assert result.scores["keywords"] == 1.0
        assert Path(result.report_path).exists()
        assert "assistant" in Path(result.report_path).read_text(encoding="utf-8")
