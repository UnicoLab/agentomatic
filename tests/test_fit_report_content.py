# pyright: reportMissingParameterType=none
"""The fit report must show what the optimization changed and whether it helped.

Reported: "I cannot see prompts. I cannot see scores … I cannot see what
changed in the prompts, the initial ones compared to the evolutions". Causes:

* every prompt panel was a HolySheet ``Accordion`` — its viewer renders none
  of them, so prompts, few-shot examples and per-example details were hidden;
* with ``epochs > 1`` only the LAST epoch's result was reported: its
  "baseline" was the prompt an earlier epoch had already improved, so the
  report compared the final prompt with itself and said "no improvement";
* no before/after numbers on the untouched test split, no per-example
  answers, and no reason why each candidate was accepted or rejected;
* single-line prompts diffed as one unreadable ``-``/``+`` pair.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentomatic.agents import AgentExample, History
from agentomatic.agents.types import EvaluationReport, ExampleResult
from agentomatic.optimize import PromptFitResult, generate_fit_report, merge_fit_results
from agentomatic.optimize.config import PromptRuntimeConfig
from agentomatic.optimize.holysheet_reports import _unified_diff

INITIAL = "You are a support assistant. Help the customer."
MIDDLE = "You are a support assistant. Quote the policy snippet, then answer."
FINAL = "You are a precise support assistant. Quote the policy snippet verbatim, then answer."


def _epoch(
    start: str,
    best: str,
    start_score: float,
    best_score: float,
    *,
    name: str,
    suggestions: list[str] | None = None,
) -> PromptFitResult:
    improved = best != start
    return PromptFitResult(
        best_config=PromptRuntimeConfig(system_prompt=best),
        baseline_config=PromptRuntimeConfig(system_prompt=start),
        best_score=best_score,
        baseline_score=start_score,
        baseline_holdout_score=start_score,
        holdout_score=best_score,
        metric_deltas={"quality": best_score - start_score},
        suggestions=suggestions or [],
        trials=[
            {
                "round": 1,
                "name": f"{name}_dup",
                "phase": "skipped",
                "decision": "duplicate",
                "reason": "Identical to a config already evaluated",
                "system_prompt": start,
            },
            {
                "round": 1,
                "name": name,
                "phase": "minibatch",
                "score": best_score,
                "decision": "promoted",
                "system_prompt": best if improved else start + " (rejected edit)",
            },
            {
                "round": 1,
                "name": name,
                "phase": "full_val",
                "score": best_score,
                "incumbent_score": start_score,
                "holdout_score": best_score,
                "confidence": 0.97,
                "decision": "accepted" if improved else "rejected",
                "reason": "better on validation" if improved else "Not significant",
                "system_prompt": best if improved else start + " (rejected edit)",
            },
        ],
        score_history=[start_score, best_score],
        baseline_examples=[
            {"query": "Refund window?", "expected": "30 days", "response": "Hi!", "score": 0.0}
        ],
        best_examples=[
            {"query": "Refund window?", "expected": "30 days", "response": "30 days", "score": 1}
        ],
        duration_seconds=2.0,
        agent="support_agent",
    )


def _epochs() -> list[PromptFitResult]:
    """Epoch 1 improves, epoch 2 improves again, epoch 3 finds nothing."""
    return [
        _epoch(INITIAL, MIDDLE, 0.2, 0.6, name="e1", suggestions=["Quote the policy"]),
        _epoch(MIDDLE, FINAL, 0.6, 0.8, name="e2", suggestions=["Say 'verbatim'"]),
        _epoch(FINAL, FINAL, 0.8, 0.8, name="e3", suggestions=["No improvement"]),
    ]


def _report(scores: dict[str, float], answers: dict[str, str]) -> EvaluationReport:
    return EvaluationReport(
        scores=scores,
        example_results=[
            ExampleResult(
                example_id=eid,
                prediction={"response": text},
                scores={"quality": scores["quality"]},
            )
            for eid, text in answers.items()
        ],
    )


TEST_SPLIT = [
    AgentExample(
        id="t1",
        input={"current_query": "Can I get my money back?"},
        expected_output={"response": "Within 30 days on annual plans."},
        split="test",
    )
]


class TestMergeFitResults:
    def test_spans_the_original_prompt_to_the_final_one(self) -> None:
        merged = merge_fit_results(_epochs())
        assert merged.baseline_config.system_prompt == INITIAL
        assert merged.best_config.system_prompt == FINAL
        assert merged.baseline_score == pytest.approx(0.2)
        assert merged.best_score == pytest.approx(0.8)
        assert merged.improved
        assert [t["epoch"] for t in merged.trials] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
        assert merged.duration_seconds == pytest.approx(6.0)
        assert merged.metric_deltas["quality"] == pytest.approx(0.6)

    def test_recommendations_come_from_the_epoch_that_produced_the_final_prompt(self) -> None:
        merged = merge_fit_results(_epochs())
        assert merged.suggestions == ["Say 'verbatim'"]

    def test_the_last_epoch_alone_would_claim_no_improvement(self) -> None:
        last = _epochs()[-1]
        assert not last.improved
        assert last.baseline_config.system_prompt == FINAL

    def test_inputs_are_not_modified(self) -> None:
        epochs = _epochs()
        merge_fit_results(epochs)
        assert "epoch" not in epochs[0].trials[0]
        assert epochs[-1].baseline_config.system_prompt == FINAL

    def test_needs_at_least_one_result(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            merge_fit_results([])


class TestGenerateFitReportInputs:
    def test_accepts_the_history_returned_by_fit(self, tmp_path: Path) -> None:
        history = History()
        history.record(-1, {"loss": 0.8, "val_loss": 0.8})
        history.record(0, {"loss": 0.2, "val_loss": 0.3})
        history.fit_results = _epochs()
        path = generate_fit_report(history, output_path=tmp_path / "r.html")
        sidecar = json.loads(Path(path).with_suffix(".json").read_text())
        assert sidecar["initial_prompt"] == INITIAL
        assert sidecar["final_prompt"] == FINAL
        assert sidecar["keras_history"]["val_loss"] == [0.8, 0.3]
        assert len(sidecar["epochs"]) == 3

    def test_accepts_the_fitted_agent(self, tmp_path: Path) -> None:
        agent = SimpleNamespace(_fit_results=_epochs())
        path = generate_fit_report(agent, output_path=tmp_path / "r.html")
        assert json.loads(Path(path).with_suffix(".json").read_text())["prompt_changed"]

    def test_rejects_unknown_objects_with_guidance(self) -> None:
        with pytest.raises(TypeError, match="History returned by agent.fit"):
            generate_fit_report(object())

    def test_says_why_when_there_is_nothing_to_report(self) -> None:
        with pytest.raises(ValueError, match="_last_optimize_status"):
            generate_fit_report(History())


class TestReportContent:
    @pytest.fixture
    def report(self, tmp_path: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        path = generate_fit_report(
            _epochs(),
            output_path=tmp_path / "train_report.html",
            baseline_eval=_report({"quality": 0.1}, {"t1": "Thanks for reaching out!"}),
            final_eval=_report({"quality": 0.9}, {"t1": "Refunds within 30 days."}),
            eval_dataset=TEST_SPLIT,
            dataset_stats={"requested": 10, "added": 8, "by_strategy": {"paraphrase": 8}},
        )
        html = Path(path).read_text(encoding="utf-8")
        sidecar = json.loads(Path(path).with_suffix(".json").read_text())
        return html, sidecar, _blocks(html)

    def test_sidecar_has_before_after_on_the_test_split(self, report) -> None:
        _, sidecar, _ = report
        assert sidecar["validation"] == {"initial": 0.2, "final": 0.8}
        assert sidecar["test"] == {"before": {"quality": 0.1}, "after": {"quality": 0.9}}
        row = sidecar["test_examples"][0]
        assert row["question"] == "Can I get my money back?"
        assert row["before"] == "Thanks for reaching out!"
        assert row["after"] == "Refunds within 30 days."
        assert row["Δ"] == "quality +0.80"

    def test_sections_exist_and_are_not_empty(self, report) -> None:
        _, _, blocks = report
        sections = {
            (b.get("props") or {}).get("title"): (b.get("props") or {}).get("children")
            for b in blocks
            if b.get("type") == "section"
        }
        for title in (
            "Verdict",
            "Test scoreboard",
            "What changed in the prompt",
            "Epochs",
            "All candidates",
            "Examples — before vs after",
            "Data & settings",
        ):
            assert sections.get(title), f"{title!r} missing or empty"

    def test_prompts_are_visible_not_hidden_in_accordions(self, report) -> None:
        html, _, blocks = report
        assert not [b for b in blocks if b.get("type") == "accordion"]
        code = "\n".join(
            str((b.get("props") or {}).get("code", ""))
            for b in blocks
            if b.get("type") == "code_block"
        )
        assert "-You are a support assistant. Help the customer." in code
        assert "verbatim" in code
        assert "(rejected edit)" in code  # rejected candidates are shown too
        assert "Thanks for reaching out!" in html

    def test_every_candidate_decision_and_reason_is_listed(self, report) -> None:
        _, _, blocks = report
        rows = [
            row
            for b in blocks
            if b.get("type") == "table" or "data" in (b.get("props") or {})
            for row in (b.get("props") or {}).get("data") or []
            if isinstance(row, dict) and "decision" in row
        ]
        decisions = {(r["epoch"], r["candidate"]): r["decision"] for r in rows}
        assert decisions[(1, "e1")] == "accepted"
        assert decisions[(3, "e3")] == "rejected"
        assert decisions[(1, "e1_dup")] == "duplicate"
        assert any("Not significant" in r["reason"] for r in rows)


class TestFallbackHtml:
    def test_without_holysheet_the_report_still_shows_before_after(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import agentomatic.optimize.holysheet_reports as hs

        def _missing(*_a: Any, **_k: Any) -> str:
            raise ImportError("holysheet")

        monkeypatch.setattr(hs, "build_fit_holysheet_report", _missing)
        path = generate_fit_report(
            _epochs(),
            output_path=tmp_path / "r.html",
            baseline_eval_scores={"quality": 0.1},
            eval_scores={"quality": 0.9},
        )
        html = Path(path).read_text(encoding="utf-8")
        assert "Test Scoreboard" in html and "+0.8000" in html
        assert "Prompt Diff (initial → final)" in html
        assert "Not significant" in html


class TestReadableDiff:
    def test_long_single_line_prompts_diff_sentence_by_sentence(self) -> None:
        before = "You are helpful. " * 12 + "Answer briefly."
        after = "You are helpful. " * 12 + "Answer briefly and quote the policy."
        diff = _unified_diff(before, after)
        changed = [
            line
            for line in diff.splitlines()
            if line[:1] in "+-" and line[:3] not in ("+++", "---")
        ]
        assert len(changed) == 2
        assert all(len(line) <= 101 for line in diff.splitlines())

    def test_identical_prompts_have_no_diff(self) -> None:
        assert _unified_diff("same", "same") == ""


def _blocks(html: str) -> list[dict[str, Any]]:
    """Every serialized HolySheet block (recursively) embedded in the page."""
    match = re.search(r'"blocks"\s*:\s*(\[.*?\])\s*,\s*"filters"', html, re.S)
    assert match, "report JSON not found"

    def walk(items: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for block in items:
            if not isinstance(block, dict):
                continue
            out.append(block)
            props = block.get("props") or {}
            out.extend(walk(props.get("children") or []))
            for tab in props.get("tabs") or []:
                out.extend(walk(tab.get("children") or []))
            for panel in props.get("panels") or []:
                out.extend(walk(panel.get("children") or []))
        return out

    return walk(json.loads(match.group(1)))
