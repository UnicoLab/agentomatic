# pyright: reportMissingParameterType=none
"""Every script in ``examples/prompt_optimization`` runs end to end.

Reported against ``train.py``: augmentation added nothing, metrics crashed
(``'OptimizeMetricAdapter' object has no attribute 'evaluate'``), and the
report showed no prompts. Each script runs here as a user would run it — a
subprocess with the shared connection flags — against a deterministic fake
model server, and its outputs are checked for the behaviour it teaches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_openai_server import IMPROVED_PROMPT, FakeOpenAIServer  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "prompt_optimization"


@pytest.fixture(scope="module")
def server():
    with FakeOpenAIServer() as srv:
        yield srv


def _run(script: str, server: FakeOpenAIServer, out: Path, *args: str) -> str:
    env = {**os.environ, "AGENTOMATIC_LOG_LEVEL": "WARNING"}
    proc = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / script),
            "--base-url",
            server.base_url,
            "--model",
            "fake-model",
            "--api-key",
            "x",
            "--out-dir",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(out),
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout


class TestHighLevelTrain:
    def test_optimizes_evaluates_and_reports(self, server, tmp_path) -> None:
        _run("train.py", server, tmp_path, "--epochs", "2", "--trials", "4")
        summary = json.loads((tmp_path / "train_summary.json").read_text())
        assert summary["prompt_changed"] is True
        assert summary["best_prompt"] == IMPROVED_PROMPT
        assert summary["validation"]["after"] > summary["validation"]["before"]
        # Measured on the untouched test split, before and after.
        for name in ("judge", "facts", "quality"):
            assert summary["test_scores_after"][name] > summary["test_scores_before"][name]

        report = json.loads((tmp_path / "train_report.json").read_text())
        assert report["initial_prompt"] != report["final_prompt"]
        assert len(report["epochs"]) == 2
        assert len(report["test_examples"]) == 6
        assert (tmp_path / "train_report.html").stat().st_size > 10_000
        # Test scores per tag, before and after.
        by_tag = summary["test_by_tag_after"]
        assert {"billing", "security", "api", "data"} <= set(by_tag)
        assert by_tag["billing"]["quality"] > summary["test_by_tag_before"]["billing"]["quality"]

    def test_augment_grows_only_train(self, server, tmp_path) -> None:
        _run(
            "train.py",
            server,
            tmp_path,
            "--epochs",
            "1",
            "--trials",
            "4",
            "--augment",
            "--n-examples",
            "40",
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "support.augmented.jsonl").read_text().splitlines()
        ]
        splits = [r["split"] for r in rows]
        assert len(rows) == 40
        assert splits.count("train") == 24
        assert (splits.count("validation"), splits.count("holdout"), splits.count("test")) == (
            6,
            4,
            6,
        )
        report = json.loads((tmp_path / "train_report.json").read_text())
        assert report["dataset_stats"]["added"] == 14


class TestLowLevelFitter:
    def test_every_knob_explicit_and_report(self, server, tmp_path) -> None:
        _run("low_level_prompt_fitter.py", server, tmp_path, "--trials", "6")
        result = json.loads((tmp_path / "low_level_result.json").read_text())
        assert result["best_score"] > result["baseline_score"]
        decisions = {t["decision"] for t in result["trials"]}
        assert "accepted" in decisions
        assert result["settings"]["min_confidence"] == 0.8
        assert result["dataset_sizes"]["holdout"] == 4
        assert (tmp_path / "low_level_report.html").exists()


class TestBuildingBlocks:
    def test_metrics_and_judges(self, server, tmp_path) -> None:
        out = _run("metrics_and_judges.py", server, tmp_path)
        assert "── 4. composite objective" in out
        # The metadata-driven metric is scored inside the composite too.
        good = next(
            line
            for line in out.splitlines()
            if line.strip().startswith("good") and "objective=" in line
        )
        assert "facts=1.00" in good
        # Metrics reading the documents and another input of the row.
        section_1 = out.split("── 1.")[1].split("── 2.")[0]
        rows = {line.split()[0]: line for line in section_1.splitlines()[1:] if line.strip()}
        assert "grounded=0.75" in rows["good"] and "grounded=0.00" in rows["vague"]
        assert "plan=1.00" in rows["good"] and "plan=0.00" in rows["vague"]

    def test_data_augmentation(self, server, tmp_path) -> None:
        _run("data_augmentation.py", server, tmp_path, "--n-examples", "30")
        rows = (tmp_path / "support.augmented.jsonl").read_text().splitlines()
        assert len(rows) == 30

    def test_anti_overfitting_rejects_the_memorizing_prompt(self, server, tmp_path) -> None:
        out = _run("anti_overfitting.py", server, tmp_path)
        lines = out.splitlines()
        memo = next(
            line
            for line in lines
            if line.strip().startswith("memorize_validation") and "rejected" in line
        )
        general = next(
            line
            for line in lines
            if line.strip().startswith("general_instruction") and "accepted" in line
        )
        assert "1.00" in memo and "0.00" in memo  # perfect on validation, nothing held out
        assert general
        assert "kept prompt: general_instruction" in out


class TestRagContext:
    def test_every_stage_sees_the_rows_context_and_the_prompt_improves(
        self, server, tmp_path
    ) -> None:
        server.requests.clear()
        out = _run("rag_context.py", server, tmp_path, "--epochs", "1", "--trials", "4")
        bodies = [
            "\n".join(str(m.get("content", "")) for m in b.get("messages") or [])
            for b in server.requests
        ]

        # The script shows each stage's view of one row.
        assert "── What each stage sees — row policy_05" in out
        assert '"customer_plan": "team"' in out
        assert "- Context documents: Audit logs are kept for 400 days" in out

        doc = "Audit logs are kept for 400 days on Enterprise and 90 days on Team."
        agent = [b for b in bodies if "Policy snippets:" in b]
        judge = [b for b in bodies if "expert evaluation judge" in b]
        rewrite = [b for b in bodies if "Optimization briefing" in b]
        # The agent answered from the row's own documents, with the plan.
        assert any(doc in b and "Customer plan: team" in b for b in agent)
        # Judges saw the documents and the plan / metadata / tags.
        assert any("## Context Documents" in b and doc in b for b in judge)
        assert any('"customer_plan": "team"' in b and '"tags"' in b for b in judge)
        # The rewrite model saw documents, inputs, metadata and tags.
        briefing = "\n".join(rewrite)
        for marker in ("Context documents:", doc, "customer_plan", "must_include", "Tags:"):
            assert marker in briefing, marker

        summary = json.loads((tmp_path / "rag_summary.json").read_text())
        assert summary["prompt_changed"] is True
        assert summary["best_prompt"] == IMPROVED_PROMPT
        for name in ("judge", "facts", "grounded", "quality"):
            assert summary["test_scores_after"][name] > summary["test_scores_before"][name]
        by_tag = summary["test_by_tag_after"]
        assert {"plan-specific", "pricing", "refusal"} <= set(by_tag)
        assert by_tag["plan-specific"]["quality"] > 0.5
        assert summary["dataset_sizes"] == {"train": 8, "validation": 5, "holdout": 3, "test": 4}
        report = (tmp_path / "rag_report.html").read_text(encoding="utf-8")
        assert "tags: security, compliance, plan-specific" in report

    def test_augmented_rows_keep_their_seed_documents(self, server, tmp_path) -> None:
        server.requests.clear()
        _run(
            "rag_context.py",
            server,
            tmp_path,
            "--epochs",
            "1",
            "--trials",
            "2",
            "--augment",
            "--n-examples",
            "30",
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "policies.augmented.jsonl").read_text().splitlines()
        ]
        added = [r for r in rows if "augmented" in r.get("tags", [])]
        assert len(rows) == 30 and len(added) == 10
        seeds = {r["id"]: r for r in rows}
        for row in added:
            seed = seeds[row["metadata"]["parent_id"]]
            assert row["split"] == "train"
            assert row["input"]["context"] == seed["input"]["context"]
            assert row["input"].get("customer_plan") == seed["input"].get("customer_plan")
            assert row["rubric"] == seed["rubric"]
        prompts = [
            "\n".join(str(m.get("content", "")) for m in b.get("messages") or [])
            for b in server.requests
        ]
        augment = [b for b in prompts if "dataset augmentation expert" in b]
        assert augment and all("## Seed context" in b for b in augment)
