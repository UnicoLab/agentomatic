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
