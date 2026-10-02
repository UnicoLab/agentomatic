# pyright: reportMissingParameterType=none
"""Scaffolded templates ship datasets that use every field, and RAG uses them.

* Every family's seed ``datasets/all.jsonl`` carries tags; ``rag`` rows add
  context documents, required facts and a rubric; multi-turn ``chatbot`` rows
  carry their earlier turns.
* The ``rag`` agent answers from documents sent with the request
  (``context.documents``), cites them, and lists them under ``citations``.
* The generated ``rag`` ``train.py`` runs end to end — augmentation, fit,
  judge, report — and every stage receives the rows' documents, inputs,
  metadata and tags.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from agentomatic.agents import AgentDataset
from agentomatic.cli.commands import cli
from agentomatic.cli.project import scaffold_project
from agentomatic.cli.templates import _RAG_SYSTEM_PROMPT, _dataset_jsonl, get_template_files

sys.path.insert(0, str(Path(__file__).parent))

from fake_openai_server import FakeOpenAIServer  # noqa: E402

FAMILIES = ("basic", "full", "chatbot", "rag", "coordinator", "extraction")
RAG_AGENT = "trc_rag"


def _rows(template: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _dataset_jsonl("demo", template).splitlines()]


# ---------------------------------------------------------------------------
# Seed datasets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template", FAMILIES)
class TestSeedDatasets:
    def test_every_row_has_tags_and_labels(self, template) -> None:
        for row in _rows(template):
            assert row["tags"] and all(isinstance(t, str) and t for t in row["tags"])
            assert row["metadata"]["difficulty"] in {"easy", "medium", "hard"}
            assert row["input"]["current_query"]
            assert row["expected_output"]["response"]

    def test_rows_load_with_every_field(self, template, tmp_path) -> None:
        path = tmp_path / "all.jsonl"
        path.write_text(_dataset_jsonl("demo", template), encoding="utf-8")
        dataset = AgentDataset.from_jsonl(str(path))
        assert len(dataset.train) >= 4 and dataset.validation and dataset.test
        for example, row in zip(dataset.examples, _rows(template), strict=True):
            assert example.tags == row["tags"]
            assert example.to_datapoint().tags == row["tags"]
            assert example.metadata == row["metadata"]
            assert example.rubric == row.get("rubric", {})


class TestRagSeedRows:
    def test_every_row_carries_documents_and_a_rubric(self) -> None:
        for row in _rows("rag"):
            docs = row["input"]["context"]["documents"]
            assert docs and all(isinstance(d, str) and d for d in docs)
            assert set(row["rubric"]) == {"groundedness", "citation", "refusal"}

    def test_answerable_rows_name_the_facts_their_documents_hold(self) -> None:
        answerable = [r for r in _rows("rag") if "refusal" not in r["tags"]]
        assert answerable
        for row in answerable:
            facts = row["metadata"]["must_include"]
            docs = " ".join(row["input"]["context"]["documents"]).lower()
            for fact in facts:
                assert fact.lower() in docs or fact == "paid", (row["id"], fact)

    def test_a_refusal_row_has_documents_that_do_not_answer(self) -> None:
        (row,) = [r for r in _rows("rag") if "refusal" in r["tags"]]
        assert "phone" in row["input"]["current_query"].lower()
        assert "phone" not in " ".join(row["input"]["context"]["documents"]).lower()
        assert "do not say" in row["expected_output"]["response"]

    def test_the_judge_reference_includes_the_rubric(self, tmp_path) -> None:
        path = tmp_path / "all.jsonl"
        path.write_text(_dataset_jsonl("demo", "rag"), encoding="utf-8")
        point = AgentDataset.from_jsonl(str(path)).examples[0].to_datapoint()
        assert "groundedness" in (point.expected_answer or "")
        assert point.context == [
            "Support policy, section 2: refunds are accepted within 30 days of purchase."
        ]


class TestChatbotSeedRows:
    def test_multi_turn_rows_carry_their_history_ending_with_this_turn(self) -> None:
        multi = [r for r in _rows("chatbot") if "multi-turn" in r["tags"]]
        assert len(multi) >= 2
        for row in multi:
            messages = row["input"]["messages"]
            assert len(messages) >= 3
            assert messages[-1] == {"role": "user", "content": row["input"]["current_query"]}
            assert [m["role"] for m in messages[:-1]][::2] == ["user"] * len(messages[:-1][::2])


def test_rag_prompts_json_is_the_prompt_the_agent_runs() -> None:
    files = get_template_files("rag", "bot")
    assert json.loads(files["prompts.json"])["v1"]["system"] == _RAG_SYSTEM_PROMPT
    agent_src = " ".join(line.strip().strip('"') for line in files["agent.py"].splitlines())
    assert _RAG_SYSTEM_PROMPT.replace(" ", "") in agent_src.replace(" ", "")
    assert "### Dataset rows" in files["README.md"]
    assert "input.context.documents" in files["README.md"]


# ---------------------------------------------------------------------------
# The scaffolded RAG agent
# ---------------------------------------------------------------------------


class RecordingLLM:
    """Stand-in chat model that records every prompt it is sent."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, payload: Any, **_: Any) -> Any:
        self.prompts.append(str(payload))

        class _Reply:
            content = "Refunds are accepted within 30 days [1]."

        return _Reply()


def _purge_agent_modules() -> None:
    for mod in [m for m in sys.modules if m == "agents" or m.startswith("agents.")]:
        sys.modules.pop(mod, None)


@pytest.fixture(scope="module")
def rag_project(tmp_path_factory):
    """A scaffolded project with one ``rag`` agent, served from its ``main:app``."""
    project = tmp_path_factory.mktemp("trc") / "proj"
    scaffold_project(project, "proj", force=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(project)
        mp.setenv("AGENTOMATIC_LOG_LEVEL", "WARNING")
        result = CliRunner().invoke(cli, ["init", RAG_AGENT, "--template", "rag"])
        assert result.exit_code == 0, result.output
        _purge_agent_modules()
        mp.syspath_prepend(str(project))
        namespace: dict[str, Any] = {}
        source = (project / "main.py").read_text()
        exec(compile(source, str(project / "main.py"), "exec"), namespace)  # noqa: S102
        registered = namespace["_platform"]._registry.get(RAG_AGENT)
        assert registered is not None
        llm = RecordingLLM()
        registered.class_instance.llm = llm
        with TestClient(namespace["app"]) as client:
            yield project, client, llm
    _purge_agent_modules()


class TestRagAgent:
    def test_answers_from_documents_sent_with_the_request(self, rag_project) -> None:
        _, client, llm = rag_project
        response = client.post(
            f"/api/v1/{RAG_AGENT}/invoke",
            json={
                "query": "What is the refund window?",
                "context": {"documents": ["Refunds are accepted within 30 days."]},
            },
        )
        assert response.status_code == 200, response.text
        body = json.dumps(response.json())
        assert "Refunds are accepted within 30 days." in body
        assert "context[1]" in body
        assert "[1] Refunds are accepted within 30 days. (source: context[1])" in llm.prompts[-1]
        assert "Answer only from the numbered context documents" in llm.prompts[-1]

    def test_document_dicts_keep_their_source(self, rag_project) -> None:
        _, client, llm = rag_project
        response = client.post(
            f"/api/v1/{RAG_AGENT}/invoke",
            json={
                "query": "Who approves migrations?",
                "context": {
                    "documents": [
                        {"content": "The team lead approves.", "source": "change-policy"}
                    ]
                },
            },
        )
        assert response.status_code == 200, response.text
        assert "(source: change-policy)" in llm.prompts[-1]

    def test_without_documents_it_falls_back_to_retrieval(self, rag_project) -> None:
        _, client, llm = rag_project
        response = client.post(f"/api/v1/{RAG_AGENT}/invoke", json={"query": "Hello?"})
        assert response.status_code == 200, response.text
        assert "knowledge_base" in llm.prompts[-1]


# ---------------------------------------------------------------------------
# The generated train.py, end to end
# ---------------------------------------------------------------------------


def test_generated_rag_train_py_runs_and_every_stage_sees_the_context(rag_project) -> None:
    project, _, _ = rag_project
    with FakeOpenAIServer() as server:
        stack = project / "stacks" / "local.yaml"
        stack.write_text(
            stack.read_text().replace("http://127.0.0.1:8000/v1", server.base_url),
            encoding="utf-8",
        )
        env = {**os.environ, "AGENTOMATIC_LOG_LEVEL": "WARNING", "PYTHONUNBUFFERED": "1"}
        env.pop("AGENTOMATIC_STACK", None)
        proc = subprocess.run(  # noqa: S603
            [
                sys.executable,
                f"agents/{RAG_AGENT}/train.py",
                "--epochs",
                "1",
                "--trials",
                "1",
                "--augment",
                "--n-examples",
                "12",
            ],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        bodies = [
            "\n".join(str(m.get("content", "")) for m in b.get("messages") or [])
            for b in server.requests
        ]

    doc = "refunds are accepted within 30 days of purchase"
    agent_calls = [b for b in bodies if "Context documents:" in b and "Question:" in b]
    judge_calls = [b for b in bodies if "expert evaluation judge" in b]
    augment_calls = [b for b in bodies if "dataset augmentation expert" in b]
    rewrite_calls = [b for b in bodies if "Optimization briefing" in b]

    # The agent answered from each row's documents; the baseline it was fitted
    # from is the grounded prompt it runs (candidates then try rewrites).
    assert any(doc in b for b in agent_calls)
    assert any("Answer only from the numbered context documents" in b for b in agent_calls)
    # Judges saw documents, required facts and tags.
    assert any("## Context Documents" in b and doc in b for b in judge_calls)
    assert any("must_include" in b and '"tags"' in b for b in judge_calls)
    # The augmenter saw the seed's documents.
    assert augment_calls and all("## Seed context" in b for b in augment_calls)
    # The optimizer's rewrite model saw documents, metadata and tags.
    assert rewrite_calls
    briefing = "\n".join(rewrite_calls)
    assert "Context documents:" in briefing and doc in briefing
    assert "Example metadata:" in briefing and "Tags:" in briefing
    assert "how to USE such context" in briefing
    assert (project / "agents" / RAG_AGENT / "reports" / f"train_{RAG_AGENT}.html").exists()
