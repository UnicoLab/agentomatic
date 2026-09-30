# pyright: reportMissingParameterType=none
"""``prepare_dataset(augment=True)`` must actually add usable, leak-free rows.

Reported: "prompt optimization doesn't augment the data". Causes found:

* one call per strategy asked a local model for dozens of examples under
  ``max_tokens=2000``; the reply was cut off and parsed to *zero* rows with
  no warning — only a clean, complete JSON array was understood;
* synthetic rows got a fabricated ``{"content": …, "next_action": "Follow up
  with stakeholder."}`` label whatever the seed schema was, and lost the
  seed's metadata — so every metric scored them wrong;
* rows were only de-duplicated against train seeds (copies of test questions
  leaked into train), and without a train split, test rows became seeds;
* ``LLMCaller.configure`` was left set process-wide; ``augmented=True`` was
  reported even when nothing was added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agentomatic.agents import AgentDataset, AgentExample
from agentomatic.optimize import load_data, prepare_dataset
from agentomatic.optimize.llm_caller import LLMCaller
from agentomatic.optimize.synthesizer import parse_generated_points

sys.path.insert(0, str(Path(__file__).parent))

from fake_openai_server import FakeOpenAIServer  # noqa: E402

EXAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "prompt_optimization"
    / "datasets"
    / "support.jsonl"
)


@pytest.fixture(scope="module")
def server():
    with FakeOpenAIServer() as srv:
        yield srv


class TestTheReportedCall:
    def test_the_example_script_call_adds_rows_and_persists(self, server, tmp_path) -> None:
        """The exact ``prepare_dataset`` call from the user's training script."""
        dataset = load_data(EXAMPLE_DATASET)
        out, written = prepare_dataset(
            dataset,
            seed_path=EXAMPLE_DATASET,
            augment=True,
            n_examples=30,
            persist=True,
            persist_path=tmp_path / "all_augmented.jsonl",
            model="omlx/fake-model",
            llm_base_url=server.base_url,
            llm_api_key=None,
            strategies=["expansion", "paraphrase"],
        )
        stats = out.metadata["augment_stats"]
        assert len(out.examples) == 30
        assert stats["added"] == 14 and out.metadata["augmented"] is True
        assert set(stats["by_strategy"]) == {"expansion", "paraphrase"}
        assert written == tmp_path / "all_augmented.jsonl"
        assert len(written.read_text().splitlines()) == 30

    def test_only_train_grows(self, server) -> None:
        dataset = load_data(EXAMPLE_DATASET)
        out, _ = _augment(dataset, server, n_examples=24)
        assert len(out.train) == len(dataset.train) + 8
        assert [e.id for e in out.validation] == [e.id for e in dataset.validation]
        assert [e.id for e in out.test] == [e.id for e in dataset.test]

    def test_no_process_wide_endpoint_side_effect(self, server) -> None:
        before = (LLMCaller._default_base_url, LLMCaller._default_api_key)  # noqa: SLF001
        _augment(load_data(EXAMPLE_DATASET), server, n_examples=20)
        assert (LLMCaller._default_base_url, LLMCaller._default_api_key) == before  # noqa: SLF001


class TestRowsMirrorTheSeed:
    def test_label_preserving_rows_keep_answer_and_metadata(self, server) -> None:
        dataset = load_data(EXAMPLE_DATASET)
        out, _ = _augment(dataset, server, n_examples=20, strategies=["paraphrase"])
        seeds = {e.id: e for e in dataset.examples}
        new = [e for e in out.examples if e.metadata.get("source") == "augment"]
        assert new
        for example in new:
            seed = seeds[example.metadata["parent_id"]]
            assert example.expected_output == seed.expected_output
            assert example.metadata["must_include"] == seed.metadata["must_include"]
            assert set(example.input) == set(seed.input)
            assert example.split == "train"

    def test_generated_answers_use_the_seed_schema(self, server) -> None:
        out, _ = _augment(
            load_data(EXAMPLE_DATASET), server, n_examples=20, strategies=["expansion"]
        )
        new = [e for e in out.examples if e.metadata.get("source") == "augment"]
        assert new
        for example in new:
            assert set(example.expected_output) == {"response"}
            assert "next_action" not in json.dumps(example.to_dict())
            assert example.expected_output["response"].startswith("Generated answer")


class TestLeakage:
    def test_near_duplicates_of_test_questions_are_dropped(self, server) -> None:
        train = AgentExample(
            id="t", input={"current_query": "How do I reset my password?"}, split="train"
        )
        test = AgentExample(
            id="x",
            input={"current_query": "Variant 999: how do i reset my password?"},
            split="test",
        )
        dataset = AgentDataset(examples=[train, test])
        out, _ = _augment(dataset, server, n_examples=4, strategies=["paraphrase"])
        stats = out.metadata["augment_stats"]
        # Every variant the fake model writes is a near-copy of the test question.
        assert stats["near_duplicates"] >= 1
        assert not [e for e in out.examples if e.metadata.get("source") == "augment"]

    def test_without_a_train_split_nothing_is_generated(self, server) -> None:
        dataset = AgentDataset(
            examples=[AgentExample(id="v", input={"current_query": "q"}, split="validation")]
        )
        out, _ = _augment(dataset, server, n_examples=5)
        assert len(out.examples) == 1
        assert out.metadata["augmented"] is False

    def test_strict_mode_raises_when_it_falls_short(self, server) -> None:
        dataset = AgentDataset(
            examples=[AgentExample(id="v", input={"current_query": "q"}, split="validation")]
        )
        with pytest.raises(RuntimeError, match="Augmentation added 0/"):
            _augment(dataset, server, n_examples=5, strict=True)


class TestPersistence:
    def test_persist_only_never_overwrites_the_seed_file(self, tmp_path) -> None:
        seed = tmp_path / "all.jsonl"
        seed.write_text(EXAMPLE_DATASET.read_text())
        before = seed.read_text()
        _, written = prepare_dataset(load_data(seed), persist=True, seed_path=seed)
        assert seed.read_text() == before
        assert written == tmp_path / "all.prepared.jsonl"


class TestReplyParsing:
    ARRAY = '[{"query": "a?", "expected_answer": "x"}, {"query": "b?", "expected_answer": "y"}]'

    @pytest.mark.parametrize(
        ("reply", "count"),
        [
            (ARRAY, 2),
            ("Sure [ok]:\n```json\n" + ARRAY + "\n```\nHope [this] helps", 2),
            ('{"examples": ' + ARRAY + "}", 2),
            ('{"query": "a?", "expected_answer": "x"}', 1),
            ('{"query": "a?"}\n{"query": "b?"}', 2),
            (ARRAY[:-30], 1),
            ("<think>[x]</think>" + ARRAY, 2),
            ("no json at all", 0),
        ],
    )
    def test_reply_shapes(self, reply: str, count: int) -> None:
        assert len(parse_generated_points(reply)) == count

    def test_truncated_replies_are_salvaged_during_augmentation(self, monkeypatch) -> None:
        async def truncated(*args: Any, **kwargs: Any) -> str:
            return '[{"query": "What is the refund window?", "expected_answer": "30 days"}, {"query": "Wha'

        monkeypatch.setattr(LLMCaller, "call", staticmethod(truncated))
        dataset = AgentDataset(
            examples=[
                AgentExample(
                    id="s",
                    input={"current_query": "Refund window?"},
                    expected_output={"response": "30 days"},
                    split="train",
                )
            ]
        )
        out, _ = prepare_dataset(dataset, augment=True, n_examples=2, model="omlx/x")
        assert len(out.examples) == 2


def _augment(dataset: Any, server: FakeOpenAIServer, **kwargs: Any) -> tuple[Any, Any]:
    kwargs.setdefault("strategies", ["paraphrase", "expansion"])
    return prepare_dataset(
        dataset,
        augment=True,
        model="omlx/fake-model",
        llm_base_url=server.base_url,
        llm_api_key="x",
        **kwargs,
    )
