# pyright: reportMissingParameterType=none
"""An example's whole context must reach every stage of prompt optimization.

A dataset row is more than a question and an answer: it carries context
documents (RAG), other agent inputs, metadata and tags. These tests pin down
who sees what:

* the **agent** receives its inputs exactly as written — the original
  ``context`` included, never a re-serialised copy — and ``fit()`` never
  changes the dataset;
* **metrics** see the same example inside ``fit()`` as in ``evaluate()``;
* **LLM judges** see the context documents (or, for a RAG agent that
  retrieves for itself, what it retrieved) plus the other inputs, metadata
  and tags;
* the **optimizer's rewrite model** sees all of it next to each question;
* the **augmenter** sees the seed's inputs and documents, and variations
  inherit them.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentomatic.agents import (
    AgentDataset,
    AgentExample,
    BaseGraphAgent,
    CallableMetric,
    PromptFitterBridge,
)
from agentomatic.optimize import LocalJudgeMetric
from agentomatic.optimize.briefing import (
    build_full_optimization_briefing,
    format_dataset_samples,
    format_eval_io,
    format_example_context,
)
from agentomatic.optimize.config import PromptRuntimeConfig
from agentomatic.optimize.dataset import (
    DataPoint,
    Dataset,
    example_context,
    normalize_context,
)
from agentomatic.optimize.fitter import _wrap_local_agent
from agentomatic.optimize.llm_caller import LLMCaller
from agentomatic.optimize.metrics import (
    ScoreMetricAdapter,
    scoring_example_labels,
    scoring_run,
)
from agentomatic.optimize.runner import AgentRunner, RunResult, retrieval_from_output
from agentomatic.optimize.synthesizer import _seed_prompt

sys.path.insert(0, str(Path(__file__).parent))

from fake_openai_server import IMPROVED_PROMPT, FakeOpenAIServer, respond  # noqa: E402

DOC = "CTXDOC: refunds are accepted within 30 days of purchase."
TIER = "INPUT-premium-tier"
POLICY = "META-policy-v7"
TAG = "TAG-billing"
RUBRIC = "RUBRIC-cite-the-policy"
RETRIEVED = "RETRIEVED: annual plans renew on the 1st."


def rich_example(i: int, split: str = "train", **overrides: Any) -> AgentExample:
    """A row using every field: context, extra input, metadata, tags, rubric."""
    fields: dict[str, Any] = {
        "id": f"r{i}",
        "split": split,
        "input": {
            "current_query": f"Can I get a refund on order {i}?",
            "context": {"documents": [DOC], "locale": "en-GB"},
            "customer_tier": TIER,
        },
        "expected_output": {"response": "Yes, within 30 days of purchase."},
        "metadata": {"policy_version": POLICY, "difficulty": "hard"},
        "tags": [TAG, "refund"],
        "rubric": {"citation": RUBRIC},
    }
    fields.update(overrides)
    return AgentExample(**fields)


def rich_dataset(n_train: int = 4, n_val: int = 3) -> AgentDataset:
    examples = [rich_example(i) for i in range(n_train)]
    examples += [rich_example(n_train + i, "validation") for i in range(n_val)]
    return AgentDataset(name="rich", examples=examples)


def requests_for(server: FakeOpenAIServer, role: str) -> list[str]:
    """Recorded request bodies (as JSON text) the fake server answered as ``role``."""
    out = []
    for body in server.requests:
        text = json.dumps(body.get("messages") or [], ensure_ascii=False)
        if role == "judge" and "expert evaluation judge" in text:
            out.append(text)
        elif role == "augment" and "dataset augmentation expert" in text:
            out.append(text)
        elif role == "rewrite" and respond(body.get("messages") or []).endswith(IMPROVED_PROMPT):
            out.append(text)
    return out


@pytest.fixture(scope="module")
def server():
    with FakeOpenAIServer() as srv:
        yield srv


@pytest.fixture
def llm(server):
    """Route optimizer-side calls to the fake server; restore the default after."""
    before = (LLMCaller._default_base_url, LLMCaller._default_api_key)  # noqa: SLF001
    LLMCaller.configure(base_url=server.base_url, api_key="x")
    server.requests.clear()
    yield server
    LLMCaller._default_base_url, LLMCaller._default_api_key = before  # noqa: SLF001


@dataclass
class _State:
    question: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class RecordingAgent(BaseGraphAgent[_State]):
    """Answers vaguely and records every input it is given."""

    agent_name = "recorder"
    system_prompt = "You are a support assistant."

    def __init__(self, *, citations: list[dict[str, Any]] | None = None, mutate: bool = False):
        super().__init__()
        self.seen: list[dict[str, Any]] = []
        self.citations = citations
        self.mutate = mutate

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: _State) -> _State:
        state.output = {"response": "Thanks for reaching out!"}
        if self.citations is not None:
            state.output["citations"] = list(self.citations)
        return state

    def input_to_state(self, data: dict[str, Any]) -> _State:
        self.seen.append(json.loads(json.dumps(data, default=str)))
        if self.mutate and isinstance(data.get("context"), dict):
            # A careless agent editing its input in place.
            data["context"].setdefault("documents", []).append("AGENT-SCRIBBLE")
            data["context"]["scribbled"] = True
        return _State(question=data.get("current_query", ""))

    def state_to_output(self, state: _State) -> dict[str, Any]:
        return state.output


def _bridge(server: FakeOpenAIServer, metric: Any, **kwargs: Any) -> PromptFitterBridge:
    return PromptFitterBridge(
        agent_name="recorder",
        task_model="omlx/fake",
        rewrite_model="omlx/fake",
        llm_base_url=server.base_url,
        llm_api_key="x",
        optimizer="rewrite",
        metric=metric,
        max_trials=1,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TestNormalizeContext:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, []),
            ("", []),
            ("one document", ["one document"]),
            (["a", "", "b"], ["a", "b"]),
            ({"documents": ["a", "b"]}, ["a", "b"]),
            ({"documents": "a"}, ["a"]),
            ({}, []),
        ],
    )
    def test_shapes(self, raw, expected) -> None:
        assert normalize_context(raw) == expected

    def test_dict_documents_keep_other_keys_as_one_extra_document(self) -> None:
        docs = normalize_context({"documents": ["a"], "locale": "en-GB", "empty": ""})
        assert docs[0] == "a"
        assert json.loads(docs[1]) == {"locale": "en-GB"}

    def test_dict_without_documents_is_one_json_document(self) -> None:
        assert json.loads(normalize_context({"snapshot": {"k": 1}})[0]) == {"snapshot": {"k": 1}}

    def test_document_dicts_become_text_with_their_source(self) -> None:
        docs = [{"content": "x", "source": "a.md"}, {"text": "y"}, {"meta": 1}]
        assert normalize_context(docs) == ["x (source: a.md)", "y", '{"meta": 1}']
        assert normalize_context({"documents": [{"page_content": "p", "title": "t"}]}) == [
            "p (source: t)"
        ]


class TestDataPoint:
    def test_tags_round_trip_through_dicts_and_jsonl(self, tmp_path) -> None:
        point = DataPoint(query="q", context="doc", metadata={"m": 1}, tags=["t"])
        assert point.context == ["doc"]
        assert point.to_dict()["tags"] == ["t"]
        assert Dataset.from_list([point.to_dict()])[0].tags == ["t"]
        path = tmp_path / "d.jsonl"
        Dataset(points=[point]).to_jsonl(str(path))
        loaded = Dataset.from_jsonl(str(path))[0]
        assert (loaded.tags, loaded.context, loaded.metadata) == (["t"], ["doc"], {"m": 1})

    def test_a_dict_context_is_normalised(self) -> None:
        assert DataPoint(query="q", context={"documents": ["a"]}).context == ["a"]  # type: ignore[arg-type]


class TestToDatapoint:
    def test_carries_documents_tags_inputs_and_metadata(self) -> None:
        dp = rich_example(1).to_datapoint()
        assert dp.context[0] == DOC
        assert dp.tags == [TAG, "refund"]
        assert dp.metadata["invoke"]["customer_tier"] == TIER
        # The agent gets the context exactly as written.
        assert dp.metadata["invoke"]["context"] == {"documents": [DOC], "locale": "en-GB"}
        assert dp.metadata["policy_version"] == POLICY
        assert RUBRIC in (dp.expected_answer or "")

    def test_a_string_context_reaches_judges(self) -> None:
        example = AgentExample(input={"current_query": "q", "context": "plain text doc"})
        assert example.to_datapoint().context == ["plain text doc"]


class TestExampleContextView:
    def test_view_drops_plumbing_and_duplicate_context(self) -> None:
        dp = rich_example(1).to_datapoint()
        meta = {
            **dp.metadata,
            "invoke": {**dp.metadata["invoke"], "model_params": {"temperature": 0}},
            "split": "train",
            "resource_id": "r",
        }
        view = example_context(meta, dp.context, dp.tags)
        assert view["inputs"] == {
            "context": {"documents": [DOC], "locale": "en-GB"},
            "customer_tier": TIER,
        }
        assert "context" not in view  # already in the inputs
        assert view["metadata"] == {"policy_version": POLICY, "difficulty": "hard"}
        assert view["tags"] == [TAG, "refund"]

    def test_plain_dataset_context_is_kept(self) -> None:
        view = example_context({"topic": "x"}, ["doc"], None)
        assert view == {"context": ["doc"], "metadata": {"topic": "x"}}

    def test_bare_example_has_no_view(self) -> None:
        assert example_context({"split": "train", "invoke": {"current_query": "q"}}) == {}


# ---------------------------------------------------------------------------
# Runner: inputs reach the agent intact, the dataset is never changed
# ---------------------------------------------------------------------------


class TestLocalRunner:
    async def test_inputs_arrive_once_and_the_dataset_is_untouched(self) -> None:
        agent = RecordingAgent(mutate=True)
        runner = AgentRunner(agent="recorder", agent_callable=_wrap_local_agent(agent))
        points = [rich_example(i).to_datapoint().to_dict() for i in range(3)]
        snapshot = copy.deepcopy(points)

        results = await runner.run_dataset(points, prompt_override="P")
        await runner.run_dataset(points, prompt_override="P")  # a second epoch

        assert points == snapshot
        for seen in agent.seen:
            assert seen["context"]["documents"] == [DOC]
            assert seen["context"]["locale"] == "en-GB"
            assert seen["customer_tier"] == TIER
        assert results[0].metadata["example_tags"] == [TAG, "refund"]
        assert results[0].context[0] == DOC  # for judges

    async def test_plain_dataset_context_is_sent_as_documents(self) -> None:
        agent = RecordingAgent()
        runner = AgentRunner(agent="recorder", agent_callable=_wrap_local_agent(agent))
        await runner.run_dataset([{"query": "q", "context": ["d1", "d2"]}])
        assert agent.seen[0]["context"] == {"documents": ["d1", "d2"]}

    async def test_local_retrieval_is_captured(self) -> None:
        agent = RecordingAgent(citations=[{"content": RETRIEVED, "source": "kb"}])
        runner = AgentRunner(agent="recorder", agent_callable=_wrap_local_agent(agent))
        (result,) = await runner.run_dataset([{"query": "q"}])
        assert result.retrieval_context == [f"{RETRIEVED} (source: kb)"]


class TestWrapLocalAgent:
    async def test_documents_merge_into_a_copy(self) -> None:
        agent = RecordingAgent()
        call = _wrap_local_agent(agent)
        invoke = {"context": {"documents": ["a"]}}
        await call("q", context=["a", "b"], invoke=invoke)
        assert agent.seen[0]["context"]["documents"] == ["a", "b"]
        assert invoke == {"context": {"documents": ["a"]}}


class TestHttpRunner:
    @pytest.fixture
    def captured(self, monkeypatch):
        payloads: list[tuple[str, dict[str, Any]]] = []
        optimize_endpoint = {"enabled": True}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            payloads.append((request.url.path, body))
            if request.url.path.endswith("/optimize/invoke") and not optimize_endpoint["enabled"]:
                return httpx.Response(404, json={"detail": "nope"})
            return httpx.Response(
                200,
                json={
                    "response": "ok",
                    "metadata": {"retrieval_context": [RETRIEVED]},
                },
            )

        real = httpx.AsyncClient

        def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            return real(*args, transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)
        return payloads, optimize_endpoint

    @pytest.mark.parametrize("optimize_endpoint", [True, False])
    async def test_payload_carries_inputs_once(self, captured, optimize_endpoint) -> None:
        payloads, toggle = captured
        toggle["enabled"] = optimize_endpoint
        points = [rich_example(1).to_datapoint().to_dict()]
        snapshot = copy.deepcopy(points)
        runner = AgentRunner(agent="recorder", api_base="http://agent.test")

        for _ in range(3):
            (result,) = await runner.run_dataset(points, prompt_override="P")

        assert points == snapshot
        path, body = payloads[-1]
        assert path.endswith("/optimize/invoke" if optimize_endpoint else "/invoke")
        assert body["context"]["documents"] == [DOC]
        assert body["context"]["locale"] == "en-GB"
        assert body["customer_tier"] == TIER
        assert result.retrieval_context == [RETRIEVED]

    async def test_plain_dataset_documents_are_not_duplicated(self, captured) -> None:
        payloads, _ = captured
        runner = AgentRunner(agent="recorder", api_base="http://agent.test")
        await runner.run_single(
            "q", context=["d1"], invoke={"context": {"documents": ["d1", "d0"]}}
        )
        assert payloads[-1][1]["context"]["documents"] == ["d1", "d0"]


class TestRetrievalFromOutput:
    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ({"retrieval_context": ["a", "b"]}, ["a", "b"]),
            ({"citations": [{"content": "a", "source": "s"}]}, ["a (source: s)"]),
            ({"citations": [{"text": "a"}]}, ["a"]),
            ({"output": {"sources": ["u1"]}}, ["u1"]),
            ({"metadata": {"retrieval_context": ["m"]}}, ["m"]),
            ({"documents": [{"page_content": "p", "title": "t"}]}, ["p (source: t)"]),
            ({"response": "no sources"}, []),
            ("not a dict", []),
        ],
    )
    def test_shapes(self, output, expected) -> None:
        assert retrieval_from_output(output) == expected

    def test_retrieval_context_wins(self) -> None:
        output = {"citations": [{"content": "c"}], "retrieval_context": ["r"]}
        assert retrieval_from_output(output) == ["r"]


# ---------------------------------------------------------------------------
# Briefing: what the rewrite model reads
# ---------------------------------------------------------------------------


def _eval_row(score: float, **extra: Any) -> dict[str, Any]:
    dp = rich_example(1).to_datapoint()
    return {
        "query": dp.query,
        "response": "Thanks for reaching out!",
        "expected": dp.expected_answer,
        "score": score,
        "feedback": "Too vague",
        "example": example_context(dp.metadata, dp.context, dp.tags),
        **extra,
    }


class TestBriefing:
    def test_failures_and_successes_show_the_whole_example(self) -> None:
        text = format_eval_io(
            [_eval_row(0.1, retrieval_context=[RETRIEVED]), _eval_row(0.9)],
            max_failures=1,
            max_successes=1,
        )
        failure, success = text.split("### Successes")
        for block in (failure, success):
            assert f"- Context documents: {DOC}" in block
            assert TIER in block and POLICY in block and TAG in block
        assert f"- Retrieved by the agent: {RETRIEVED}" in failure

    def test_dataset_samples_show_the_whole_example(self) -> None:
        sample = rich_example(1).to_datapoint().to_dict()
        text = format_dataset_samples([sample])
        assert DOC in text and TIER in text and POLICY in text and TAG in text

    def test_dataset_sample_objects_work_too(self) -> None:
        text = format_dataset_samples([rich_example(1).to_datapoint()])
        assert DOC in text and TAG in text

    def test_long_documents_are_clipped(self) -> None:
        view = {"context": ["x" * 5000, "y" * 5000]}
        (line,) = format_example_context(view, limit=400)
        assert len(line) < 600

    def test_the_expected_answer_comes_before_the_rubric(self) -> None:
        dp = rich_example(1).to_datapoint()
        assert (dp.expected_answer or "").index("## Rubric") < dp.expected_answer.index(
            "## Expected answer"
        )  # the judge's reading order
        text = format_dataset_samples([dp])
        assert text.index("## Expected answer") < text.index("## Rubric")
        assert "Yes, within 30 days of purchase." in text

    def test_instructions_ask_to_use_context_not_copy_it(self) -> None:
        briefing = build_full_optimization_briefing(
            current_config=PromptRuntimeConfig(system_prompt="P"),
            eval_results=[_eval_row(0.1)],
        )
        assert "how to USE such context" in briefing
        bare = build_full_optimization_briefing(
            current_config=PromptRuntimeConfig(system_prompt="P"),
            eval_results=[{"query": "q", "response": "r", "score": 0.1}],
        )
        assert "how to USE such context" not in bare


# ---------------------------------------------------------------------------
# Metrics and judges
# ---------------------------------------------------------------------------


class TestScoreMetricAdapter:
    async def test_rebuilds_the_example_the_metric_sees_in_evaluate(self) -> None:
        seen: list[AgentExample] = []

        class Recorder:
            name = "rec"

            def score(self, example: AgentExample, prediction: dict[str, Any]) -> float:
                seen.append(example)
                return 1.0

        dp = rich_example(1).to_datapoint()
        run = RunResult(
            query=dp.query,
            response="r",
            context=dp.context,
            metadata={"example": dp.metadata, "example_tags": dp.tags},
        )
        with scoring_run(run):
            await ScoreMetricAdapter(Recorder()).evaluate(dp.query, "r", dp.expected_answer)
        (example,) = seen
        assert example.tags == [TAG, "refund"]
        assert example.input["customer_tier"] == TIER
        assert example.input["context"] == {"documents": [DOC], "locale": "en-GB"}
        assert example.metadata["policy_version"] == POLICY


class TestJudgeLabels:
    def test_labels_from_the_scored_run(self) -> None:
        dp = rich_example(1).to_datapoint()
        run = RunResult(
            query="q",
            response="r",
            metadata={"example": dp.metadata, "example_tags": dp.tags},
        )
        with scoring_run(run):
            labels = json.loads(scoring_example_labels(has_context=True))
            with_docs = json.loads(scoring_example_labels(has_context=False))
        assert labels == {
            "inputs": {"customer_tier": TIER},
            "metadata": {"policy_version": POLICY, "difficulty": "hard"},
            "tags": [TAG, "refund"],
        }
        assert with_docs["inputs"]["context"]["documents"] == [DOC]

    def test_no_run_no_labels(self) -> None:
        assert scoring_example_labels(has_context=False) == ""


# ---------------------------------------------------------------------------
# End to end: fit() and evaluate() against the fake model server
# ---------------------------------------------------------------------------


class TestFitEndToEnd:
    def test_agent_metric_and_rewriter_see_everything(self, llm) -> None:
        dataset = rich_dataset()
        snapshot = copy.deepcopy([e.to_dict() for e in dataset.examples])
        seen_by_metric: list[AgentExample] = []

        def metric_fn(example: AgentExample, prediction: dict[str, Any]) -> float:
            seen_by_metric.append(example)
            return 0.2

        agent = RecordingAgent(mutate=True)
        metric = CallableMetric("m", metric_fn)
        agent.compile(dataset, metrics=[metric], optimizer=_bridge(llm, metric))
        agent.fit(dataset, epochs=2, verbose=0)

        # The dataset is exactly what was written, after two epochs.
        assert [e.to_dict() for e in dataset.examples] == snapshot
        # The agent always got its inputs as written.
        assert agent.seen
        for seen in agent.seen:
            assert seen["context"]["documents"] == [DOC]
            assert seen["customer_tier"] == TIER
        # The metric saw tags + inputs on every call (fit and evaluate paths).
        assert seen_by_metric
        for example in seen_by_metric:
            assert TAG in example.tags
            assert example.input["customer_tier"] == TIER
        # The optimizer's rewrite model saw the whole example.
        rewrites = requests_for(llm, "rewrite")
        assert rewrites, llm.roles()
        joined = "\n".join(rewrites)
        for marker in (DOC, TIER, POLICY, TAG, RUBRIC):
            assert marker in joined, marker
        assert "how to USE such context" in joined

    def test_judge_sees_documents_inputs_metadata_and_tags(self, llm) -> None:
        dataset = rich_dataset()
        judge = LocalJudgeMetric(
            name="judge", model="omlx/fake", criteria="Correct and grounded?", temperature=0.0
        )
        agent = RecordingAgent()
        agent.compile(dataset, metrics=[judge], optimizer=_bridge(llm, judge))
        agent.fit(dataset, epochs=1, verbose=0)

        judged = requests_for(llm, "judge")
        assert judged
        for text in judged:
            assert "## Context Documents" in text and DOC in text
            assert "## Example inputs, metadata and tags" in text
            for marker in (TIER, POLICY, TAG):
                assert marker in text, marker

    def test_rag_agent_that_retrieves_for_itself(self, llm) -> None:
        """No documents in the data: judge and optimizer use what the agent retrieved."""
        examples = [
            AgentExample(
                id=f"q{i}",
                split="train" if i < 4 else "validation",
                input={"current_query": f"When do annual plans renew? ({i})"},
                expected_output={"response": "On the 1st."},
                tags=["renewal"],
            )
            for i in range(7)
        ]
        dataset = AgentDataset(name="rag", examples=examples)
        judge = LocalJudgeMetric(name="judge", model="omlx/fake", criteria="Grounded?")
        agent = RecordingAgent(citations=[{"content": RETRIEVED, "source": "kb"}])
        agent.compile(dataset, metrics=[judge], optimizer=_bridge(llm, judge))
        agent.fit(dataset, epochs=1, verbose=0)

        judged = requests_for(llm, "judge")
        assert judged and all(RETRIEVED in text for text in judged)
        rewrites = "\n".join(requests_for(llm, "rewrite"))
        assert f"Retrieved by the agent: {RETRIEVED}" in rewrites
        assert "renewal" in rewrites

    def test_evaluate_judges_against_retrieved_sources_too(self, llm) -> None:
        judge = LocalJudgeMetric(name="judge", model="omlx/fake", criteria="Grounded?")
        agent = RecordingAgent(citations=[{"content": RETRIEVED}])
        example = AgentExample(
            id="e1",
            input={"current_query": "When do annual plans renew?", "customer_tier": TIER},
            expected_output={"response": "On the 1st."},
            metadata={"policy_version": POLICY},
            tags=[TAG],
            split="test",
        )
        agent.evaluate([example], metrics=[judge])
        (text,) = requests_for(llm, "judge")
        for marker in (RETRIEVED, TIER, POLICY, TAG):
            assert marker in text, marker


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    def test_seed_prompt_shows_inputs_documents_and_tags(self) -> None:
        prompt = _seed_prompt(rich_example(1).to_datapoint(), "expansion", 2)
        assert "## Seed context" in prompt
        for marker in (DOC, TIER, TAG):
            assert marker in prompt
        assert POLICY not in prompt  # the seed answer's labels, not a variation's
        assert "Ask only questions the context documents answer" in prompt

    def test_bare_seed_prompt_is_unchanged(self) -> None:
        prompt = _seed_prompt(DataPoint(query="q", expected_answer="a"), "paraphrase", 2)
        assert "## Seed context" not in prompt
        assert "context documents answer" not in prompt

    def test_variations_inherit_a_copy_of_the_seed_inputs(self, llm) -> None:
        from agentomatic.optimize import prepare_dataset

        dataset = rich_dataset(n_train=2, n_val=1)
        out, _ = prepare_dataset(
            dataset,
            augment=True,
            n_examples=7,
            strategies=["paraphrase"],
            model="omlx/fake",
            llm_base_url=llm.base_url,
            llm_api_key="x",
        )
        added = [e for e in out.examples if "augmented" in e.tags]
        assert added
        seed_inputs = {id(e.input["context"]) for e in dataset.examples}
        for example in added:
            assert example.input["context"] == {"documents": [DOC], "locale": "en-GB"}
            assert example.input["customer_tier"] == TIER
            assert TAG in example.tags
            assert example.metadata["policy_version"] == POLICY  # label-preserving
            assert id(example.input["context"]) not in seed_inputs
        prompts = requests_for(llm, "augment")
        assert prompts and all(DOC in p for p in prompts)


def test_run_dataset_is_safe_under_concurrency() -> None:
    """Concurrent evaluations of shared points never see each other's edits."""
    agent = RecordingAgent(mutate=True)
    runner = AgentRunner(agent="recorder", agent_callable=_wrap_local_agent(agent), concurrency=8)
    points = [rich_example(i).to_datapoint().to_dict() for i in range(16)]
    snapshot = copy.deepcopy(points)
    asyncio.run(runner.run_dataset(points + points, prompt_override="P"))
    assert points == snapshot
    assert all(seen["context"]["documents"] == [DOC] for seen in agent.seen)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class TestReports:
    def test_context_summary(self) -> None:
        from agentomatic.optimize.holysheet_reports import _context_summary

        dp = rich_example(1).to_datapoint()
        line = _context_summary(example_context(dp.metadata, dp.context, dp.tags))
        assert line.startswith(f"tags: {TAG}, refund")
        assert "doc(s): CTXDOC" in line and f"customer_tier={TIER}" in line
        assert f"policy_version={POLICY}" in line
        assert _context_summary({}) == ""

    def test_context_column_only_when_some_example_has_context(self) -> None:
        from agentomatic.optimize.holysheet_reports import _drop_empty_context

        bare = [{"example": "a", "context": ""}, {"example": "b", "context": ""}]
        assert _drop_empty_context(bare) == [{"example": "a"}, {"example": "b"}]
        rich = [{"example": "a", "context": "tags: x"}, {"example": "b", "context": ""}]
        assert _drop_empty_context(rich) == rich

    def test_persisted_example_views_are_compact(self) -> None:
        from agentomatic.optimize.fitter import _compact_examples

        view = {
            "inputs": {"context": {"documents": ["d" * 5000] * 9}, "history": ["h" * 400]},
            "metadata": {"policy_version": POLICY, "notes": "n" * 1000},
            "tags": [TAG],
        }
        (row,) = _compact_examples([{"query": "q", "example": view}])
        compact = row["example"]
        assert len(compact["context"]) == 5
        assert all(len(doc) <= 300 for doc in compact["context"])
        assert len(compact["inputs"]["history"]) <= 300
        assert compact["metadata"]["policy_version"] == POLICY
        assert len(compact["metadata"]["notes"]) <= 300
        assert compact["tags"] == [TAG]
        assert view["inputs"]["context"]["documents"][0] == "d" * 5000  # source untouched

    def test_fit_report_shows_each_examples_context(self, llm, tmp_path) -> None:
        from agentomatic.optimize import generate_fit_report

        dataset = rich_dataset()
        test = [rich_example(90 + i, "test") for i in range(2)]
        metric = CallableMetric("m", lambda example, prediction: 0.5)
        agent = RecordingAgent()
        agent.compile(dataset, metrics=[metric], optimizer=_bridge(llm, metric))
        before = agent.evaluate(test, metrics=[metric])
        history = agent.fit(dataset, epochs=1, verbose=0)
        after = agent.evaluate(test, metrics=[metric])
        path = generate_fit_report(
            history,
            output_path=tmp_path / "r.html",
            baseline_eval=before,
            final_eval=after,
            eval_dataset=test,
        )
        html = Path(path).read_text(encoding="utf-8")
        assert f"tags: {TAG}, refund" in html
        assert f"customer_tier={TIER}" in html
        # The fitter's own per-example records carry the view too.
        result = agent._last_fit_result  # noqa: SLF001
        assert result.baseline_examples[0]["example"]["tags"] == [TAG, "refund"]
