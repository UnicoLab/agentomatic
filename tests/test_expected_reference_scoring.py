# pyright: reportMissingParameterType=none
"""Deterministic metrics must score against the answer, not its scaffolding.

``AgentExample.to_datapoint`` builds a judge-facing reference — judge
guidance, a rubric, an ``## Expected answer`` section, the structured output
as JSON. An LLM judge reads all of it. ``ContainsMetric`` and
``ExactMatchMetric`` compare strings, so they were matching responses against
markdown headers: every candidate scored ~0, and a Keras-style ``fit()`` over
an ``AgentDataset`` reported "no improvement" forever, whatever the optimizer
proposed.
"""

from __future__ import annotations

import pytest

from agentomatic.agents.types import AgentExample
from agentomatic.optimize.metrics import (
    ContainsMetric,
    ExactMatchMetric,
    plain_expected,
)

#: What ``to_datapoint`` produces for a dataset with structured expectations.
RICH = (
    '## Expected answer\nOPT, banana\n\n## Expected structured output\n{"response": "OPT, banana"}'
)


class TestPlainExpected:
    def test_a_plain_string_passes_through(self) -> None:
        assert plain_expected("Paris") == "Paris"

    def test_the_answer_is_lifted_out_of_a_rich_reference(self) -> None:
        assert plain_expected(RICH) == "OPT, banana"

    def test_later_sections_are_not_included(self) -> None:
        assert "structured" not in (plain_expected(RICH) or "")
        assert "{" not in (plain_expected(RICH) or "")

    def test_earlier_sections_are_not_included(self) -> None:
        with_guidance = "## Judge guidance\nBe kind.\n\n" + RICH

        assert plain_expected(with_guidance) == "OPT, banana"

    def test_a_multi_line_answer_is_kept_whole(self) -> None:
        text = "## Expected answer\nline one\nline two\n\n## Rubric\n{}"

        assert plain_expected(text) == "line one\nline two"

    def test_indented_sections_are_handled(self) -> None:
        """The briefing indents the block under its label."""
        text = "  ## Expected answer\n  OPT, banana\n\n  ## Rubric\n  {}"

        assert plain_expected(text) == "OPT, banana"

    def test_the_header_match_is_case_insensitive(self) -> None:
        assert plain_expected("## EXPECTED ANSWER\nParis") == "Paris"

    def test_an_empty_section_falls_back_to_the_whole_text(self) -> None:
        text = "## Expected answer\n\n## Rubric\n{}"

        assert plain_expected(text) == text

    def test_none_and_empty_are_returned_unchanged(self) -> None:
        assert plain_expected(None) is None
        assert plain_expected("") == ""


class TestContainsMetricScoresTheAnswer:
    @pytest.mark.asyncio
    async def test_a_correct_response_scores_full_marks(self) -> None:
        result = await ContainsMetric().evaluate(
            query="capital of france",
            response="OPT: answer to capital of france banana",
            expected=RICH,
        )

        assert result.score == 1.0, result.reason

    @pytest.mark.asyncio
    async def test_a_wrong_response_still_scores_zero(self) -> None:
        result = await ContainsMetric().evaluate(
            query="capital of france", response="BASE: nothing here", expected=RICH
        )

        assert result.score == 0.0, result.reason

    @pytest.mark.asyncio
    async def test_a_plain_keyword_list_behaves_as_before(self) -> None:
        result = await ContainsMetric().evaluate(
            query="q", response="the answer is Paris, in France", expected="Paris, France"
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_a_trailing_comma_does_not_create_an_empty_keyword(self) -> None:
        """An empty keyword is 'in' every string and inflates the score."""
        result = await ContainsMetric().evaluate(
            query="q", response="nothing relevant", expected="Paris,"
        )

        assert result.score == 0.0, result.reason


class TestExactMatchScoresTheAnswer:
    @pytest.mark.asyncio
    async def test_the_ratio_reflects_the_answer_not_the_markdown(self) -> None:
        rich = await ExactMatchMetric().evaluate(query="q", response="OPT, banana", expected=RICH)
        plain = await ExactMatchMetric().evaluate(
            query="q", response="OPT, banana", expected="OPT, banana"
        )

        assert rich.score == pytest.approx(plain.score)
        assert rich.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_strict_mode_matches_the_answer(self) -> None:
        result = await ExactMatchMetric(fuzzy=False).evaluate(
            query="q", response="OPT, banana", expected=RICH
        )

        assert result.score == 1.0


class TestTheWholePathFromAnAgentExample:
    """Guard the actual failure: dataset -> datapoint -> metric."""

    @pytest.mark.asyncio
    async def test_an_agent_examples_expectation_is_scorable(self) -> None:
        example = AgentExample(
            id="e0",
            input={"current_query": "capital of france"},
            expected_output={"response": "OPT, banana"},
        )
        point = example.to_datapoint()

        result = await ContainsMetric().evaluate(
            query=point.query,
            response="OPT: answer to capital of france banana",
            expected=point.expected_answer,
        )

        assert result.score == 1.0, (
            f"expected_answer={point.expected_answer!r} scored {result.score}"
        )
