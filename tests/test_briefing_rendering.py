# pyright: reportMissingParameterType=none
"""A briefing's expected answer must stay readable, and stay under its label.

``AgentExample.to_datapoint`` renders an expected answer as a markdown block.
Inlined after a label, that produced ``Expected: ## Expected answer`` with the
answer itself on the next line, de-indented out of the list item it belongs
to — mangling the single field a rewrite model most needs to find.
"""

from __future__ import annotations

from agentomatic.optimize.briefing import (
    _labelled,
    format_dataset_samples,
    format_eval_io,
)

#: What ``to_datapoint`` produces for a dataset with structured expectations.
BLOCK = (
    '## Expected answer\nOPT, banana\n\n## Expected structured output\n{"response": "OPT, banana"}'
)


class TestLabelled:
    def test_a_single_line_value_stays_inline(self) -> None:
        assert _labelled("Expected", "Paris") == "Expected: Paris"

    def test_a_multi_line_value_gets_its_own_block(self) -> None:
        out = _labelled("Expected", BLOCK)

        assert out.startswith("Expected:\n")
        assert "Expected: ## Expected answer" not in out

    def test_every_line_of_the_block_is_indented_under_the_label(self) -> None:
        out = _labelled("Expected", BLOCK)

        body = [line for line in out.splitlines()[1:] if line.strip()]
        assert body, out
        assert all(line.startswith("  ") for line in body), body

    def test_indent_is_applied_to_the_label_too(self) -> None:
        out = _labelled("Expected", BLOCK, indent="   ")

        assert out.startswith("   Expected:\n")

    def test_blank_lines_stay_blank_rather_than_becoming_whitespace(self) -> None:
        out = _labelled("Expected", BLOCK)

        assert "\n\n" in out
        assert not any(line.isspace() for line in out.splitlines())

    def test_clipping_still_applies(self) -> None:
        out = _labelled("Expected", "x" * 500, 50)

        assert len(out) < 120

    def test_a_missing_value_reads_as_absent(self) -> None:
        assert _labelled("Expected", None) == "Expected: N/A"


class TestTheAnswerSurvivesInTheBriefing:
    """The whole point: a rewriter must be able to find the target token."""

    def test_dataset_samples_keep_the_answer_findable(self) -> None:
        out = format_dataset_samples([{"query": "capital of france", "expected_answer": BLOCK}])

        assert "banana" in out
        assert "Expected: ## Expected answer" not in out

    def test_failures_keep_the_answer_findable(self) -> None:
        out = format_eval_io(
            [{"query": "capital of france", "expected": BLOCK, "response": "BASE", "score": 0.0}]
        )

        assert "banana" in out
        assert "- Expected: ## Expected answer" not in out

    def test_single_line_expectations_are_unchanged(self) -> None:
        out = format_eval_io(
            [{"query": "capital of france", "expected": "Paris", "response": "BASE", "score": 0.0}]
        )

        assert "- Expected: Paris" in out
