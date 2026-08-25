# pyright: reportMissingParameterType=none
"""A score of 0.0000 must not be reported when nothing was evaluated.

Every agent call failing (server down, wrong host, bad credentials) produced
per-call warnings and then a confident summary::

    Baseline score: 0.0000
    Best score:     0.0000
    ❌ Recommendation: keep the baseline config (no improvement).

An operator reads that as "my prompt scores zero" and starts fixing a prompt
that was never exercised. The number is now labelled as not being a
measurement, and the reason is raised ahead of every other advisory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentomatic.optimize.dataset import DataPoint
from agentomatic.optimize.fitter import PromptFitter
from agentomatic.optimize.metrics import ContainsMetric


@dataclass
class _Run:
    """Stand-in for a runner ``RunResult``."""

    query: str
    response: str = ""
    expected: str | None = None
    error: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    steps_taken: list[Any] = field(default_factory=list)
    reasoning: str = ""
    retrieval_context: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    trace: Any = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class _Runner:
    """Runner that returns whatever results the test hands it."""

    def __init__(self, results: list[_Run]) -> None:
        self._results = results

    async def run_dataset(self, points, **_: Any) -> list[_Run]:
        del points
        return self._results


def _fitter() -> PromptFitter:
    return PromptFitter(agent="blackout", auto_report=False, concurrency=1)


POINTS = [
    DataPoint(query="capital of france", expected_answer="Paris"),
    DataPoint(query="capital of japan", expected_answer="Tokyo"),
]


class TestBlackoutIsDetected:
    @pytest.mark.asyncio
    async def test_all_failed_sets_the_blackout(self) -> None:
        fitter = _fitter()
        fitter._runner = _Runner(  # noqa: SLF001
            [
                _Run(query="capital of france", error="Client error '404 Not Found'"),
                _Run(query="capital of japan", error="Client error '404 Not Found'"),
            ]
        )

        score, _dims, _details = await fitter._evaluate_config(  # noqa: SLF001
            fitter._load_baseline_config(),  # noqa: SLF001
            POINTS,
            ContainsMetric(),
        )

        assert score == 0.0
        assert "not a measurement" in fitter._eval_blackout  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_the_first_error_is_quoted(self) -> None:
        fitter = _fitter()
        fitter._runner = _Runner(  # noqa: SLF001
            [_Run(query="q", error="Connection refused to http://localhost:8000")]
        )

        await fitter._evaluate_config(  # noqa: SLF001
            fitter._load_baseline_config(),  # noqa: SLF001
            POINTS[:1],
            ContainsMetric(),
        )

        assert "Connection refused" in fitter._eval_blackout  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_the_call_count_is_reported(self) -> None:
        fitter = _fitter()
        fitter._runner = _Runner([_Run(query=f"q{i}", error="boom") for i in range(3)])  # noqa: SLF001

        await fitter._evaluate_config(  # noqa: SLF001
            fitter._load_baseline_config(),  # noqa: SLF001
            POINTS,
            ContainsMetric(),
        )

        assert "all 3 agent call(s) failed" in fitter._eval_blackout  # noqa: SLF001


class TestTheCauseIsNamedAccurately:
    """ "Agent never answered" and "metric could not score" need different fixes."""

    @pytest.mark.asyncio
    async def test_a_metric_failure_is_not_blamed_on_the_agent(self) -> None:
        class _Exploding(ContainsMetric):
            async def evaluate(self, *a: Any, **k: Any):  # noqa: ANN201
                raise RuntimeError("metric blew up")

        fitter = _fitter()
        fitter._runner = _Runner(  # noqa: SLF001
            [_Run(query="capital of france", response="Paris", expected="Paris")]
        )

        await fitter._evaluate_config(  # noqa: SLF001
            fitter._load_baseline_config(),  # noqa: SLF001
            POINTS[:1],
            _Exploding(),
        )

        blackout = fitter._eval_blackout  # noqa: SLF001
        assert "metric scored none" in blackout, blackout
        assert "agent call(s) failed" not in blackout, blackout
        assert "metric blew up" in blackout, blackout


class TestAPartialRunIsNotABlackout:
    """One bad datapoint is normal — the average is still a real measurement."""

    @pytest.mark.asyncio
    async def test_one_success_clears_it(self) -> None:
        fitter = _fitter()
        fitter._runner = _Runner(  # noqa: SLF001
            [
                _Run(query="capital of france", response="Paris", expected="Paris"),
                _Run(query="capital of japan", error="boom"),
            ]
        )

        score, _dims, _details = await fitter._evaluate_config(  # noqa: SLF001
            fitter._load_baseline_config(),  # noqa: SLF001
            POINTS,
            ContainsMetric(),
        )

        assert score == 1.0
        assert not fitter._eval_blackout  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_a_genuine_zero_is_not_flagged(self) -> None:
        """Every answer wrong is a real result and must stay unqualified."""
        fitter = _fitter()
        fitter._runner = _Runner(  # noqa: SLF001
            [_Run(query="capital of france", response="Berlin", expected="Paris")]
        )

        score, _dims, _details = await fitter._evaluate_config(  # noqa: SLF001
            fitter._load_baseline_config(),  # noqa: SLF001
            POINTS[:1],
            ContainsMetric(),
        )

        assert score == 0.0
        assert not fitter._eval_blackout  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_an_empty_run_is_not_flagged(self) -> None:
        """No results at all is a different condition, handled elsewhere."""
        fitter = _fitter()
        fitter._runner = _Runner([])  # noqa: SLF001

        await fitter._evaluate_config(  # noqa: SLF001
            fitter._load_baseline_config(),  # noqa: SLF001
            [],
            ContainsMetric(),
        )

        assert not fitter._eval_blackout  # noqa: SLF001


def test_a_fresh_fitter_starts_clear() -> None:
    assert _fitter()._eval_blackout == ""  # noqa: SLF001
