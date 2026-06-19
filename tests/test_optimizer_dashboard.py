from __future__ import annotations

import pytest

from agentomatic.optimize.dashboard import DashboardCallback
from agentomatic.optimize.events import EventData, OptimizationEvent


@pytest.mark.asyncio
async def test_dashboard_callback_collects_events():
    cb = DashboardCallback()
    assert len(cb._events) == 0

    data = EventData(agent="test", experiment_id="123")
    await cb.on_event(OptimizationEvent.RUN_START, data)

    assert len(cb._events) == 1
    assert cb._events[0] == (OptimizationEvent.RUN_START, data)


@pytest.mark.asyncio
async def test_dashboard_callback_tracks_scores():
    cb = DashboardCallback()

    await cb.on_event(
        OptimizationEvent.ROUND_END, EventData(agent="test", experiment_id="1", score=0.8)
    )

    assert cb._scores == [0.8]

    await cb.on_event(
        OptimizationEvent.STEP_COMPLETE, EventData(agent="test", experiment_id="1", score=0.9)
    )

    assert cb._scores == [0.8, 0.9]


@pytest.mark.asyncio
async def test_dashboard_callback_tracks_candidates():
    cb = DashboardCallback()

    await cb.on_event(
        OptimizationEvent.CANDIDATE_EVALUATED,
        EventData(
            agent="test",
            experiment_id="1",
            round_idx=1,
            candidate_name="cand_1",
            candidate_source="test_source",
            score=0.95,
        ),
    )

    assert len(cb._candidates) == 1
    assert cb._candidates[0]["name"] == "cand_1"
    assert cb._candidates[0]["score"] == 0.95
    await cb.on_event(
        OptimizationEvent.CANDIDATE_ACCEPTED,
        EventData(
            agent="test",
            experiment_id="1",
            candidate_name="cand_1",
        ),
    )
