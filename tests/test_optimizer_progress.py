from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from agentomatic.optimize.events import OptimizationEvent, EventData
from agentomatic.optimize.progress import LogProgressCallback, _make_sparkline

def test_make_sparkline():
    assert _make_sparkline([]) == ""
    assert _make_sparkline([0.5]) == "▅"
    scores = [0.1, 0.5, 0.9]
    spark = _make_sparkline(scores)
    assert len(spark) == 3

@pytest.mark.asyncio
async def test_log_progress_callback():
    cb = LogProgressCallback()
    
    # Run Start
    await cb.on_event(
        OptimizationEvent.RUN_START,
        EventData(agent="test_agent", experiment_id="123")
    )
    
    # Step Start
    await cb.on_event(
        OptimizationEvent.STEP_START,
        EventData(agent="test", experiment_id="123", step_idx=0, total_steps=5)
    )

    # Sample Result
    await cb.on_event(
        OptimizationEvent.SAMPLE_RESULT,
        EventData(agent="test", experiment_id="123", sample_score=0.9)
    )
    
    # Step Complete
    await cb.on_event(
        OptimizationEvent.STEP_COMPLETE,
        EventData(agent="test", experiment_id="123", step_idx=0, total_steps=5, score=0.8)
    )
    
    # Rewrite Accepted
    await cb.on_event(
        OptimizationEvent.REWRITE_ACCEPTED,
        EventData(agent="test", experiment_id="123", step_idx=0, prompt="new prompt", prompt_length=10)
    )

    # Run Complete
    await cb.on_event(
        OptimizationEvent.RUN_COMPLETE,
        EventData(agent="test", experiment_id="123", best_score=0.95, improvement=0.1)
    )
    
    # Assert successful execution
    assert True
