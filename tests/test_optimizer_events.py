from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentomatic.optimize.events import (
    CallbackManager,
    EventData,
    OptimizationCallback,
    OptimizationEvent,
)
from agentomatic.optimize.loop import PromptOptimizationLoop


@pytest.mark.asyncio
async def test_callback_manager():
    cb = MagicMock(spec=OptimizationCallback)
    cb.on_event = AsyncMock()

    manager = CallbackManager([cb])
    data = EventData(agent="test", experiment_id="123")

    await manager.emit(OptimizationEvent.RUN_START, data)

    cb.on_event.assert_called_once_with(OptimizationEvent.RUN_START, data)


@pytest.mark.asyncio
async def test_loop_emits_events():
    async def mock_invoke(dp, prompt):
        return {"actual_response": "dummy"}

    def mock_score(expected, actual):
        return 0.9

    loop = PromptOptimizationLoop(
        agent_name="test_agent",
        invoke_fn=mock_invoke,
        score_fn=mock_score,
        dataset_path="dummy_path.jsonl",
    )

    # Mock _load_dataset to avoid file access
    import agentomatic.optimize.loop as loop_module

    original_load = loop_module._load_dataset
    loop_module._load_dataset = lambda p: [{"query": "q", "expected_response": "e"}]  # type: ignore

    cb = MagicMock(spec=OptimizationCallback)
    cb.on_event = AsyncMock()
    loop._callbacks.add(cb)

    try:
        await loop.run("initial", steps=1)
    finally:
        loop_module._load_dataset = original_load

    # Verify key events were emitted
    events_emitted = [call.args[0] for call in cb.on_event.call_args_list]
    assert OptimizationEvent.RUN_START in events_emitted
    assert OptimizationEvent.STEP_START in events_emitted
    assert OptimizationEvent.SAMPLE_RESULT in events_emitted
    assert OptimizationEvent.STEP_COMPLETE in events_emitted
    assert OptimizationEvent.RUN_COMPLETE in events_emitted
