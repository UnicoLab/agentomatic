"""Tests for the dynamic ``MapStepConfig`` fan-out step.

Covers:

* Fan-out over a runtime list resolved from the pipeline context.
* Concurrency is bounded by ``max_concurrency`` (Semaphore).
* Keyed fan-in preserves per-index results (no ``dict.update`` overwrite).
* Per-item retry via the shared ``execute_with_retry`` helper.
* Progress and checkpoint callbacks fire once per completed item.
* Empty items list yields an early SUCCESS result.
* Wrong ``items`` type is a graceful, structured failure.
* YAML loader discriminator recognises the ``map`` key.
* Full end-to-end run through :class:`PipelineEngine`.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentomatic.pipelines.context import PipelineContext
from agentomatic.pipelines.engine import PipelineEngine
from agentomatic.pipelines.loader import PipelineLoader
from agentomatic.pipelines.models import (
    MapStepConfig,
    PipelineConfig,
    PipelineStatus,
    RetryConfig,
    StepStatus,
)
from agentomatic.pipelines.steps import execute_map_step


def _make_registry(node_fn):  # type: ignore[no-untyped-def]
    """Build a mock registry that returns one agent with the supplied ``node_fn``."""
    registry = MagicMock()
    agent = MagicMock()
    agent.node_fn = node_fn
    agent.graph_fn = None
    agent.schema_validator = None
    registry.get.return_value = agent
    registry.list_names.return_value = ["extractor"]
    return registry


# ---------------------------------------------------------------------------
# Fan-out & keyed fan-in
# ---------------------------------------------------------------------------


class TestMapFanOut:
    """Fan-out semantics and keyed aggregation."""

    @pytest.mark.asyncio
    async def test_fan_out_over_items(self):
        """One agent invocation per item, all successes aggregated."""
        seen: list[str] = []

        async def node_fn(state):  # type: ignore[no-untyped-def]
            seen.append(state["scope"])
            return {"response": f"ok_{state['scope']}", "scope": state["scope"]}

        registry = _make_registry(AsyncMock(side_effect=node_fn))
        ctx = PipelineContext(input_data={"scopes": ["a", "b", "c"]})

        config = MapStepConfig(
            name="extract_all",
            agent="extractor",
            items="$.input.scopes",
            item_key="scope",
            max_concurrency=3,
        )
        result = await execute_map_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert result.sub_results is not None
        assert len(result.sub_results) == 3
        assert sorted(seen) == ["a", "b", "c"]
        # Keyed fan-in preserves per-index results — no dict.update overwrite.
        assert result.output["count"] == 3
        assert result.output["succeeded"] == 3
        assert set(result.output["by_key"].keys()) == {"0", "1", "2"}
        assert result.output["items"][0]["output"]["scope"] == "a"
        assert result.output["items"][2]["output"]["scope"] == "c"

    @pytest.mark.asyncio
    async def test_empty_items_yields_success(self):
        """An empty items list is a no-op success (no agent invocation)."""
        node_fn = AsyncMock(return_value={"response": "should not be called"})
        registry = _make_registry(node_fn)
        ctx = PipelineContext(input_data={"scopes": []})

        config = MapStepConfig(
            name="noop",
            agent="extractor",
            items="$.input.scopes",
        )
        result = await execute_map_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert result.output["count"] == 0
        assert result.output["items"] == []
        assert result.output["by_key"] == {}
        assert not node_fn.called

    @pytest.mark.asyncio
    async def test_non_list_items_fails_gracefully(self):
        """A non-list ``items`` expression fails without crashing the engine."""
        registry = _make_registry(AsyncMock())
        ctx = PipelineContext(input_data={"not_a_list": "hello"})

        config = MapStepConfig(
            name="bad",
            agent="extractor",
            items="$.input.not_a_list",
        )
        result = await execute_map_step(config, ctx, registry)

        assert result.status == StepStatus.FAILED
        assert "did not resolve to a list" in (result.error or "")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestMapConcurrency:
    """The semaphore actually caps in-flight items."""

    @pytest.mark.asyncio
    async def test_concurrency_is_bounded(self):
        inflight = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def node_fn(state):  # type: ignore[no-untyped-def]
            nonlocal inflight, max_seen
            async with lock:
                inflight += 1
                max_seen = max(max_seen, inflight)
            try:
                await asyncio.sleep(0.02)
                return {"response": "ok", "scope": state["scope"]}
            finally:
                async with lock:
                    inflight -= 1

        registry = _make_registry(AsyncMock(side_effect=node_fn))
        ctx = PipelineContext(input_data={"scopes": list(range(8))})

        config = MapStepConfig(
            name="cap",
            agent="extractor",
            items="$.input.scopes",
            item_key="scope",
            max_concurrency=2,
        )
        result = await execute_map_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert result.output["count"] == 8
        assert max_seen <= 2


# ---------------------------------------------------------------------------
# Retry & progress
# ---------------------------------------------------------------------------


class TestMapRetryAndProgress:
    """Per-item retry via ``execute_with_retry`` + progress/checkpoint hooks."""

    @pytest.mark.asyncio
    async def test_per_item_retry_recovers(self):
        attempts: dict[str, int] = {}

        async def node_fn(state):  # type: ignore[no-untyped-def]
            scope = state["scope"]
            attempts[scope] = attempts.get(scope, 0) + 1
            if scope == "flaky" and attempts[scope] < 3:
                raise RuntimeError("transient failure")
            return {"response": "ok", "scope": scope}

        registry = _make_registry(AsyncMock(side_effect=node_fn))
        ctx = PipelineContext(input_data={"scopes": ["stable", "flaky"]})

        config = MapStepConfig(
            name="retrying",
            agent="extractor",
            items="$.input.scopes",
            item_key="scope",
            max_concurrency=2,
            retry=RetryConfig(max_attempts=3, backoff="fixed", base_delay=0.1),
        )
        result = await execute_map_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert attempts["flaky"] == 3

    @pytest.mark.asyncio
    async def test_progress_and_checkpoint_callbacks(self):
        async def node_fn(state):  # type: ignore[no-untyped-def]
            return {"response": "ok", "scope": state["scope"]}

        registry = _make_registry(AsyncMock(side_effect=node_fn))
        ctx = PipelineContext(input_data={"scopes": ["a", "b", "c", "d"]})

        progress_events: list[tuple[int, int]] = []
        checkpoint_events: list[int] = []

        async def progress_cb(current, total, message):  # type: ignore[no-untyped-def]
            progress_events.append((current, total))

        async def checkpoint_cb(index, sub_result):  # type: ignore[no-untyped-def]
            checkpoint_events.append(index)

        config = MapStepConfig(
            name="cb",
            agent="extractor",
            items="$.input.scopes",
            item_key="scope",
            max_concurrency=2,
        )
        result = await execute_map_step(
            config,
            ctx,
            registry,
            progress_cb=progress_cb,
            checkpoint_cb=checkpoint_cb,
        )

        assert result.status == StepStatus.SUCCESS
        assert len(progress_events) == 4
        assert progress_events[-1] == (4, 4)
        assert sorted(checkpoint_events) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestMapLoader:
    """YAML/dict loader recognises the ``map`` discriminator."""

    def test_loader_parses_map_step(self):
        raw = {
            "name": "extraction_pipeline",
            "steps": [
                {
                    "name": "extract_all",
                    "map": {
                        "agent": "extractor",
                        "items": "$.input.scopes",
                        "item_key": "scope",
                        "max_concurrency": 8,
                        "retry": {"max_attempts": 2, "backoff": "linear", "base_delay": 0.5},
                        "input": {"markdown": "$.steps.to_md.output.path"},
                    },
                }
            ],
        }
        config = PipelineLoader.from_dict(raw)
        assert isinstance(config.steps[0], MapStepConfig)
        step = config.steps[0]
        assert step.agent == "extractor"
        assert step.items == "$.input.scopes"
        assert step.item_key == "scope"
        assert step.max_concurrency == 8
        assert step.retry is not None
        assert step.retry.max_attempts == 2
        assert step.retry.backoff == "linear"
        assert step.input.mappings["markdown"] == "$.steps.to_md.output.path"


# ---------------------------------------------------------------------------
# End-to-end via PipelineEngine
# ---------------------------------------------------------------------------


class TestMapPipelineEngine:
    """Engine dispatches to the map executor and progress reaches callers."""

    @pytest.mark.asyncio
    async def test_engine_runs_map_step(self):
        async def node_fn(state):  # type: ignore[no-untyped-def]
            return {"response": "ok", "scope": state["scope"]}

        registry = _make_registry(AsyncMock(side_effect=node_fn))
        config = PipelineConfig(
            name="extraction",
            steps=[
                MapStepConfig(
                    name="extract_all",
                    agent="extractor",
                    items="$.input.scopes",
                    item_key="scope",
                    max_concurrency=2,
                )
            ],
        )
        engine = PipelineEngine(config, registry)
        assert engine.validate() == []

        progress_events: list[dict] = []
        checkpoint_events: list[tuple[str, int]] = []

        async def progress_cb(**kwargs):  # type: ignore[no-untyped-def]
            progress_events.append(kwargs)

        async def checkpoint_cb(step_name, index, sub_result):  # type: ignore[no-untyped-def]
            checkpoint_events.append((step_name, index))

        result = await engine.run(
            {"scopes": ["parties", "dates", "amounts"]},
            progress_cb=progress_cb,
            checkpoint_cb=checkpoint_cb,
        )

        assert result.status == PipelineStatus.SUCCESS
        step_result = result.steps["extract_all"]
        assert step_result.status == StepStatus.SUCCESS
        assert step_result.output["count"] == 3
        assert set(step_result.output["by_key"].keys()) == {"0", "1", "2"}
        # Both progress and checkpoint hooks fired for every item + step boundary.
        assert len(checkpoint_events) == 3
        assert any(evt.get("event") == "map_item_completed" for evt in progress_events)
        assert any(evt.get("event") == "step" for evt in progress_events)
