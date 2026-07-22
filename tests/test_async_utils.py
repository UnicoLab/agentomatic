"""Tests for persistent-loop ``run_sync`` (fit → evaluate connection safety)."""

from __future__ import annotations

import asyncio

import pytest

from agentomatic.async_utils import run_sync


@pytest.mark.asyncio
async def test_run_sync_from_inside_running_loop() -> None:
    """Worker-thread path must still return the coroutine result."""

    async def _coro() -> int:
        await asyncio.sleep(0)
        return 7

    assert run_sync(_coro()) == 7


def test_run_sync_reuses_loop_across_calls() -> None:
    """Repeated sync entry must not close the loop between calls."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def _capture() -> int:
        loops.append(asyncio.get_running_loop())
        return 1

    assert run_sync(_capture()) == 1
    assert run_sync(_capture()) == 1
    assert len(loops) == 2
    assert loops[0] is loops[1]
    assert not loops[0].is_closed()
