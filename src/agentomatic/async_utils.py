"""Async helpers that keep a persistent event loop across sync entrypoints.

``asyncio.run`` creates a loop, runs one coroutine, then *closes* the loop.
LangChain OpenAI / httpx ``AsyncClient`` instances bind to that loop; once it
is closed, later ``ainvoke`` calls fail immediately with
``APIConnectionError: Connection error`` — the failure mode seen after
``agent.fit()`` when ``evaluate`` / epoch metrics re-enter async node handlers.

A thread-local loop that is never closed keeps those clients usable for the
rest of the process (fit → epoch logs → ``agent.evaluate``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_local = threading.local()


def _thread_loop() -> asyncio.AbstractEventLoop:
    """Return (or create) the persistent event loop for the current thread."""
    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _local.loop = loop
    return loop


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* to completion from synchronous code without closing the loop.

    When no event loop is running in the current thread, uses a thread-local
    persistent loop via :meth:`asyncio.AbstractEventLoop.run_until_complete`.
    When already inside a running loop (FastAPI, notebooks), schedules the
    coroutine on a worker thread's persistent loop so callers do not need
    ``afit`` / ``ainvoke``.

    Args:
        coro: Coroutine to drive to completion.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _thread_loop().run_until_complete(coro)

    def _runner(c: Coroutine[Any, Any, T]) -> T:
        return _thread_loop().run_until_complete(c)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner, coro).result()
