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
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any, TypeVar

T = TypeVar("T")

_local = threading.local()

# Event loop that handed the current worker-thread call off via
# :func:`run_in_worker_thread`. Lets sync code running in that thread (a graph
# node, say) schedule work back onto the loop that is serving the request.
_owner_loop: ContextVar[asyncio.AbstractEventLoop | None] = ContextVar(
    "agentomatic_owner_loop", default=None
)


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


async def run_in_worker_thread(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a blocking callable in a worker thread without stalling the loop.

    Sync graph nodes routinely make blocking LLM / HTTP calls. Running them
    inline on the event loop froze every other request for the duration —
    an "async" task submission only returned once the model had answered,
    and task polling, SSE progress and health checks all stalled with it.

    The call runs under a copy of the caller's context (``asyncio.to_thread``
    semantics), so ContextVars such as the task-progress context stay
    visible, and :func:`owner_loop` reports the loop that offloaded it.

    Args:
        fn: Blocking callable.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        Whatever ``fn`` returns.
    """
    loop = asyncio.get_running_loop()

    def _call() -> T:
        _owner_loop.set(loop)
        return fn(*args, **kwargs)

    return await asyncio.to_thread(_call)


def owner_loop() -> asyncio.AbstractEventLoop | None:
    """Return the loop that offloaded the current worker-thread call, if any.

    Returns:
        The loop captured by :func:`run_in_worker_thread`, or ``None`` when
        the caller was not started through it.
    """
    return _owner_loop.get()
