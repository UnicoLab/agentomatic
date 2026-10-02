"""Agent runner — invokes agents with prompt overrides for evaluation.

Supports both remote (HTTP API) and local (direct import) invocation.
Uses the dedicated ``/optimize/invoke`` endpoint when available, falling
back to ``/invoke`` for full pipeline context capture (retrieval docs,
tool calls, reasoning steps).
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from agentomatic.optimize.dataset import document_text

#: Keys an agent output (or response ``metadata``) may list its sources under.
_RETRIEVAL_KEYS = ("retrieval_context", "citations", "sources", "documents")


def retrieval_from_output(data: Any) -> list[str]:
    """Documents an agent output says it answered from.

    A RAG agent that retrieves for itself reports its sources in its output;
    surfacing them lets judges score groundedness and lets the optimizer see
    what each answer was built from.

    Args:
        data: An agent output / response dict. ``retrieval_context`` wins,
            then ``citations``, ``sources`` and ``documents`` — at the top
            level, under ``output`` or under ``metadata``. Items may be
            strings or dicts (``content`` / ``text`` / ``page_content`` …).

    Returns:
        The documents as strings; empty when the output lists none.
    """
    if not isinstance(data, dict):
        return []
    containers = [data]
    for key in ("output", "metadata"):
        nested = data.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for key in _RETRIEVAL_KEYS:
        for container in containers:
            items = container.get(key)
            if isinstance(items, str):
                items = [items]
            if isinstance(items, list) and items:
                docs = [d for d in (document_text(i) for i in items) if d]
                if docs:
                    return docs
    return []


def _copy_inputs(invoke: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of a point's agent inputs (shallow when not copyable)."""
    import copy

    try:
        return copy.deepcopy(invoke)
    except Exception:  # noqa: BLE001 - uncopyable user objects: shallow copy
        return dict(invoke)


def _merge_documents(existing: Any, extra: list[str]) -> list[Any]:
    """``existing`` documents plus those of ``extra`` not already present."""
    if isinstance(existing, str):
        docs: list[Any] = [existing]
    elif isinstance(existing, (list, tuple)):
        docs = list(existing)
    else:
        docs = []
    return docs + [d for d in extra if d not in docs]


def _response_text(data: dict[str, Any]) -> str:
    """Prefer structured ``output`` dict, else string ``response``."""

    output = data.get("output")
    if isinstance(output, dict) and output:
        return json.dumps(output, ensure_ascii=False)
    response = data.get("response", "")
    if not isinstance(response, str):
        return json.dumps(response, ensure_ascii=False)
    return response or str(data)


@dataclass
class RunResult:
    """Result of running a single data point through an agent."""

    query: str
    response: str
    expected: str | None = None
    context: list[str] = field(default_factory=list)
    # Full pipeline context (from /optimize/invoke)
    retrieval_context: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps_taken: list[str] = field(default_factory=list)
    reasoning: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AgentRunner:
    """Runs dataset points through an agent endpoint.

    Supports:
    - Remote: HTTP calls to the platform API
    - Local: Direct callable invocation (no HTTP server required)
    - Optimize: Dedicated ``/optimize/invoke`` endpoint with full context

    When *agent_callable* is provided the runner bypasses all HTTP calls
    and invokes the callable directly.  The callable must accept::

        async def my_fn(
            query: str,
            *,
            prompt_override: str | None,
            context: list[str] | None,
            invoke: dict[str, Any] | None,
        ) -> str: ...

    Sync callables are also accepted — they are run via
    ``asyncio.to_thread`` so the event loop is never blocked.

    The runner automatically tries ``/optimize/invoke`` first (HTTP mode),
    falling back to ``/invoke`` if the optimization endpoint is not
    available.
    """

    def __init__(
        self,
        agent: str,
        api_base: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        timeout: float = 60.0,
        use_optimize_endpoint: bool = True,
        agent_callable: Callable[..., str | Awaitable[str]] | None = None,
        concurrency: int = 5,
        n_runners: int | None = None,
    ):
        self.agent = agent
        self.api_base = api_base
        self.api_prefix = api_prefix
        self.timeout = timeout
        self.use_optimize_endpoint = use_optimize_endpoint
        self.agent_callable = agent_callable
        self._optimize_available: bool | None = None  # Auto-detect
        # n_runners is an Agent Lightning–style alias for concurrency.
        self.concurrency = max(1, int(n_runners if n_runners is not None else concurrency))
        self.n_runners = self.concurrency

    async def run_single(
        self,
        query: str,
        prompt_override: str | None = None,
        context: list[str] | None = None,
        invoke: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run a single query through the agent.

        When *agent_callable* is set the call is dispatched locally
        (no HTTP).  Otherwise tries ``/optimize/invoke`` first for full
        context, falling back to ``/invoke`` if not available.

        Args:
            query: User query string.
            prompt_override: Optional system prompt override.
            context: Optional retrieval / document context list.
            invoke: Extra top-level invoke fields (e.g. agent-specific
                inputs from ``metadata.invoke`` on a dataset point).
        """
        if self.agent_callable is not None:
            return await self._run_local(query, prompt_override, context, invoke=invoke)

        if self.use_optimize_endpoint and self._optimize_available is not False:
            result = await self._run_optimize(query, prompt_override, context, invoke=invoke)
            if result.error and "404" in result.error:
                # Optimize endpoint not available, fall back
                self._optimize_available = False
                logger.info(
                    f"Optimize endpoint not available for '{self.agent}', falling back to /invoke"
                )
                return await self._run_invoke(query, prompt_override, context, invoke=invoke)
            self._optimize_available = True
            return result
        return await self._run_invoke(query, prompt_override, context, invoke=invoke)

    async def _run_local(
        self,
        query: str,
        prompt_override: str | None = None,
        context: list[str] | None = None,
        invoke: dict[str, Any] | None = None,
    ) -> RunResult:
        """Invoke the agent via the local callable (no HTTP)."""
        import asyncio
        import inspect

        t0 = time.perf_counter()
        try:
            fn = self.agent_callable
            if inspect.iscoroutinefunction(fn):
                raw = await fn(
                    query,
                    prompt_override=prompt_override,
                    context=context,
                    invoke=invoke,
                )
            else:
                # fn is sync — run in a thread so the event loop stays free.
                # Cast away the Awaitable branch: iscoroutinefunction already
                # ruled it out, but mypy needs an explicit hint.
                sync_fn: Callable[..., str] = fn  # type: ignore[assignment]
                raw = await asyncio.to_thread(
                    sync_fn,
                    query,
                    prompt_override=prompt_override,
                    context=context,
                    invoke=invoke,
                )
            duration = (time.perf_counter() - t0) * 1000
            if isinstance(raw, dict):
                response = _response_text(raw)
            else:
                response = str(raw) if raw is not None else ""
            return RunResult(
                query=query,
                response=response,
                # What a RAG agent says it answered from (judges, optimizer).
                retrieval_context=retrieval_from_output(raw),
                duration_ms=duration,
                # The full structured output, for metrics that score fields
                # other than the response text (see ScoreMetricAdapter).
                metadata={"output": raw} if isinstance(raw, dict) else {},
            )
        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            logger.warning(f"Local agent call failed for '{query[:50]}': {exc}")
            return RunResult(
                query=query,
                response="",
                duration_ms=duration,
                error=str(exc),
            )

    async def _run_optimize(
        self,
        query: str,
        prompt_override: str | None = None,
        context: list[str] | None = None,
        invoke: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run via /optimize/invoke for full pipeline context."""
        import httpx

        t0 = time.perf_counter()
        try:
            payload: dict[str, Any] = {
                "query": query,
                "user_id": "optimizer",
                "include_retrieval_context": True,
                "include_steps": True,
            }
            if invoke:
                payload.update({k: v for k, v in invoke.items() if k not in {"query", "user_id"}})
            if prompt_override:
                payload["system_prompt_override"] = prompt_override
            if context:
                ctx = payload.get("context")
                ctx = dict(ctx) if isinstance(ctx, dict) else {}
                ctx["documents"] = _merge_documents(ctx.get("documents"), list(context))
                payload["context"] = ctx

            async with httpx.AsyncClient(base_url=self.api_base, timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.api_prefix}/{self.agent}/optimize/invoke",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            duration = (time.perf_counter() - t0) * 1000
            return RunResult(
                query=query,
                response=_response_text(data),
                retrieval_context=retrieval_from_output(data),
                tool_calls=data.get("tool_calls", []),
                steps_taken=data.get("steps_taken", []),
                reasoning=data.get("reasoning", ""),
                citations=data.get("citations", []),
                duration_ms=duration,
                metadata=data.get("metadata", {}),
            )
        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            return RunResult(
                query=query,
                response="",
                duration_ms=duration,
                error=str(exc),
            )

    async def _run_invoke(
        self,
        query: str,
        prompt_override: str | None = None,
        context: list[str] | None = None,
        invoke: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run via standard /invoke endpoint."""
        import httpx

        t0 = time.perf_counter()
        try:
            payload: dict[str, Any] = {
                "query": query,
                "user_id": "optimizer",
            }
            if invoke:
                payload.update({k: v for k, v in invoke.items() if k not in {"query", "user_id"}})
            ctx: dict[str, Any] = {}
            if isinstance(payload.get("context"), dict):
                ctx.update(payload.pop("context"))
            if prompt_override:
                ctx["system_prompt_override"] = prompt_override
            if context:
                docs = _merge_documents(ctx.get("documents"), list(context))
                if docs:
                    ctx["documents"] = docs
            if ctx:
                payload["context"] = ctx

            async with httpx.AsyncClient(base_url=self.api_base, timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.api_prefix}/{self.agent}/invoke",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            duration = (time.perf_counter() - t0) * 1000
            return RunResult(
                query=query,
                response=_response_text(data),
                # Sources from the response (``metadata.retrieval_context``,
                # ``output.citations``, …).
                retrieval_context=retrieval_from_output(data),
                citations=data.get("citations", []),
                steps_taken=data.get("steps_taken", []),
                duration_ms=duration,
                metadata=data,
            )
        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000
            logger.warning(f"Agent call failed for '{query[:50]}...': {exc}")
            return RunResult(
                query=query,
                response="",
                duration_ms=duration,
                error=str(exc),
            )

    async def run_dataset(
        self,
        points: list[dict[str, Any]],
        prompt_override: str | None = None,
        concurrency: int | None = None,
        n_runners: int | None = None,
    ) -> list[RunResult]:
        """Run a list of data points through the agent.

        Args:
            points: List of dicts with 'query', optionally 'expected_answer',
                'context', and ``metadata.invoke`` for agent-specific fields.
            prompt_override: Optional system prompt to inject.
            concurrency: Max concurrent requests (defaults to instance value).
            n_runners: Alias for *concurrency* (Agent Lightning naming).

        Returns:
            List of RunResult objects.
        """
        import asyncio

        limit = concurrency
        if n_runners is not None:
            limit = n_runners
        if limit is None:
            limit = self.concurrency
        semaphore = asyncio.Semaphore(max(1, int(limit)))

        async def _run_one(point: dict[str, Any]) -> RunResult:
            async with semaphore:
                meta = point.get("metadata") or {}
                invoke = meta.get("invoke") if isinstance(meta, dict) else None
                # A copy: an agent (or a payload builder) must never be able to
                # change the dataset it is evaluated on.
                invoke = _copy_inputs(invoke) if isinstance(invoke, dict) else {}
                context = list(point.get("context") or [])
                result = await self.run_single(
                    query=point["query"],
                    prompt_override=prompt_override,
                    # An example's own context already travels in its inputs,
                    # unchanged; sending its documents again would hand the
                    # agent a second, re-serialised copy.
                    context=None if "context" in invoke else (context or None),
                    invoke=invoke,
                )
                result.expected = point.get("expected_answer") or point.get("expected")
                if not result.context:
                    result.context = context
                if isinstance(meta, dict) and meta:
                    # Dataset metadata (rubric hints, required facts, split…)
                    # for example-aware metrics.
                    result.metadata = {**result.metadata, "example": dict(meta)}
                tags = point.get("tags")
                if tags:
                    result.metadata = {**result.metadata, "example_tags": list(tags)}
                return result

        return list(await asyncio.gather(*[_run_one(p) for p in points]))

    async def submit_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        *,
        correction: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Submit feedback for an agent response.

        Useful for feeding optimization results back.
        """
        import httpx

        try:
            async with httpx.AsyncClient(base_url=self.api_base, timeout=10.0) as client:
                resp = await client.post(
                    f"{self.api_prefix}/{self.agent}/feedback",
                    json={
                        "query": query,
                        "response": response,
                        "rating": rating,
                        "correction": correction,
                        "comment": comment,
                        "user_id": "optimizer",
                        "feedback_type": "correction" if correction else "thumbs",
                    },
                )
                return dict(resp.json())
        except Exception as exc:
            logger.warning(f"Failed to submit feedback: {exc}")
            return {"status": "error", "error": str(exc)}
