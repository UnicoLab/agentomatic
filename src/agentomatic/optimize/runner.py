"""Agent runner — invokes agents with prompt overrides for evaluation.

Supports both remote (HTTP API) and local (direct import) invocation.
Uses the dedicated ``/optimize/invoke`` endpoint when available, falling
back to ``/invoke`` for full pipeline context capture (retrieval docs,
tool calls, reasoning steps).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


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
    - Local: Direct function invocation (for speed)
    - Optimize: Dedicated ``/optimize/invoke`` endpoint with full context

    The runner automatically tries ``/optimize/invoke`` first,
    falling back to ``/invoke`` if the optimization endpoint
    is not available.
    """

    def __init__(
        self,
        agent: str,
        api_base: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        timeout: float = 60.0,
        use_optimize_endpoint: bool = True,
    ):
        self.agent = agent
        self.api_base = api_base
        self.api_prefix = api_prefix
        self.timeout = timeout
        self.use_optimize_endpoint = use_optimize_endpoint
        self._optimize_available: bool | None = None  # Auto-detect

    async def run_single(
        self,
        query: str,
        prompt_override: str | None = None,
        context: list[str] | None = None,
    ) -> RunResult:
        """Run a single query through the agent.

        Tries ``/optimize/invoke`` first for full context, falls back
        to ``/invoke`` if not available.
        """
        if self.use_optimize_endpoint and self._optimize_available is not False:
            result = await self._run_optimize(query, prompt_override, context)
            if result.error and "404" in result.error:
                # Optimize endpoint not available, fall back
                self._optimize_available = False
                logger.info(
                    f"Optimize endpoint not available for '{self.agent}', falling back to /invoke"
                )
                return await self._run_invoke(query, prompt_override, context)
            self._optimize_available = True
            return result
        return await self._run_invoke(query, prompt_override, context)

    async def _run_optimize(
        self,
        query: str,
        prompt_override: str | None = None,
        context: list[str] | None = None,
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
            if prompt_override:
                payload["system_prompt_override"] = prompt_override
            if context:
                payload["context"] = {"documents": context}

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
                response=data.get("response", str(data)),
                retrieval_context=data.get("retrieval_context", []),
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
    ) -> RunResult:
        """Run via standard /invoke endpoint."""
        import httpx

        t0 = time.perf_counter()
        try:
            payload: dict[str, Any] = {
                "query": query,
                "user_id": "optimizer",
            }
            if prompt_override:
                payload["context"] = {
                    "system_prompt_override": prompt_override,
                }
            if context:
                payload.setdefault("context", {})
                payload["context"]["documents"] = context

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
                response=data.get("response", str(data)),
                # Try to extract context from standard response
                retrieval_context=data.get("metadata", {}).get("retrieval_context", []),
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
        concurrency: int = 5,
    ) -> list[RunResult]:
        """Run a list of data points through the agent.

        Args:
            points: List of dicts with 'query', optionally 'expected_answer', 'context'.
            prompt_override: Optional system prompt to inject.
            concurrency: Max concurrent requests.

        Returns:
            List of RunResult objects.
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)
        results: list[RunResult] = []

        async def _run_one(point: dict[str, Any]) -> RunResult:
            async with semaphore:
                result = await self.run_single(
                    query=point["query"],
                    prompt_override=prompt_override,
                    context=point.get("context"),
                )
                result.expected = point.get("expected_answer")
                if not result.context:
                    result.context = point.get("context", [])
                return result

        tasks = [_run_one(p) for p in points]
        results = await asyncio.gather(*tasks)
        return list(results)

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
