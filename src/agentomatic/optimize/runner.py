"""Agent runner — invokes agents with prompt overrides for evaluation.

Supports both remote (HTTP API) and local (direct import) invocation.
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
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AgentRunner:
    """Runs dataset points through an agent endpoint.

    Supports:
    - Remote: HTTP calls to the platform API
    - Local: Direct function invocation (for speed)
    """

    def __init__(
        self,
        agent: str,
        api_base: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        timeout: float = 60.0,
    ):
        self.agent = agent
        self.api_base = api_base
        self.api_prefix = api_prefix
        self.timeout = timeout

    async def run_single(
        self,
        query: str,
        prompt_override: str | None = None,
        context: list[str] | None = None,
    ) -> RunResult:
        """Run a single query through the agent."""
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

            async with httpx.AsyncClient(
                base_url=self.api_base, timeout=self.timeout
            ) as client:
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
                result.context = point.get("context", [])
                return result

        tasks = [_run_one(p) for p in points]
        results = await asyncio.gather(*tasks)
        return list(results)
