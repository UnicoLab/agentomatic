"""HTTP clients for calling deployed model services from custom endpoints.

``UpstreamClient`` wraps a single authenticated service (with retries and
metrics), while ``MultiModelClient`` fans out to several upstreams
concurrently and aggregates their responses.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from agentomatic.endpoints.auth import UpstreamAuthenticator, resolve_env
from agentomatic.endpoints.models import (
    AggregationStrategy,
    UpstreamConfig,
    UpstreamResult,
)

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover - httpx is a core dependency
    httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False


class UpstreamClient:
    """Authenticated HTTP client bound to a single upstream service.

    The client owns its own ``httpx.AsyncClient`` (created lazily) and
    applies static headers, resolved authentication, retries on transport
    errors, and best-effort Prometheus metrics.
    """

    def __init__(self, config: UpstreamConfig) -> None:
        if not _HAS_HTTPX:  # pragma: no cover
            raise ImportError("httpx is required for custom endpoints")
        self.config = config
        self._auth = UpstreamAuthenticator(config.auth)
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        """Return the upstream's logical name."""
        return self.config.name

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the underlying ``httpx.AsyncClient``."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=resolve_env(self.config.base_url),
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _build_headers(self, client: httpx.AsyncClient) -> dict[str, str]:
        """Merge static headers with resolved authentication headers."""
        headers = {k: resolve_env(v) for k, v in self.config.headers.items()}
        headers.update(await self._auth.headers(client))
        return headers

    async def request(
        self,
        payload: Any = None,
        *,
        path: str | None = None,
        method: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> UpstreamResult:
        """Call the upstream and return a structured :class:`UpstreamResult`.

        Args:
            payload: JSON-serialisable request body (ignored for GET).
            path: Override the configured default path.
            method: Override the configured default HTTP method.
            params: Optional query parameters.

        Returns:
            An :class:`UpstreamResult` capturing success/failure, the parsed
            body (JSON when possible, else text), and timing information.
        """
        t0 = time.perf_counter()
        client = await self._ensure_client()
        use_method = (method or self.config.method).upper()
        use_path = path if path is not None else self.config.path

        last_error: str | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                headers = await self._build_headers(client)
                kwargs: dict[str, Any] = {"headers": headers, "params": params}
                if use_method not in ("GET", "HEAD", "DELETE") and payload is not None:
                    kwargs["json"] = payload
                resp = await client.request(use_method, use_path, **kwargs)
                data: Any
                try:
                    data = resp.json()
                except Exception:  # noqa: BLE001
                    data = resp.text
                ok = resp.is_success
                if not ok:
                    last_error = f"HTTP {resp.status_code}"
                self._observe(ok, time.perf_counter() - t0)
                return UpstreamResult(
                    upstream=self.name,
                    ok=ok,
                    status_code=resp.status_code,
                    data=data,
                    error=None if ok else last_error,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self.config.max_retries:
                    delay = 0.5 * (2**attempt)
                    logger.warning(
                        f"Upstream '{self.name}' call failed "
                        f"(attempt {attempt + 1}), retrying in {delay:.1f}s: {exc}"
                    )
                    await asyncio.sleep(delay)

        self._observe(False, time.perf_counter() - t0)
        return UpstreamResult(
            upstream=self.name,
            ok=False,
            error=last_error or "unknown error",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _observe(ok: bool, elapsed: float) -> None:
        """Emit best-effort Prometheus metrics for an upstream call."""
        try:
            from agentomatic.observability.metrics import (
                UPSTREAM_CALL_COUNT,
                UPSTREAM_DURATION,
            )

            UPSTREAM_CALL_COUNT.labels(status="ok" if ok else "error").inc()
            UPSTREAM_DURATION.observe(elapsed)
        except Exception:  # noqa: BLE001 - metrics are optional
            pass


class MultiModelClient:
    """Fan out a request to multiple upstreams and aggregate results.

    Example::

        client = MultiModelClient([cfg_a, cfg_b])
        response = await client.fan_out({"prompt": "hi"})
        await client.aclose()
    """

    def __init__(
        self,
        upstreams: list[UpstreamConfig],
        *,
        strategy: AggregationStrategy = AggregationStrategy.ALL,
        max_concurrency: int = 5,
    ) -> None:
        self._clients: dict[str, UpstreamClient] = {u.name: UpstreamClient(u) for u in upstreams}
        self._weights: dict[str, float] = {u.name: u.weight for u in upstreams}
        self.strategy = strategy
        self._sem = asyncio.Semaphore(max_concurrency)

    @property
    def clients(self) -> dict[str, UpstreamClient]:
        """Return the per-upstream clients keyed by name."""
        return self._clients

    def get(self, name: str) -> UpstreamClient | None:
        """Return a single upstream client by name."""
        return self._clients.get(name)

    async def aclose(self) -> None:
        """Close all underlying HTTP clients."""
        for client in self._clients.values():
            await client.aclose()

    async def fan_out(
        self,
        payload: Any = None,
        *,
        upstreams: list[str] | None = None,
        strategy: AggregationStrategy | None = None,
        **request_kwargs: Any,
    ) -> tuple[bool, Any, list[UpstreamResult]]:
        """Call selected upstreams concurrently and aggregate the results.

        Args:
            payload: JSON payload forwarded to each upstream.
            upstreams: Optional subset of upstream names (defaults to all).
            strategy: Override the default aggregation strategy.
            **request_kwargs: Extra keyword arguments forwarded to
                :meth:`UpstreamClient.request` (e.g. ``path``, ``method``).

        Returns:
            A tuple of ``(ok, aggregated, results)`` where ``aggregated`` is
            the combined output according to the chosen strategy.
        """
        selected = upstreams or list(self._clients.keys())
        active = [self._clients[n] for n in selected if n in self._clients]
        use_strategy = strategy or self.strategy

        async def _run(client: UpstreamClient) -> UpstreamResult:
            async with self._sem:
                return await client.request(payload, **request_kwargs)

        results = list(await asyncio.gather(*(_run(c) for c in active)))
        ok, aggregated = self._aggregate(results, use_strategy)
        return ok, aggregated, results

    def _aggregate(
        self,
        results: list[UpstreamResult],
        strategy: AggregationStrategy,
    ) -> tuple[bool, Any]:
        """Combine individual upstream results per the aggregation strategy."""
        successes = [r for r in results if r.ok]

        if strategy == AggregationStrategy.FIRST_SUCCESS:
            if successes:
                return True, successes[0].data
            return False, None

        if strategy == AggregationStrategy.MAJORITY:
            if not successes:
                return False, None
            # Weighted vote: sum each upstream's weight per identical result.
            scores: dict[str, float] = {}
            repr_map: dict[str, Any] = {}
            for r in successes:
                key = repr(r.data)
                scores[key] = scores.get(key, 0.0) + self._weights.get(r.upstream, 1.0)
                repr_map.setdefault(key, r.data)
            winner = max(scores, key=lambda k: scores[k])
            return True, repr_map[winner]

        # ALL — return a mapping of upstream name → data
        aggregated = {r.upstream: r.data for r in successes}
        return (len(successes) == len(results) and bool(results)), aggregated
