"""OpenTelemetry auto-instrumentation for agentomatic.

Auto-configures tracing and metrics for all agent invocations.
Gracefully degrades to no-ops if opentelemetry is not installed.

Usage::

    # Auto-setup in platform.build():
    from agentomatic.observability.telemetry import setup_telemetry
    setup_telemetry(app)

    # Custom spans in agent code:
    from agentomatic.observability.telemetry import traced

    @traced("my_agent.retrieve")
    async def retrieve_docs(query: str) -> list[str]:
        ...

Environment variables:
    OTEL_SERVICE_NAME: Service name (default: 'agentomatic')
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    OTEL_EXPORTER_OTLP_HEADERS: Optional headers
    OTEL_TRACES_SAMPLER: Sampling strategy
"""

from __future__ import annotations

import asyncio
import functools
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Conditional OpenTelemetry imports
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.trace import StatusCode

    class SafeConsoleSpanExporter(ConsoleSpanExporter):
        """ConsoleSpanExporter that catches I/O errors when stdout is closed during exit."""

        def export(self, spans: Any) -> SpanExportResult:
            try:
                if self.out and getattr(self.out, "closed", False):
                    return SpanExportResult.SUCCESS
                return super().export(spans)
            except (ValueError, OSError):
                return SpanExportResult.SUCCESS

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

# Module-level tracer singleton
_tracer: Any | None = None


# ── No-op helpers (used when opentelemetry is not installed) ──────────────


class _NoOpSpan:
    """Minimal span stand-in so decorated code runs unmodified."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """Minimal tracer stand-in that yields ``_NoOpSpan``."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


# ── Public API ────────────────────────────────────────────────────────────


def setup_telemetry(
    app: Any = None,
    service_name: str | None = None,
    enable_console: bool = False,
) -> Any:
    """Auto-configure OpenTelemetry tracing.

    Parameters
    ----------
    app:
        An optional ``FastAPI`` application instance.  When provided the
        module will call ``FastAPIInstrumentor.instrument_app(app)``.
    service_name:
        Override the service name.  Falls back to the ``OTEL_SERVICE_NAME``
        environment variable, then to ``'agentomatic'``.
    enable_console:
        If ``True`` a :class:`ConsoleSpanExporter` is attached even when an
        OTLP endpoint is configured (useful for local debugging).

    Returns
    -------
    The configured :class:`~opentelemetry.trace.Tracer`, or ``None`` when
    OpenTelemetry is not installed.
    """
    global _tracer  # noqa: PLW0603

    if not HAS_OTEL:
        logger.debug("OpenTelemetry not installed — tracing disabled")
        return None

    svc_name: str = service_name or os.getenv("OTEL_SERVICE_NAME") or "agentomatic"
    resource = Resource.create(
        {
            "service.name": svc_name,
            "service.version": "1.0.0",
        }
    )
    provider = TracerProvider(resource=resource)

    # ── Configure exporter ────────────────────────────────────────────
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if otlp_endpoint:
        # Prefer the gRPC exporter; fall back to HTTP/protobuf
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter: SpanExporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"📡 OTEL traces → {otlp_endpoint}")
        except ImportError:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter as OTLPSpanExporterHTTP,
                )

                exporter = OTLPSpanExporterHTTP(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"📡 OTEL traces (HTTP) → {otlp_endpoint}")
            except ImportError:
                logger.warning("OTLP exporter packages not installed — falling back to console")
                enable_console = True

    if enable_console or not otlp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(SafeConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("agentomatic")

    # ── Auto-instrument FastAPI ───────────────────────────────────────
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            logger.info("🔭 FastAPI auto-instrumented with OpenTelemetry")
        except ImportError:
            logger.debug("opentelemetry-instrumentation-fastapi not installed — skipping")

    # ── Auto-instrument httpx ─────────────────────────────────────────
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.debug("🔭 httpx auto-instrumented")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-httpx not installed — skipping")

    logger.info(f"🔭 OpenTelemetry configured (service={svc_name})")
    return _tracer


def get_tracer(name: str = "agentomatic") -> Any:
    """Return the module-level tracer, lazily falling back to a no-op.

    Parameters
    ----------
    name:
        Tracer instrumentation scope name.  Ignored when a tracer has already
        been initialised via :func:`setup_telemetry`.
    """
    global _tracer  # noqa: PLW0603

    if _tracer is not None:
        return _tracer
    if HAS_OTEL:
        return trace.get_tracer(name)
    return _NoOpTracer()


def traced(name: str | None = None) -> Callable[[F], F]:
    """Decorator that wraps a function in an OpenTelemetry span.

    Works for **both** sync and async callables.  When OpenTelemetry is not
    installed the decorator is a transparent pass-through (zero overhead).

    Parameters
    ----------
    name:
        Explicit span name.  Defaults to ``<module>.<qualname>``.

    Example
    -------
    ::

        @traced("agent.plan")
        async def plan(goal: str) -> str:
            ...
    """

    def decorator(fn: F) -> F:
        if not HAS_OTEL:
            return fn  # No-op passthrough — no wrapping at all

        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        # ── async path ────────────────────────────────────────────────
        @functools.wraps(fn)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                _set_safe_attributes(span, fn, args, kwargs)
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    span.set_attribute("duration_ms", elapsed)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    span.set_attribute(
                        "duration_ms",
                        (time.perf_counter() - t0) * 1000,
                    )
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

        # ── sync path ─────────────────────────────────────────────────
        @functools.wraps(fn)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                _set_safe_attributes(span, fn, args, kwargs)
                t0 = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    span.set_attribute("duration_ms", elapsed)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    span.set_attribute(
                        "duration_ms",
                        (time.perf_counter() - t0) * 1000,
                    )
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise

        if asyncio.iscoroutinefunction(fn):
            return _async_wrapper  # type: ignore[return-value]
        return _sync_wrapper  # type: ignore[return-value]

    return decorator


# ── Internal helpers ──────────────────────────────────────────────────────


def _set_safe_attributes(
    span: Any,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Attach function metadata to *span* without risking serialisation errors.

    Only primitive-typed keyword arguments are recorded; complex objects are
    skipped to avoid blowing up the span payload.
    """
    span.set_attribute("code.function", fn.__qualname__)
    span.set_attribute("code.namespace", fn.__module__)

    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"arg.{key}", value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, int, float, bool)) for v in value
        ):
            span.set_attribute(f"arg.{key}", list(value))
