"""Deep Agentomatic / ai_platform Prometheus instrumentation.

Agentomatic already defines rich counters/histograms
(``agentomatic_agent_invocations_total``, ``agentomatic_llm_duration_seconds``,
…) but many are never recorded. This module:

1. Wraps ``invoke_registered_agent`` → per-agent duration / success / error
2. Wraps ``invoke_with_retry`` → LLM duration, upstream calls, token usage
3. Wraps pipelines / plugins / ingestion / endpoints / tasks
4. Adds ASGI middleware classifying sync / async / stream / task / …
5. Publishes registry + per-component inventory gauges on startup

Import and call :func:`install_telemetry` once after ``platform.build()``.
"""

from __future__ import annotations

import contextvars
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

try:
    from prometheus_client import Counter, Gauge, Histogram

    _HAS_PROM = True
except ImportError:  # pragma: no cover
    _HAS_PROM = False

# Distinguishes direct HTTP/API calls from nested pipeline-step invocations.
_TELEM_SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "scooper_telem_source", default="direct"
)

_INSTALLED = False

_BUCKETS_HTTP = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_BUCKETS_WORK = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)

# ── Custom Scooper metrics (always labeled for dashboards) ───────────────────

if _HAS_PROM:
    AI_MODE_REQUESTS = Counter(
        "scooper_ai_mode_requests_total",
        "AI platform HTTP requests by interaction mode",
        ["mode", "resource", "name", "method", "status"],
    )
    AI_MODE_DURATION = Histogram(
        "scooper_ai_mode_duration_seconds",
        "AI platform HTTP duration by interaction mode",
        ["mode", "resource", "name", "method"],
        buckets=_BUCKETS_HTTP,
    )
    AI_LLM_TOKENS = Counter(
        "scooper_ai_llm_tokens_total",
        "LLM token usage observed from provider responses",
        ["direction", "provider", "model"],
    )
    AI_LLM_CALLS = Counter(
        "scooper_ai_llm_calls_total",
        "LLM invoke_with_retry attempts",
        ["provider", "model", "status"],
    )
    AI_LLM_DURATION = Histogram(
        "scooper_ai_llm_duration_seconds",
        "LLM invoke_with_retry wall time",
        ["provider", "model"],
        buckets=_BUCKETS_WORK,
    )
    AI_LLM_TTFT = Histogram(
        "scooper_ai_llm_ttft_seconds",
        "Time to first token / first usable content",
        ["provider", "model"],
        buckets=_BUCKETS_WORK,
    )
    AI_LLM_THINKING = Histogram(
        "scooper_ai_llm_thinking_seconds",
        "Estimated or measured thinking/reasoning phase duration",
        ["provider", "model"],
        buckets=_BUCKETS_WORK,
    )
    AI_LLM_STRUCTURE_ERRORS = Counter(
        "scooper_ai_llm_structure_errors_total",
        "Structured output / JSON schema parse failures",
        ["provider", "model", "agent"],
    )
    AI_LLM_OUTCOMES = Counter(
        "scooper_ai_llm_outcomes_total",
        "LLM outcomes for model-fit (mirrors agentomatic when available)",
        ["provider", "model", "outcome"],
    )
    AI_REGISTRY = Gauge(
        "scooper_ai_registry",
        "Registered platform inventory counts",
        ["kind"],
    )
    AI_COMPONENT = Gauge(
        "scooper_ai_component_info",
        "Registered component presence (1 = registered)",
        ["kind", "name"],
    )
    AI_PIPELINE_RUNS = Counter(
        "scooper_ai_pipeline_runs_total",
        "Pipeline engine executions",
        ["pipeline", "status"],
    )
    AI_PIPELINE_DURATION = Histogram(
        "scooper_ai_pipeline_duration_seconds",
        "Pipeline engine wall time",
        ["pipeline"],
        buckets=_BUCKETS_WORK,
    )
    AI_PLUGIN_CALLS = Counter(
        "scooper_ai_plugin_calls_total",
        "Plugin predict invocations",
        ["plugin", "status", "source"],
    )
    AI_PLUGIN_DURATION = Histogram(
        "scooper_ai_plugin_duration_seconds",
        "Plugin predict wall time",
        ["plugin", "source"],
        buckets=_BUCKETS_WORK,
    )
    AI_INGESTION_RUNS = Counter(
        "scooper_ai_ingestion_runs_total",
        "Ingestor run invocations",
        ["ingestor", "status"],
    )
    AI_INGESTION_DURATION = Histogram(
        "scooper_ai_ingestion_duration_seconds",
        "Ingestor run wall time",
        ["ingestor"],
        buckets=_BUCKETS_WORK,
    )
    AI_ENDPOINT_CALLS = Counter(
        "scooper_ai_endpoint_calls_total",
        "Custom endpoint invocations (HTTP + pipeline steps)",
        ["endpoint", "status", "source"],
    )
    AI_ENDPOINT_DURATION = Histogram(
        "scooper_ai_endpoint_duration_seconds",
        "Custom endpoint wall time",
        ["endpoint", "source"],
        buckets=_BUCKETS_WORK,
    )
    AI_TASK_EVENTS = Counter(
        "scooper_ai_task_events_total",
        "Task lifecycle events",
        ["target_type", "target", "event"],
    )
    AI_TASKS_ACTIVE = Gauge(
        "scooper_ai_tasks_active",
        "In-flight async tasks (running queue)",
    )
else:  # pragma: no cover

    class _N:
        def labels(self, **_k: Any) -> _N:
            return self

        def inc(self, *_a: Any, **_k: Any) -> None:
            return None

        def observe(self, *_a: Any, **_k: Any) -> None:
            return None

        def set(self, *_a: Any, **_k: Any) -> None:
            return None

        def dec(self, *_a: Any, **_k: Any) -> None:
            return None

    AI_MODE_REQUESTS = AI_MODE_DURATION = AI_LLM_TOKENS = AI_LLM_CALLS = AI_LLM_DURATION = (
        AI_LLM_TTFT
    ) = AI_LLM_THINKING = AI_LLM_STRUCTURE_ERRORS = AI_LLM_OUTCOMES = AI_REGISTRY = AI_COMPONENT = (
        AI_PIPELINE_RUNS
    ) = AI_PIPELINE_DURATION = AI_PLUGIN_CALLS = AI_PLUGIN_DURATION = AI_INGESTION_RUNS = (
        AI_INGESTION_DURATION
    ) = AI_ENDPOINT_CALLS = AI_ENDPOINT_DURATION = AI_TASK_EVENTS = AI_TASKS_ACTIVE = _N()


_UUIDISH = re.compile(r"^[0-9a-fA-F-]{8,}$")
_HEXISH = re.compile(r"^[0-9a-fA-F]{12,}$")
_TASKISH = re.compile(r"^task_[0-9a-fA-F]+$", re.I)


def classify_ai_path(path: str) -> tuple[str, str, str]:
    """Return ``(mode, resource, name)`` for an AI platform URL path.

    Modes: ``sync``, ``async``, ``stream``, ``batch``, ``task``, ``studio``,
    ``other``.
    Resources: ``agent``, ``pipeline``, ``plugin``, ``ingestion``, ``endpoint``,
    ``task``, ``platform``, ``http``, ``ui``.
    """
    parts = [p for p in path.split("/") if p]
    norm_parts: list[str] = []
    for p in parts:
        if _UUIDISH.match(p) or _HEXISH.match(p) or _TASKISH.match(p) or p.isdigit():
            norm_parts.append("{id}")
        else:
            norm_parts.append(p)

    if len(norm_parts) >= 3 and norm_parts[0] == "api" and norm_parts[1] == "v1":
        head = norm_parts[2]

        if head == "tasks":
            if len(norm_parts) == 3:
                return "async", "task", "submit"
            action = norm_parts[-1] if norm_parts[-1] != "{id}" else "status"
            if action == "events":
                return "stream", "task", "events"
            if action in {"result", "cancel", "status"}:
                return "task", "task", action
            return "task", "task", action

        if head == "pipelines":
            if len(norm_parts) == 3:
                return "other", "pipeline", "list"
            name = norm_parts[3]
            if "batch" in norm_parts:
                return "batch", "pipeline", name
            if "async" in norm_parts:
                return "async", "pipeline", name
            if "stream" in norm_parts:
                return "stream", "pipeline", name
            if norm_parts[-1] == "run":
                return "sync", "pipeline", name
            return "other", "pipeline", name

        if head == "plugins":
            if len(norm_parts) == 3:
                return "other", "plugin", "list"
            if norm_parts[-1] == "reload" and len(norm_parts) == 4:
                return "other", "plugin", "reload_all"
            name = norm_parts[3]
            if "batch" in norm_parts:
                return "batch", "plugin", name
            if "async" in norm_parts:
                return "async", "plugin", name
            if "predict" in norm_parts:
                return "sync", "plugin", name
            return "other", "plugin", name

        if head == "ingestion":
            if len(norm_parts) == 3:
                return "other", "ingestion", "list"
            name = norm_parts[3]
            if "batch" in norm_parts:
                return "batch", "ingestion", name
            if "async" in norm_parts:
                return "async", "ingestion", name
            if "run" in norm_parts:
                return "sync", "ingestion", name
            return "other", "ingestion", name

        if head == "endpoints":
            if len(norm_parts) == 3:
                return "other", "endpoint", "list"
            name = norm_parts[3]
            if "async" in norm_parts:
                return "async", "endpoint", name
            if "batch" in norm_parts:
                return "batch", "endpoint", name
            return "sync", "endpoint", name

        # agents: /api/v1/{agent}/invoke[/async|/stream|/batch]
        if len(norm_parts) >= 4 and norm_parts[3] == "invoke":
            agent = norm_parts[2]
            if "batch" in norm_parts:
                return "batch", "agent", agent
            if "async" in norm_parts:
                return "async", "agent", agent
            if "stream" in norm_parts:
                return "stream", "agent", agent
            return "sync", "agent", agent

        if head in {"studio", "a2a"}:
            return "studio", head, norm_parts[3] if len(norm_parts) > 3 else head
        if head in {"status", "health", "ready"}:
            return "other", "platform", head

    if path.startswith("/studio") or path.startswith("/docs"):
        return "studio", "ui", path.strip("/").split("/")[0] or "ui"
    return "other", "http", path


def _llm_identity(llm: Any) -> tuple[str, str]:
    model = (
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or getattr(llm, "deployment_name", None)
        or "unknown"
    )
    provider = (
        getattr(llm, "_llm_type", None) or getattr(llm, "provider", None) or type(llm).__name__
    )
    return str(provider or "unknown"), str(model or "unknown")


def _extract_usage(result: Any) -> tuple[int, int, int]:
    """Return ``(prompt, completion, total)`` tokens when present."""
    prompt = completion = total = 0
    usage_meta = getattr(result, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        prompt = int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
        completion = int(
            usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0
        )
        total = int(usage_meta.get("total_tokens") or (prompt + completion))
    meta = getattr(result, "response_metadata", None) or {}
    if isinstance(meta, dict):
        usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(usage, dict):
            prompt = prompt or int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = completion or int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            total = total or int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


class AiModeMetricsMiddleware(BaseHTTPMiddleware):
    """Classify every AI HTTP call into sync/async/stream/task/pipeline…"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path in {"/health", "/ready", "/metrics", "/openapi.json"}:
            return await call_next(request)
        mode, resource, name = classify_ai_path(path)
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        status = str(getattr(response, "status_code", 0))
        AI_MODE_REQUESTS.labels(
            mode=mode,
            resource=resource,
            name=name,
            method=request.method,
            status=status,
        ).inc()
        AI_MODE_DURATION.labels(
            mode=mode, resource=resource, name=name, method=request.method
        ).observe(dt)
        return response


def _patch_callable(module_name: str, attr: str, wrapper: Any) -> bool:
    mod = sys.modules.get(module_name)
    if mod is None or not hasattr(mod, attr):
        return False
    setattr(mod, attr, wrapper)
    return True


def _rebind_symbol(original: Any, wrapper: Any, attr: str) -> None:
    for _mod_name, mod in list(sys.modules.items()):
        if getattr(mod, attr, None) is original:
            setattr(mod, attr, wrapper)


def _install_agent_invoke_wrapper() -> None:
    from agentomatic.core import agent_invoke as agent_invoke_mod
    from agentomatic.observability.metrics import track_agent_invocation

    original = agent_invoke_mod.invoke_registered_agent

    async def wrapped(agent: Any, state: dict[str, Any]) -> Any:
        name = (
            getattr(agent, "name", None)
            or getattr(getattr(agent, "class_instance", None), "agent_name", None)
            or "unknown"
        )
        async with track_agent_invocation(str(name)):
            return await original(agent, state)

    agent_invoke_mod.invoke_registered_agent = wrapped
    for mod_name in (
        "agentomatic.core.agent_invoke",
        "agentomatic.core.router_factory",
        "agentomatic.pipelines.steps",
        "agentomatic.pipelines.flow",
        "agentomatic.tasks.dispatchers",
    ):
        _patch_callable(mod_name, "invoke_registered_agent", wrapped)
    _rebind_symbol(original, wrapped, "invoke_registered_agent")


def _install_llm_wrapper() -> None:
    from agentomatic.providers import llm as llm_mod

    original = llm_mod.invoke_with_retry

    async def wrapped_llm(
        llm: Any,
        messages: list,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        provider, model = _llm_identity(llm)
        t0 = time.perf_counter()
        status = "ok"
        outcome = "success"
        try:
            result = await original(
                llm, messages, max_retries=max_retries, retry_delay=retry_delay, **kwargs
            )
            prompt, completion, total = _extract_usage(result)
            if prompt:
                AI_LLM_TOKENS.labels(direction="prompt", provider=provider, model=model).inc(
                    prompt
                )
            if completion:
                AI_LLM_TOKENS.labels(direction="completion", provider=provider, model=model).inc(
                    completion
                )
            if total and not (prompt or completion):
                AI_LLM_TOKENS.labels(direction="total", provider=provider, model=model).inc(total)
            # Thinking tokens / duration when agentomatic attached metadata.
            try:
                from agentomatic.providers import message_thinking

                thinking = message_thinking(result) or ""
            except Exception:  # noqa: BLE001
                thinking = ""
            if thinking:
                AI_LLM_TOKENS.labels(direction="thinking", provider=provider, model=model).inc(
                    max(1, len(thinking.split()))
                )
            return result
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__} {exc}".lower()
            if "timeout" in err:
                outcome = "timeout"
            elif "http" in err or "status" in err:
                outcome = "http_error"
            else:
                outcome = "error"
            raise
        finally:
            dt = time.perf_counter() - t0
            AI_LLM_CALLS.labels(provider=provider, model=model, status=status).inc()
            AI_LLM_DURATION.labels(provider=provider, model=model).observe(dt)
            AI_LLM_OUTCOMES.labels(provider=provider, model=model, outcome=outcome).inc()
            # agentomatic invoke_with_retry already records agentomatic_* metrics.

    llm_mod.invoke_with_retry = wrapped_llm
    _rebind_symbol(original, wrapped_llm, "invoke_with_retry")
    _patch_callable("agentomatic.providers", "invoke_with_retry", wrapped_llm)
    _patch_callable("agentomatic.providers.llm", "invoke_with_retry", wrapped_llm)


def record_assistant_structure_error(*, llm: Any = None, model: str = "unknown") -> None:
    """Bridge for agents when JSON/schema parse fails after a successful LLM call."""
    provider, resolved_model = ("unknown", model)
    if llm is not None:
        provider, resolved_model = _llm_identity(llm)
    AI_LLM_STRUCTURE_ERRORS.labels(
        provider=provider, model=resolved_model, agent="assistant"
    ).inc()
    AI_LLM_OUTCOMES.labels(
        provider=provider, model=resolved_model, outcome="format_error"
    ).inc()
    try:
        from agentomatic.observability.metrics import record_structure_error

        record_structure_error(llm=llm, provider=provider, model=resolved_model, agent="assistant")
    except Exception:  # noqa: BLE001
        pass


def _install_pipeline_wrapper() -> None:
    from agentomatic.pipelines.engine import PipelineEngine

    original = PipelineEngine.run

    async def wrapped(
        self: Any,
        input_data: dict[str, Any] | None = None,
        *,
        progress_cb: Any = None,
        checkpoint_cb: Any = None,
        completed_indices: Any = None,
        **kwargs: Any,
    ) -> Any:
        name = getattr(getattr(self, "config", None), "name", None) or "unknown"
        t0 = time.perf_counter()
        status = "success"
        try:
            result = await original(
                self,
                input_data,
                progress_cb=progress_cb,
                checkpoint_cb=checkpoint_cb,
                completed_indices=completed_indices,
                **kwargs,
            )
            result_status = getattr(result, "status", None)
            if result_status is not None:
                status = getattr(result_status, "value", str(result_status)).lower()
                if status in {"failed", "error", "cancelled"}:
                    status = "error" if status != "cancelled" else "cancelled"
                elif status in {"success", "succeeded", "completed"}:
                    status = "success"
            return result
        except Exception:
            status = "error"
            raise
        finally:
            AI_PIPELINE_RUNS.labels(pipeline=str(name), status=status).inc()
            AI_PIPELINE_DURATION.labels(pipeline=str(name)).observe(time.perf_counter() - t0)

    PipelineEngine.run = wrapped  # type: ignore[method-assign]


def _install_pipeline_step_wrappers() -> None:
    """Mark nested step calls as ``source=pipeline`` (metrics on predict/call/run)."""
    from agentomatic.pipelines import steps as steps_mod

    original_plugin = steps_mod.execute_plugin_step
    original_endpoint = steps_mod.execute_endpoint_step
    original_ingestion = steps_mod.execute_ingestion_step

    async def wrapped_plugin(config: Any, ctx: Any, plugins: Any) -> Any:
        token = _TELEM_SOURCE.set("pipeline")
        try:
            return await original_plugin(config, ctx, plugins)
        finally:
            _TELEM_SOURCE.reset(token)

    async def wrapped_endpoint(config: Any, ctx: Any, endpoints: Any) -> Any:
        token = _TELEM_SOURCE.set("pipeline")
        try:
            return await original_endpoint(config, ctx, endpoints)
        finally:
            _TELEM_SOURCE.reset(token)

    async def wrapped_ingestion(config: Any, ctx: Any, ingestors: Any) -> Any:
        token = _TELEM_SOURCE.set("pipeline")
        try:
            return await original_ingestion(config, ctx, ingestors)
        finally:
            _TELEM_SOURCE.reset(token)

    steps_mod.execute_plugin_step = wrapped_plugin
    steps_mod.execute_endpoint_step = wrapped_endpoint
    steps_mod.execute_ingestion_step = wrapped_ingestion
    _rebind_symbol(original_plugin, wrapped_plugin, "execute_plugin_step")
    _rebind_symbol(original_endpoint, wrapped_endpoint, "execute_endpoint_step")
    _rebind_symbol(original_ingestion, wrapped_ingestion, "execute_ingestion_step")


def _install_ingestion_wrapper() -> None:
    from agentomatic.ingestion.base import BaseIngestor

    original = BaseIngestor.run

    async def wrapped(self: Any, request: Any = None, ctx: Any = None) -> Any:
        name = getattr(self, "ingestor_name", None) or type(self).__name__
        t0 = time.perf_counter()
        status = "success"
        try:
            result = await original(self, request, ctx)
            result_status = getattr(result, "status", None)
            if result_status is not None:
                raw = getattr(result_status, "value", str(result_status)).lower()
                if raw in {"failed", "error"}:
                    status = "error"
            return result
        except Exception:
            status = "error"
            raise
        finally:
            AI_INGESTION_RUNS.labels(ingestor=str(name), status=status).inc()
            AI_INGESTION_DURATION.labels(ingestor=str(name)).observe(time.perf_counter() - t0)

    BaseIngestor.run = wrapped  # type: ignore[method-assign]


def _wrap_plugin_predict(plugin: Any) -> None:
    if getattr(plugin, "_scooper_telem_wrapped", False):
        return
    original = plugin.predict
    name = getattr(plugin, "plugin_name", None) or type(plugin).__name__

    async def wrapped(inputs: Any, *args: Any, **kwargs: Any) -> Any:
        source = _TELEM_SOURCE.get()
        t0 = time.perf_counter()
        status = "success"
        try:
            return await original(inputs, *args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            AI_PLUGIN_CALLS.labels(plugin=str(name), status=status, source=source).inc()
            AI_PLUGIN_DURATION.labels(plugin=str(name), source=source).observe(
                time.perf_counter() - t0
            )

    plugin.predict = wrapped
    plugin._scooper_telem_wrapped = True


def _wrap_endpoint_call(endpoint: Any) -> None:
    if getattr(endpoint, "_scooper_telem_wrapped", False):
        return
    original = endpoint.call
    name = (
        getattr(endpoint, "endpoint_name", None)
        or getattr(endpoint, "name", None)
        or type(endpoint).__name__
    )

    async def wrapped(payload: Any = None, **kwargs: Any) -> Any:
        from agentomatic.observability import metrics as ametrics

        source = _TELEM_SOURCE.get()
        t0 = time.perf_counter()
        status = "success"
        try:
            return await original(payload, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            dt = time.perf_counter() - t0
            AI_ENDPOINT_CALLS.labels(endpoint=str(name), status=status, source=source).inc()
            AI_ENDPOINT_DURATION.labels(endpoint=str(name), source=source).observe(dt)
            try:
                ametrics.ENDPOINT_CALL_COUNT.labels(endpoint=str(name), status=status).inc()
                ametrics.ENDPOINT_DURATION.labels(endpoint=str(name)).observe(dt)
            except Exception:  # noqa: BLE001
                pass

    endpoint.call = wrapped
    endpoint._scooper_telem_wrapped = True


def _install_task_wrapper(platform: Any) -> None:
    manager = getattr(platform, "task_manager", None) or getattr(platform, "_task_manager", None)
    if manager is None:
        return
    if getattr(manager, "_scooper_telem_wrapped", False):
        return

    original_submit = manager.submit
    original_finalize = manager._finalize

    async def wrapped_submit(*args: Any, **kwargs: Any) -> Any:
        record = await original_submit(*args, **kwargs)
        target_type = getattr(record, "target_type", None)
        tt = getattr(target_type, "value", str(target_type or "unknown"))
        target = str(getattr(record, "target", "unknown"))
        AI_TASK_EVENTS.labels(target_type=tt, target=target, event="queued").inc()
        try:
            AI_TASKS_ACTIVE.set(len(getattr(manager, "_running", {}) or {}))
        except Exception:  # noqa: BLE001
            pass
        return record

    async def wrapped_finalize(record: Any, status: Any, **kwargs: Any) -> Any:
        result = await original_finalize(record, status, **kwargs)
        target_type = getattr(record, "target_type", None)
        tt = getattr(target_type, "value", str(target_type or "unknown"))
        target = str(getattr(record, "target", "unknown"))
        event = getattr(status, "value", str(status)).lower()
        AI_TASK_EVENTS.labels(target_type=tt, target=target, event=event).inc()
        try:
            AI_TASKS_ACTIVE.set(len(getattr(manager, "_running", {}) or {}))
        except Exception:  # noqa: BLE001
            pass
        return result

    manager.submit = wrapped_submit
    manager._finalize = wrapped_finalize
    manager._scooper_telem_wrapped = True


def _registry_count(obj: Any) -> int:
    """Return ``.count`` / ``len`` / ``list_names`` size for a registry-like object."""
    if obj is None:
        return 0
    count = getattr(obj, "count", None)
    if isinstance(count, int):
        return count
    if callable(count):
        try:
            return int(count())
        except Exception:  # noqa: BLE001
            pass
    names = getattr(obj, "list_names", None)
    if callable(names):
        try:
            return len(list(names()))
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, dict):
        return len(obj)
    if hasattr(obj, "__len__"):
        try:
            return len(obj)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return 0
    return 0


def _registry_names(obj: Any) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [str(k) for k in obj]
    names = getattr(obj, "list_names", None)
    if callable(names):
        try:
            return [str(n) for n in names()]
        except Exception:  # noqa: BLE001
            pass
    all_fn = getattr(obj, "all", None)
    if callable(all_fn):
        try:
            data = all_fn()
            if isinstance(data, dict):
                return [str(k) for k in data]
        except Exception:  # noqa: BLE001
            pass
    list_plugins = getattr(obj, "list_plugins", None)
    if callable(list_plugins):
        try:
            data = list_plugins()
            if isinstance(data, dict):
                return [str(k) for k in data]
        except Exception:  # noqa: BLE001
            pass
    return []


def publish_registry(platform: Any) -> None:
    """Set inventory gauges from the live AgentPlatform instance."""
    try:
        from agentomatic.observability import metrics as ametrics

        registry = getattr(platform, "registry", None)
        pipelines = getattr(platform, "pipelines", None) or getattr(platform, "_pipelines", None)
        plugins = getattr(platform, "_plugin_registry", None) or getattr(
            platform, "plugin_registry", None
        )
        endpoints = getattr(platform, "endpoint_registry", None) or getattr(
            platform, "_endpoint_registry", None
        )
        ingestors = getattr(platform, "ingestion_registry", None) or getattr(
            platform, "_ingestion_registry", None
        )

        agent_names = _registry_names(registry)
        pipe_names = _registry_names(pipelines)
        plug_names = _registry_names(plugins)
        end_names = _registry_names(endpoints)
        ing_names = _registry_names(ingestors)

        agent_count = len(agent_names) or _registry_count(registry)
        pipe_n = len(pipe_names) or _registry_count(pipelines)
        plug_n = len(plug_names) or _registry_count(plugins)
        end_n = len(end_names) or _registry_count(endpoints)
        ing_n = len(ing_names) or _registry_count(ingestors)

        AI_REGISTRY.labels(kind="agents").set(agent_count)
        AI_REGISTRY.labels(kind="pipelines").set(pipe_n)
        AI_REGISTRY.labels(kind="plugins").set(plug_n)
        AI_REGISTRY.labels(kind="endpoints").set(end_n)
        AI_REGISTRY.labels(kind="ingestors").set(ing_n)

        for kind, names in (
            ("agent", agent_names),
            ("pipeline", pipe_names),
            ("plugin", plug_names),
            ("endpoint", end_names),
            ("ingestor", ing_names),
        ):
            for name in names:
                AI_COMPONENT.labels(kind=kind, name=name).set(1)

        # Wrap live plugin/endpoint instances (HTTP + direct predict/call).
        if plugins is not None:
            list_plugins = getattr(plugins, "list_plugins", None)
            if callable(list_plugins):
                for _name, plugin in (list_plugins() or {}).items():
                    _wrap_plugin_predict(plugin)
        if endpoints is not None:
            all_fn = getattr(endpoints, "all", None)
            items = all_fn() if callable(all_fn) else None
            if isinstance(items, dict):
                for _name, endpoint in items.items():
                    _wrap_endpoint_call(endpoint)

        _install_task_wrapper(platform)

        try:
            ametrics.REGISTERED_AGENTS.set(agent_count)
            ametrics.REGISTERED_ENDPOINTS.set(end_n)
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "Telemetry registry: agents={} pipelines={} plugins={} endpoints={} ingestors={}",
            agent_count,
            pipe_n,
            plug_n,
            end_n,
            ing_n,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telemetry registry publish failed: {}", exc)


def install_telemetry(app: Any, platform: Any | None = None) -> None:
    """Patch Agentomatic call sites and attach mode middleware (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not _HAS_PROM:
        logger.warning("prometheus_client missing — AI telemetry disabled")
        return
    try:
        _install_agent_invoke_wrapper()
        _install_llm_wrapper()
        _install_pipeline_wrapper()
        _install_pipeline_step_wrappers()
        _install_ingestion_wrapper()
        app.add_middleware(AiModeMetricsMiddleware)
        if platform is not None:
            publish_registry(platform)
            on_startup = getattr(platform, "on_startup", None)
            if callable(on_startup):

                @on_startup
                async def _refresh_registry() -> None:
                    publish_registry(platform)

        _INSTALLED = True
        logger.info(
            "AI telemetry installed (agents/LLM/pipelines/plugins/"
            "ingestion/endpoints/tasks + mode middleware)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI telemetry install failed: {}", exc)
