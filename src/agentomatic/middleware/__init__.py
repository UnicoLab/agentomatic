"""Pluggable middleware stack for agentomatic."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "AuthMiddleware",
    "MetricsMiddleware",
    "FeedbackCollector",
    "collect_feedback",
]

# Lazy imports to avoid cascading failures when optional dependencies
# (e.g. prometheus_client for MetricsMiddleware) are not installed.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AuthMiddleware": (".auth", "AuthMiddleware"),
    "LoggingMiddleware": (".logging", "LoggingMiddleware"),
    "MetricsMiddleware": (".metrics", "MetricsMiddleware"),
    "RateLimitMiddleware": (".rate_limit", "RateLimitMiddleware"),
    "FeedbackCollector": (".feedback", "FeedbackCollector"),
    "collect_feedback": (".feedback", "collect_feedback"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
