"""Tests for middleware path skip helpers and Studio/auth skip prefixes."""

from __future__ import annotations

from agentomatic.middleware.pathutils import path_is_skipped
from agentomatic.security.jwt_auth import _DEFAULT_SKIP_PATHS


class TestPathIsSkipped:
    def test_exact_match(self) -> None:
        assert path_is_skipped("/health", {"/health"})

    def test_prefix_match(self) -> None:
        skips = {"/studio", "/status"}
        assert path_is_skipped("/studio", skips)
        assert path_is_skipped("/studio/info", skips)
        assert path_is_skipped("/studio/ui/", skips)
        assert path_is_skipped("/status", skips)
        assert path_is_skipped("/status/platform", skips)
        assert not path_is_skipped("/api/v1/agent/invoke", skips)

    def test_jwt_defaults_include_studio(self) -> None:
        assert "/studio" in _DEFAULT_SKIP_PATHS
        assert path_is_skipped("/studio/agents", _DEFAULT_SKIP_PATHS)
