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

    def test_jwt_defaults_exempt_only_the_studio_ui_shell(self) -> None:
        """Regression: the bare "/studio" prefix must NOT be in the default
        skip set — via path_is_skipped's prefix matching, that would also
        exempt the entire Studio debug REST API (/studio/agents,
        /studio/.../threads/{id}/state, etc.) from JWT auth, letting an
        unauthenticated caller read/mutate any agent's run state.

        Only the static UI shell ("/studio/ui") is public, matching how
        "/docs" only exempts the Swagger UI shell, not the API it documents.
        """
        assert "/studio" not in _DEFAULT_SKIP_PATHS
        assert "/studio/ui" in _DEFAULT_SKIP_PATHS
        assert path_is_skipped("/studio/ui/", _DEFAULT_SKIP_PATHS)
        assert not path_is_skipped("/studio/agents", _DEFAULT_SKIP_PATHS)
        assert not path_is_skipped(
            "/studio/agents/hello/threads/victim/state", _DEFAULT_SKIP_PATHS
        )
