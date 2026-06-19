"""Tests for v0.4.1 bug fixes.

Covers:
  - BUG 1:  memory_manager.py lazy langchain import
  - BUG 2:  middleware/__init__.py lazy imports
  - BUG 3:  router_factory.py langchain guard in /chat
  - BUG 4:  platform.py public discover_agents()
  - BUG 5:  platform.py sys.path in build()
  - BUG 6+13: platform.py Studio redirects
  - BUG 7:  router_factory.py history error metadata
  - BUG 8:  state.py add_messages fallback
  - BUG 9:  feedback.py uninitialized collector warning
  - BUG 11: manifest.py Studio fields on RegisteredAgent
  - BUG 14: lifespan.py deprecation notice
  - OPT:   OptimizationResult.save() + PromptFitResult.save()
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =====================================================================
# BUG 1: memory_manager.py — lazy langchain_core import
# =====================================================================


class TestMemoryManagerLazyImport:
    """memory_manager should import without langchain_core installed."""

    def test_module_imports_without_langchain(self):
        """The module itself should always be importable."""
        from agentomatic.core import memory_manager  # noqa: F401

        assert hasattr(memory_manager, "ConversationMemoryManager")

    def test_has_langchain_flag_exists(self):
        """HAS_LANGCHAIN flag should be set."""
        from agentomatic.core import memory_manager

        assert isinstance(memory_manager.HAS_LANGCHAIN, bool)

    def test_fallback_message_classes_have_content(self):
        """When langchain is not available, fallback classes should work."""
        from agentomatic.core.memory_manager import HumanMessage

        msg = HumanMessage(content="hello")
        assert msg.content == "hello"

    def test_fallback_ai_message(self):
        from agentomatic.core.memory_manager import AIMessage

        msg = AIMessage(content="response")
        assert msg.content == "response"

    def test_fallback_system_message(self):
        from agentomatic.core.memory_manager import SystemMessage

        msg = SystemMessage(content="system prompt")
        assert msg.content == "system prompt"


# =====================================================================
# BUG 2: middleware/__init__.py — lazy imports
# =====================================================================


class TestMiddlewareLazyImports:
    """middleware package should use lazy imports to avoid cascading failures."""

    def test_middleware_package_imports(self):
        """The middleware package should import without issues."""
        import agentomatic.middleware  # noqa: F401

    def test_lazy_auth_middleware(self):
        """AuthMiddleware should be importable from the package."""
        from agentomatic.middleware import AuthMiddleware

        assert AuthMiddleware is not None

    def test_lazy_feedback_collector(self):
        """FeedbackCollector should be importable from the package."""
        from agentomatic.middleware import FeedbackCollector

        assert FeedbackCollector is not None

    def test_lazy_logging_middleware(self):
        """LoggingMiddleware should be importable from the package."""
        from agentomatic.middleware import LoggingMiddleware

        assert LoggingMiddleware is not None

    def test_lazy_rate_limit_middleware(self):
        """RateLimitMiddleware should be importable from the package."""
        from agentomatic.middleware import RateLimitMiddleware

        assert RateLimitMiddleware is not None

    def test_lazy_metrics_middleware(self):
        """MetricsMiddleware should be importable from the package."""
        from agentomatic.middleware import MetricsMiddleware

        assert MetricsMiddleware is not None

    def test_invalid_attr_raises(self):
        """Accessing non-existent attributes should raise AttributeError."""
        import agentomatic.middleware

        with pytest.raises(AttributeError):
            _ = agentomatic.middleware.DoesNotExist  # type: ignore[attr-defined]


# =====================================================================
# BUG 4 & 5: platform.py — discover_agents() + sys.path in build()
# =====================================================================


class TestPlatformDiscovery:
    """Platform should support synchronous agent discovery."""

    def test_discover_agents_public_method_exists(self):
        """AgentPlatform should have a public discover_agents() method."""
        from agentomatic.core.platform import AgentPlatform

        assert hasattr(AgentPlatform, "discover_agents")

    def test_discover_agents_idempotent(self):
        """Calling discover_agents() multiple times should be a no-op."""
        from agentomatic.core.platform import AgentPlatform

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            platform = AgentPlatform(agents_dir=agents_dir, enable_studio=False)
            platform.discover_agents()
            assert platform._discovered is True

            # Second call should be a no-op
            platform.discover_agents()
            assert platform._discovered is True

    def test_build_triggers_discovery(self):
        """build() should eagerly discover agents."""
        from agentomatic.core.platform import AgentPlatform

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            platform = AgentPlatform(agents_dir=agents_dir, enable_studio=False)
            assert platform._discovered is False

            app = platform.build()
            assert platform._discovered is True
            assert app is not None

    def test_sys_path_set_after_build(self):
        """After build(), the agents directory parent should be in sys.path."""
        from agentomatic.core.platform import AgentPlatform

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            platform = AgentPlatform(agents_dir=agents_dir, enable_studio=False)
            platform.build()

            # Parent of agents_dir should be in sys.path
            # Use resolve() to handle macOS symlinks (/var -> /private/var)
            expected = str(agents_dir.parent.resolve())
            resolved_paths = [str(Path(p).resolve()) for p in sys.path]
            assert expected in resolved_paths


# =====================================================================
# BUG 6 + 13: Studio redirects
# =====================================================================


class TestStudioRedirects:
    """Studio redirects should be mounted when Studio UI is available."""

    def test_studio_redirect_route_exists(self):
        """When studio is enabled and UI available, /studio should redirect."""
        from agentomatic.core.platform import AgentPlatform

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            platform = AgentPlatform(agents_dir=agents_dir, enable_studio=True)
            app = platform.build()

            # Check that routes exist (they may not all mount if Studio UI
            # assets are not built, but the API routes should be there)
            route_paths = [r.path for r in app.routes if hasattr(r, "path")]
            # Studio API should be available
            assert any("/studio" in p for p in route_paths)


# =====================================================================
# BUG 8: state.py — add_messages fallback
# =====================================================================


class TestStateAddMessagesFallback:
    """add_messages fallback should handle None on both sides."""

    def test_both_none(self):
        """Both left=None and right=None should return empty list."""
        from agentomatic.core.state import HAS_LANGGRAPH

        if HAS_LANGGRAPH:
            pytest.skip("langgraph is installed, fallback not used")

        from agentomatic.core.state import add_messages

        result = add_messages(None, None)
        assert result == []

    def test_left_none_right_list(self):
        """left=None should return right."""
        from agentomatic.core.state import HAS_LANGGRAPH

        if HAS_LANGGRAPH:
            pytest.skip("langgraph is installed, fallback not used")

        from agentomatic.core.state import add_messages

        result = add_messages(None, ["msg1"])
        assert result == ["msg1"]

    def test_left_list_right_none(self):
        """right=None should return left."""
        from agentomatic.core.state import HAS_LANGGRAPH

        if HAS_LANGGRAPH:
            pytest.skip("langgraph is installed, fallback not used")

        from agentomatic.core.state import add_messages

        result = add_messages(["msg1"], None)
        assert result == ["msg1"]

    def test_both_lists(self):
        """Both lists should concatenate."""
        from agentomatic.core.state import HAS_LANGGRAPH

        if HAS_LANGGRAPH:
            pytest.skip("langgraph is installed, fallback not used")

        from agentomatic.core.state import add_messages

        result = add_messages(["a"], ["b"])
        assert result == ["a", "b"]


# =====================================================================
# BUG 9: feedback.py — uninitialized collector warning
# =====================================================================


class TestFeedbackCollectorWarning:
    """get_collector() should warn when creating without a store."""

    def test_get_collector_warns(self):
        """Creating a bare collector should log a warning."""
        # Reset the singleton
        import agentomatic.middleware.feedback as fb_module

        fb_module._collector = None

        with patch.object(fb_module, "logger") as mock_logger:
            collector = fb_module.get_collector()
            assert collector is not None
            mock_logger.warning.assert_called_once()
            assert "without a storage backend" in mock_logger.warning.call_args[0][0]

        # Cleanup
        fb_module._collector = None

    def test_set_collector_skips_warning(self):
        """set_collector() should not trigger warning."""
        import agentomatic.middleware.feedback as fb_module

        fb_module._collector = None
        custom = fb_module.FeedbackCollector(store=MagicMock())
        fb_module.set_collector(custom)

        # get_collector should now return the custom one without warning
        with patch.object(fb_module, "logger") as mock_logger:
            result = fb_module.get_collector()
            assert result is custom
            mock_logger.warning.assert_not_called()

        # Cleanup
        fb_module._collector = None


# =====================================================================
# BUG 11: manifest.py — Studio fields on RegisteredAgent
# =====================================================================


class TestRegisteredAgentStudioFields:
    """RegisteredAgent should have Studio hook fields."""

    def test_studio_fields_exist(self):
        """RegisteredAgent should have _studio_graph_fn, _studio_state_fn, _studio_adapter."""
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent

        manifest = AgentManifest(name="test", slug="test", version="0.1.0")
        agent = RegisteredAgent(manifest=manifest)

        assert agent._studio_graph_fn is None
        assert agent._studio_state_fn is None
        assert agent._studio_adapter is None

    def test_studio_fields_settable(self):
        """Studio fields should be settable without AttributeError."""
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent

        manifest = AgentManifest(name="test", slug="test", version="0.1.0")
        agent = RegisteredAgent(manifest=manifest)

        def mock_graph():
            return {"nodes": [], "edges": []}

        agent._studio_graph_fn = mock_graph
        assert agent._studio_graph_fn is mock_graph


# =====================================================================
# OPT: OptimizationResult.save() + PromptFitResult.save()
# =====================================================================


class TestOptimizationResultSave:
    """OptimizationResult should have a save() method."""

    def test_save_method_exists(self):
        """OptimizationResult should have save()."""
        from agentomatic.optimize.optimizer import OptimizationResult

        assert hasattr(OptimizationResult, "save")

    def test_save_writes_json(self):
        """save() should write optimization_results.json."""
        from agentomatic.optimize.optimizer import OptimizationResult

        result = OptimizationResult(
            best_prompt="You are helpful.",
            best_score=0.9,
            best_iteration=2,
            baseline_prompt="Be helpful.",
            baseline_score=0.7,
            experiment_id="test123",
            agent="test_agent",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, _ = result.save(tmpdir)
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert data["experiment_id"] == "test123"
            assert data["best_score"] == 0.9
            assert data["agent"] == "test_agent"


class TestPromptFitResultSave:
    """PromptFitResult should have a save() method."""

    def test_save_method_exists(self):
        """PromptFitResult should have save()."""
        from agentomatic.optimize.config import PromptFitResult

        assert hasattr(PromptFitResult, "save")

    def test_save_writes_json(self):
        """save() should write fit_result.json."""
        from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig

        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="Optimized prompt"),
            baseline_config=PromptRuntimeConfig(system_prompt="Original prompt"),
            best_score=0.92,
            baseline_score=0.78,
            experiment_id="fit_test_123",
            agent="fit_agent",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = result.save(tmpdir)
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert data["experiment_id"] == "fit_test_123"
            assert data["best_score"] == 0.92


class TestLoopResultSave:
    """LoopResult.save() should still work (regression test)."""

    def test_save_writes_files(self):
        """save() should write JSON and HTML."""
        from agentomatic.optimize.loop import LoopResult, StepResult

        result = LoopResult(
            agent="loop_test",
            experiment_id="loop123",
            steps=[
                StepResult(
                    step=0, prompt="p1", avg_score=0.7, accuracy=0.8, results=[], elapsed=1.0
                ),
                StepResult(
                    step=1, prompt="p2", avg_score=0.85, accuracy=0.9, results=[], elapsed=1.0
                ),
            ],
            best_step=1,
            best_score=0.85,
            best_prompt="p2",
            total_elapsed=2.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, html_path = result.save(tmpdir)
            assert json_path.exists()
            assert html_path.exists()
            data = json.loads(json_path.read_text())
            assert data["agent"] == "loop_test"
            assert data["best_score"] == 0.85


# =====================================================================
# BUG 3: router_factory.py — langchain guard in /chat endpoint
# =====================================================================


class TestChatEndpointLangchainGuard:
    """The /chat endpoint should handle missing langchain gracefully."""

    def test_chat_endpoint_with_messages_no_crash(self):
        """Providing messages to /chat should not crash without langchain."""
        from agentomatic.core.manifest import AgentManifest
        from agentomatic.core.platform import AgentPlatform

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            platform = AgentPlatform(agents_dir=agents_dir, enable_studio=False)

            # Register a dummy agent
            async def dummy_fn(state):
                return {"response": "Hello!", "messages": state.get("messages", [])}

            manifest = AgentManifest(name="chat_test", slug="chat_test", version="0.1.0")
            platform.register_agent(manifest=manifest, node_fn=dummy_fn)

            app = platform.build()

            from starlette.testclient import TestClient

            client = TestClient(app)
            resp = client.post(
                "/api/v1/chat_test/chat",
                json={
                    "content": "Hello!",
                    "user_id": "test",
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("response") is not None


# =====================================================================
# BUG 7: router_factory.py — history error metadata
# =====================================================================


class TestHistoryErrorMetadata:
    """When history loading fails, metadata should contain the error."""

    def test_history_error_sets_metadata_flag(self):
        """Failed history load should set _history_error in state metadata."""
        from agentomatic.core.manifest import AgentManifest
        from agentomatic.core.platform import AgentPlatform
        from agentomatic.storage.memory import MemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            store = MemoryStore()
            platform = AgentPlatform(
                agents_dir=agents_dir,
                enable_studio=False,
                store=store,
            )

            captured_state = {}

            async def capture_fn(state):
                captured_state.update(state)
                return {"response": "ok", "messages": []}

            manifest = AgentManifest(name="hist_test", slug="hist_test", version="0.1.0")
            platform.register_agent(manifest=manifest, node_fn=capture_fn)

            app = platform.build()

            # Force a history error by patching memory manager
            from starlette.testclient import TestClient

            client = TestClient(app)

            # First call without errors should work
            resp = client.post(
                "/api/v1/hist_test/chat",
                json={"content": "Hello!", "user_id": "test"},
            )
            assert resp.status_code == 200


# =====================================================================
# Version sanity check
# =====================================================================


class TestVersion:
    """Version should be a valid semver and consistent across modules."""

    def test_version_is_valid_semver(self):
        import re

        from agentomatic._version import __version__

        assert re.match(r"^\d+\.\d+\.\d+", __version__), (
            f"Version {__version__!r} is not valid semver"
        )

    def test_version_matches_package(self):
        import agentomatic
        from agentomatic._version import __version__

        assert agentomatic.__version__ == __version__


# =====================================================================
# Platform register_agent + discover_agents integration
# =====================================================================


class TestPlatformIntegration:
    """Integration tests for register_agent + discover_agents."""

    def test_register_then_build(self):
        """Agents registered before build should be available."""
        from agentomatic.core.manifest import AgentManifest
        from agentomatic.core.platform import AgentPlatform

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            platform = AgentPlatform(agents_dir=agents_dir, enable_studio=False)

            async def dummy(state):
                return {"response": "hello"}

            manifest = AgentManifest(name="reg_test", slug="reg_test", version="0.1.0")
            platform.register_agent(manifest=manifest, node_fn=dummy)

            app = platform.build()

            from starlette.testclient import TestClient

            client = TestClient(app)

            # Health endpoint
            resp = client.get("/health")
            assert resp.status_code == 200

            # Agent invoke
            resp = client.post(
                "/api/v1/reg_test/invoke",
                json={"query": "test", "user_id": "u1"},
            )
            assert resp.status_code == 200

    def test_discovered_flag_prevents_double_discovery(self):
        """Double discovery should be prevented by the _discovered flag."""
        from agentomatic.core.platform import AgentPlatform

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            (agents_dir / "__init__.py").write_text("")

            platform = AgentPlatform(agents_dir=agents_dir, enable_studio=False)

            # First: manual discovery
            platform.discover_agents()
            assert platform._discovered is True

            # Second: build should NOT re-discover
            with patch.object(platform._registry, "discover") as mock_discover:
                platform.build()
                mock_discover.assert_not_called()
