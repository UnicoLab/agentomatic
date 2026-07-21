"""Tests for the agentomatic.stacks module."""

from __future__ import annotations

import pytest
import yaml

from agentomatic.stacks.defaults import (
    BUILTIN_STACKS,
    get_default_local_stack,
    get_default_remote_stack,
    get_default_stack_yaml,
)
from agentomatic.stacks.manager import (
    AuthStackConfig,
    DatabaseStackEntry,
    EmbeddingStackEntry,
    FeaturesStackEntry,
    LLMStackEntry,
    StackConfig,
    StackManager,
)

# ---------------------------------------------------------------------------
# LLMStackEntry
# ---------------------------------------------------------------------------


class TestLLMStackEntry:
    """Tests for LLMStackEntry model."""

    def test_creation_with_defaults(self) -> None:
        entry = LLMStackEntry(provider="openai", model="gpt-4o")
        assert entry.provider == "openai"
        assert entry.model == "gpt-4o"
        assert entry.temperature == 0.1
        assert entry.max_tokens == 8192
        assert entry.api_key == ""
        assert entry.base_url == ""
        assert entry.extra == {}

    def test_creation_with_custom_values(self) -> None:
        entry = LLMStackEntry(
            provider="ollama",
            model="mistral:7b",
            temperature=0.5,
            max_tokens=2048,
            api_key="sk-test",
            base_url="http://localhost:11434",
            extra={"top_p": 0.9},
        )
        assert entry.temperature == 0.5
        assert entry.max_tokens == 2048
        assert entry.api_key == "sk-test"
        assert entry.extra == {"top_p": 0.9}

    def test_serialization_roundtrip(self) -> None:
        entry = LLMStackEntry(provider="azure", model="gpt-4")
        data = entry.model_dump()
        restored = LLMStackEntry.model_validate(data)
        assert restored == entry


# ---------------------------------------------------------------------------
# StackConfig
# ---------------------------------------------------------------------------


class TestStackConfig:
    """Tests for StackConfig model."""

    def test_creation_with_defaults(self) -> None:
        config = StackConfig(name="test")
        assert config.name == "test"
        assert config.description == ""
        assert config.llm == {}
        assert isinstance(config.embedding, EmbeddingStackEntry)
        assert isinstance(config.database, DatabaseStackEntry)
        assert isinstance(config.features, FeaturesStackEntry)
        assert isinstance(config.auth, AuthStackConfig)
        assert config.env_file is None
        assert config.environment == {}
        assert config.agent_overrides == {}

    def test_creation_with_llm_profiles(self) -> None:
        config = StackConfig(
            name="multi",
            llm={
                "default": LLMStackEntry(provider="openai", model="gpt-4o"),
                "fast": LLMStackEntry(provider="openai", model="gpt-4o-mini"),
            },
        )
        assert len(config.llm) == 2
        assert config.llm["default"].model == "gpt-4o"
        assert config.llm["fast"].model == "gpt-4o-mini"

    def test_features_defaults(self) -> None:
        config = StackConfig(name="f")
        assert config.features.enable_streaming is True
        assert config.features.enable_a2a is True
        assert config.features.enable_metrics is False
        assert config.features.enable_rate_limit is False
        assert config.features.enable_auth is False
        assert config.features.enable_db is False


# ---------------------------------------------------------------------------
# StackManager — interpolate_env
# ---------------------------------------------------------------------------


class TestInterpolateEnv:
    """Tests for StackManager.interpolate_env()."""

    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "secret123")
        mgr = StackManager()
        assert mgr.interpolate_env("${MY_KEY}") == "secret123"

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        mgr = StackManager()
        result = mgr.interpolate_env("postgres://${HOST}:${PORT}/db")
        assert result == "postgres://localhost:5432/db"

    def test_missing_var_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        mgr = StackManager()
        assert mgr.interpolate_env("${NONEXISTENT_VAR_XYZ}") == ""

    def test_no_pattern_passthrough(self) -> None:
        mgr = StackManager()
        assert mgr.interpolate_env("no variables here") == "no variables here"


# ---------------------------------------------------------------------------
# StackManager — load
# ---------------------------------------------------------------------------


class TestStackManagerLoad:
    """Tests for StackManager.load()."""

    def test_load_from_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        stack_data = {
            "name": "test-stack",
            "description": "A test stack",
            "llm": {
                "default": {
                    "provider": "ollama",
                    "model": "mistral:7b",
                    "temperature": 0.2,
                },
            },
        }
        yaml_file = tmp_path / "test-stack.yaml"
        yaml_file.write_text(yaml.dump(stack_data))

        mgr = StackManager(stacks_dir=tmp_path)
        stack = mgr.load("test-stack")

        assert stack.name == "test-stack"
        assert stack.description == "A test stack"
        assert "default" in stack.llm
        assert stack.llm["default"].provider == "ollama"
        assert stack.llm["default"].temperature == 0.2

    def test_load_missing_file_falls_back_to_builtin(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        mgr = StackManager(stacks_dir=tmp_path)
        stack = mgr.load("local")
        assert stack.name == "local"
        assert "default" in stack.llm

    def test_load_unknown_stack_returns_minimal(self, tmp_path: pytest.TempPathFactory) -> None:
        mgr = StackManager(stacks_dir=tmp_path)
        stack = mgr.load("totally_unknown")
        assert stack.name == "totally_unknown"

    def test_from_file(self, tmp_path: pytest.TempPathFactory) -> None:
        stack_data = {
            "name": "from-file",
            "llm": {
                "default": {"provider": "openai", "model": "gpt-4o"},
            },
        }
        yaml_file = tmp_path / "from-file.yaml"
        yaml_file.write_text(yaml.dump(stack_data))

        mgr = StackManager.from_file(yaml_file)
        assert mgr._active_stack is not None
        assert mgr._active_stack.name == "from-file"


# ---------------------------------------------------------------------------
# StackManager — get_llm_config
# ---------------------------------------------------------------------------


class TestGetLLMConfig:
    """Tests for StackManager.get_llm_config()."""

    def test_happy_path(self, tmp_path: pytest.TempPathFactory) -> None:
        stack_data = {
            "name": "llm-test",
            "llm": {
                "default": {"provider": "openai", "model": "gpt-4o"},
                "fast": {"provider": "openai", "model": "gpt-4o-mini"},
            },
        }
        yaml_file = tmp_path / "llm-test.yaml"
        yaml_file.write_text(yaml.dump(stack_data))

        mgr = StackManager(stacks_dir=tmp_path)
        mgr.load("llm-test")

        entry = mgr.get_llm_config("default")
        assert entry.model == "gpt-4o"

        fast = mgr.get_llm_config("fast")
        assert fast.model == "gpt-4o-mini"

    def test_missing_profile_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        stack_data = {
            "name": "llm-test",
            "llm": {"default": {"provider": "openai", "model": "gpt-4o"}},
        }
        yaml_file = tmp_path / "llm-test.yaml"
        yaml_file.write_text(yaml.dump(stack_data))

        mgr = StackManager(stacks_dir=tmp_path)
        mgr.load("llm-test")

        with pytest.raises(ValueError, match="not found"):
            mgr.get_llm_config("nonexistent")

    def test_no_stack_loaded_raises(self) -> None:
        mgr = StackManager()
        with pytest.raises(ValueError, match="No stack loaded"):
            mgr.get_llm_config()


# ---------------------------------------------------------------------------
# StackManager — resolve
# ---------------------------------------------------------------------------


class TestResolve:
    """Tests for StackManager.resolve()."""

    def test_resolve_interpolates_env_vars(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "resolved-key-123")
        stack_data = {
            "name": "resolve-test",
            "llm": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "${TEST_API_KEY}",
                },
            },
        }
        yaml_file = tmp_path / "resolve-test.yaml"
        yaml_file.write_text(yaml.dump(stack_data))

        mgr = StackManager(stacks_dir=tmp_path)
        mgr.load("resolve-test")
        resolved = mgr.resolve()

        assert resolved.llm["default"].api_key == "resolved-key-123"

    def test_resolve_no_stack_raises(self) -> None:
        mgr = StackManager()
        with pytest.raises(ValueError, match="No stack loaded"):
            mgr.resolve()


# ---------------------------------------------------------------------------
# StackManager — get_agent_llm_config
# ---------------------------------------------------------------------------


class TestGetAgentLLMConfig:
    """Tests for StackManager.get_agent_llm_config()."""

    def test_returns_override(self, tmp_path: pytest.TempPathFactory) -> None:
        stack_data = {
            "name": "agent-test",
            "llm": {"default": {"provider": "openai", "model": "gpt-4o"}},
            "agent_overrides": {
                "planner": {"provider": "openai", "model": "o1-preview"},
            },
        }
        yaml_file = tmp_path / "agent-test.yaml"
        yaml_file.write_text(yaml.dump(stack_data))

        mgr = StackManager(stacks_dir=tmp_path)
        mgr.load("agent-test")

        override = mgr.get_agent_llm_config("planner")
        assert override is not None
        assert override.model == "o1-preview"

    def test_returns_none_for_unknown_agent(self, tmp_path: pytest.TempPathFactory) -> None:
        stack_data = {
            "name": "agent-test",
            "llm": {"default": {"provider": "openai", "model": "gpt-4o"}},
        }
        yaml_file = tmp_path / "agent-test.yaml"
        yaml_file.write_text(yaml.dump(stack_data))

        mgr = StackManager(stacks_dir=tmp_path)
        mgr.load("agent-test")

        assert mgr.get_agent_llm_config("unknown_agent") is None

    def test_returns_none_when_no_stack_loaded(self) -> None:
        mgr = StackManager()
        assert mgr.get_agent_llm_config("any") is None


# ---------------------------------------------------------------------------
# Default stacks
# ---------------------------------------------------------------------------


class TestDefaultStacks:
    """Tests for built-in default stack factories."""

    def test_local_stack_factory(self) -> None:
        stack = get_default_local_stack()
        assert stack.name == "local"
        assert "default" in stack.llm
        assert stack.llm["default"].provider == "ollama"
        assert stack.features.enable_auth is False
        assert stack.auth.method == "api_key"

    def test_remote_stack_factory(self) -> None:
        stack = get_default_remote_stack()
        assert stack.name == "remote"
        assert "default" in stack.llm
        assert stack.llm["default"].provider == "openai"
        assert stack.llm["default"].api_key == "${OPENAI_API_KEY}"
        assert stack.features.enable_auth is True
        assert stack.auth.method == "jwt"
        assert stack.database.url == "${DATABASE_URL}"

    def test_builtin_stacks_mapping(self) -> None:
        assert "local" in BUILTIN_STACKS
        assert "remote" in BUILTIN_STACKS
        assert callable(BUILTIN_STACKS["local"])
        assert callable(BUILTIN_STACKS["remote"])

    def test_get_default_stack_yaml_local(self) -> None:
        yaml_str = get_default_stack_yaml("local")
        data = yaml.safe_load(yaml_str)
        assert data["name"] == "local"
        assert "default" in data["llm"]

    def test_get_default_stack_yaml_remote(self) -> None:
        yaml_str = get_default_stack_yaml("remote")
        data = yaml.safe_load(yaml_str)
        assert data["name"] == "remote"
        assert data["llm"]["default"]["provider"] == "openai"

    def test_get_default_stack_yaml_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown built-in stack"):
            get_default_stack_yaml("nonexistent")

    def test_yaml_roundtrip(self) -> None:
        """Verify that YAML output can be loaded back into a StackConfig."""
        yaml_str = get_default_stack_yaml("local")
        data = yaml.safe_load(yaml_str)
        stack = StackConfig.model_validate(data)
        assert stack.name == "local"
        assert stack.llm["default"].provider == "ollama"
