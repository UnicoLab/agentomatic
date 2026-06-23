"""Tests for agentomatic v0.6 core module enhancements.

Covers:
- AgentManifest new fields (llm_config, security_policy, delegation_targets)
- RegisteredAgent new fields (llm_config, schema_validator, security_policy, delegation_config)
- SchemaValidator (input/output validation, OpenAPI schemas, property checks)
- PromptManager LangChain integration (import-error paths)
- LLM named instances (get_named_llm, reset_llm, get_llm_for_agent)
- load_environment / get_settings_from_dict
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Lightweight stand-in for LLM objects in tests that don't need langchain."""

    def __init__(self, *, id: int = 0, provider: str = "dummy") -> None:
        self.id = id
        self.provider = provider


# ---------------------------------------------------------------------------
# Test models for SchemaValidator
# ---------------------------------------------------------------------------


class SampleRequest(BaseModel):
    """Sample request model for validation tests."""

    query: str
    temperature: float = 0.5


class SampleResponse(BaseModel):
    """Sample response model for validation tests."""

    answer: str
    score: float


# ===================================================================
# 1. AgentManifest new fields
# ===================================================================


class TestAgentManifestV06:
    """Tests for v0.6 AgentManifest fields."""

    def test_manifest_defaults(self):
        """Manifest with only name+slug gets correct v0.6 defaults."""
        from agentomatic.core.manifest import AgentManifest

        m = AgentManifest(name="demo", slug="demo-agent")

        assert m.llm_config == {"default": "default"}
        assert m.security_policy == ""
        assert m.delegation_targets == []

    def test_manifest_custom_llm_config(self):
        """Manifest accepts a custom llm_config mapping."""
        from agentomatic.core.manifest import AgentManifest

        cfg = {"default": "fast", "judge": "gpt-4o"}
        m = AgentManifest(name="demo", slug="demo-agent", llm_config=cfg)

        assert m.llm_config == {"default": "fast", "judge": "gpt-4o"}
        assert m.llm_config["judge"] == "gpt-4o"

    def test_manifest_with_delegation_targets(self):
        """Manifest stores delegation_targets list correctly."""
        from agentomatic.core.manifest import AgentManifest

        m = AgentManifest(
            name="orch",
            slug="orch-agent",
            delegation_targets=["agent_a", "agent_b"],
        )

        assert m.delegation_targets == ["agent_a", "agent_b"]
        assert len(m.delegation_targets) == 2

    def test_manifest_frozen(self):
        """Frozen dataclass prevents attribute mutation."""
        from agentomatic.core.manifest import AgentManifest

        m = AgentManifest(name="demo", slug="demo-agent")

        with pytest.raises((FrozenInstanceError, AttributeError)):
            m.name = "changed"  # type: ignore[misc]

        with pytest.raises((FrozenInstanceError, AttributeError)):
            m.llm_config = {}  # type: ignore[misc]

        with pytest.raises((FrozenInstanceError, AttributeError)):
            m.delegation_targets = ["x"]  # type: ignore[misc]


# ===================================================================
# 2. RegisteredAgent new fields
# ===================================================================


class TestRegisteredAgentV06:
    """Tests for v0.6 RegisteredAgent fields."""

    def test_registered_agent_defaults(self):
        """RegisteredAgent defaults v0.6 fields to None."""
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent

        manifest = AgentManifest(name="test", slug="test-agent")
        ra = RegisteredAgent(manifest=manifest)

        assert ra.llm_config is None
        assert ra.schema_validator is None
        assert ra.security_policy is None
        assert ra.delegation_config is None

    def test_registered_agent_with_validator(self):
        """RegisteredAgent stores a SchemaValidator instance."""
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent
        from agentomatic.core.schemas import SchemaValidator

        manifest = AgentManifest(name="val", slug="val-agent")
        validator = SchemaValidator(
            request_model=SampleRequest,
            response_model=SampleResponse,
        )
        ra = RegisteredAgent(manifest=manifest, schema_validator=validator)

        assert ra.schema_validator is validator
        assert ra.schema_validator.has_request_schema is True
        assert ra.schema_validator.has_response_schema is True

    def test_registered_agent_name_slug_proxy(self):
        """RegisteredAgent.name and .slug proxy to the manifest."""
        from agentomatic.core.manifest import AgentManifest, RegisteredAgent

        manifest = AgentManifest(name="proxy", slug="proxy-slug")
        ra = RegisteredAgent(manifest=manifest)

        assert ra.name == "proxy"
        assert ra.slug == "proxy-slug"


# ===================================================================
# 3. SchemaValidator
# ===================================================================


class TestSchemaValidator:
    """Tests for the SchemaValidator class."""

    def test_validator_validate_input_valid(self):
        """Valid input passes through validation and is coerced."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator(request_model=SampleRequest)
        result = sv.validate_input({"query": "hello", "temperature": 0.9})

        assert result["query"] == "hello"
        assert result["temperature"] == 0.9

    def test_validator_validate_input_defaults(self):
        """Input with missing optional field gets the Pydantic default."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator(request_model=SampleRequest)
        result = sv.validate_input({"query": "hello"})

        assert result["query"] == "hello"
        assert result["temperature"] == 0.5  # default

    def test_validator_validate_input_invalid(self):
        """Missing required field raises ValidationError."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator(request_model=SampleRequest)

        with pytest.raises(ValidationError):
            sv.validate_input({"temperature": 0.7})  # missing 'query'

    def test_validator_validate_input_no_schema(self):
        """Without a request model, data passes through unchanged."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator()  # no models
        data = {"anything": "goes", "num": 42}
        result = sv.validate_input(data)

        assert result is data  # exact same object

    def test_validator_validate_output_valid(self):
        """Valid output passes validation and is returned."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator(response_model=SampleResponse)
        data = {"answer": "yes", "score": 0.95}
        result = sv.validate_output(data)

        assert result == data

    def test_validator_validate_output_invalid_logs_warning(self, caplog):
        """Invalid output logs a warning but still returns the data."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator(response_model=SampleResponse)
        bad_data = {"wrong_field": "oops"}

        with patch("agentomatic.core.schemas.logger") as mock_logger:
            result = sv.validate_output(bad_data)
            mock_logger.warning.assert_called_once()
            call_msg = mock_logger.warning.call_args[0][0]
            assert "Output validation warning" in call_msg
            assert "SampleResponse" in call_msg

        # Data is still returned despite validation failure
        assert result is bad_data

    def test_validator_validate_output_no_schema(self):
        """Without a response model, data passes through unchanged."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator()
        data = {"key": "value"}
        result = sv.validate_output(data)

        assert result is data

    def test_validator_get_openapi_schemas(self):
        """Returns correct JSON schemas for both models."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator(
            request_model=SampleRequest,
            response_model=SampleResponse,
        )
        schemas = sv.get_openapi_schemas()

        assert "request" in schemas
        assert "response" in schemas

        # Check request schema structure
        req_schema = schemas["request"]
        assert req_schema["title"] == "SampleRequest"
        assert "query" in req_schema["properties"]
        assert "temperature" in req_schema["properties"]

        # Check response schema structure
        resp_schema = schemas["response"]
        assert resp_schema["title"] == "SampleResponse"
        assert "answer" in resp_schema["properties"]
        assert "score" in resp_schema["properties"]

    def test_validator_get_openapi_schemas_partial(self):
        """Only included models appear in the schema output."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator(request_model=SampleRequest)
        schemas = sv.get_openapi_schemas()

        assert "request" in schemas
        assert "response" not in schemas

    def test_validator_get_openapi_schemas_empty(self):
        """No models means empty schema dict."""
        from agentomatic.core.schemas import SchemaValidator

        sv = SchemaValidator()
        schemas = sv.get_openapi_schemas()

        assert schemas == {}

    def test_validator_has_request_schema(self):
        """has_request_schema property reports correctly."""
        from agentomatic.core.schemas import SchemaValidator

        assert SchemaValidator(request_model=SampleRequest).has_request_schema is True
        assert SchemaValidator().has_request_schema is False

    def test_validator_has_response_schema(self):
        """has_response_schema property reports correctly."""
        from agentomatic.core.schemas import SchemaValidator

        assert SchemaValidator(response_model=SampleResponse).has_response_schema is True
        assert SchemaValidator().has_response_schema is False


# ===================================================================
# 4. PromptManager LangChain integration
# ===================================================================


class TestPromptManagerLangChain:
    """Tests for PromptManager LangChain integration fallback paths."""

    def _make_manager(self, tmp_path):
        """Create a PromptManager with sample prompt data."""
        import json

        from agentomatic.prompts.manager import PromptManager

        prompts = {
            "v1": {
                "system": "You are a helpful assistant named {name}.",
                "user_template": "Answer: {question}",
            },
        }
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text(json.dumps(prompts))

        return PromptManager(agent_name="test", prompts_file=prompts_file)

    def test_as_langchain_template_returns_none_when_not_installed(self, tmp_path):
        """as_langchain_template returns None when langchain_core is missing."""
        pm = self._make_manager(tmp_path)

        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.prompts": None}):
            with patch(
                "builtins.__import__",
                side_effect=_import_error_for("langchain_core.prompts"),
            ):
                result = pm.as_langchain_template("v1", "system")

        assert result is None

    def test_as_langchain_template_returns_none_for_missing_version(self, tmp_path):
        """as_langchain_template returns None when version doesn't exist."""
        pm = self._make_manager(tmp_path)
        result = pm.as_langchain_template("v999", "system")
        assert result is None

    def test_as_chat_template_returns_none_for_missing_version(self, tmp_path):
        """as_chat_template returns None when version doesn't exist."""
        pm = self._make_manager(tmp_path)
        result = pm.as_chat_template("v999")
        assert result is None

    def test_as_chat_template_returns_none_when_not_installed(self, tmp_path):
        """as_chat_template returns None when langchain_core is missing."""
        pm = self._make_manager(tmp_path)

        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.prompts": None}):
            with patch(
                "builtins.__import__",
                side_effect=_import_error_for("langchain_core.prompts"),
            ):
                result = pm.as_chat_template("v1")

        assert result is None


def _import_error_for(blocked_module: str):
    """Return a side_effect function that raises ImportError for a specific module."""
    original_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def _side_effect(name, *args, **kwargs):
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    return _side_effect


# ===================================================================
# 5. LLM named instances
# ===================================================================


class TestLLMNamedInstances:
    """Tests for get_named_llm, reset_llm, and get_llm_for_agent."""

    @pytest.fixture(autouse=True)
    def _reset_llm_state(self):
        """Reset LLM singletons before and after each test."""
        from agentomatic.providers.llm import reset_llm

        reset_llm()
        yield
        reset_llm()

    @pytest.fixture(autouse=True)
    def _mock_build_llm(self):
        """Mock _build_llm and _build_dummy_llm so tests don't need langchain."""
        counter = {"n": 0}

        def fake_build(provider, **kwargs):
            if provider == "nonexistent_xyz":
                raise ValueError(f"Unknown LLM provider: {provider}")
            counter["n"] += 1
            return FakeLLM(id=counter["n"], provider=provider)

        def fake_dummy():
            counter["n"] += 1
            return FakeLLM(id=counter["n"], provider="dummy")

        with (
            patch("agentomatic.providers.llm._build_llm", side_effect=fake_build),
            patch("agentomatic.providers.llm._build_dummy_llm", side_effect=fake_dummy),
        ):
            yield

    def test_get_named_llm_creates_instance(self):
        """get_named_llm with provider='dummy' returns an LLM instance."""
        from agentomatic.providers.llm import get_named_llm

        llm = get_named_llm(name="test_default", provider="dummy")
        assert llm is not None
        assert isinstance(llm, FakeLLM)

    def test_get_named_llm_returns_cached(self):
        """Same name returns the exact same cached object."""
        from agentomatic.providers.llm import get_named_llm

        first = get_named_llm(name="cached", provider="dummy")
        second = get_named_llm(name="cached", provider="dummy")

        assert first is second

    def test_get_named_llm_different_names(self):
        """Different names produce different LLM instances."""
        from agentomatic.providers.llm import get_named_llm

        a = get_named_llm(name="alpha", provider="dummy")
        b = get_named_llm(name="beta", provider="dummy")

        assert a is not b

    def test_reset_llm_clears_named(self):
        """reset_llm clears named instances so next call creates fresh ones."""
        from agentomatic.providers.llm import get_named_llm, reset_llm

        first = get_named_llm(name="ephemeral", provider="dummy")
        reset_llm()
        second = get_named_llm(name="ephemeral", provider="dummy")

        assert first is not second

    def test_get_llm_for_agent_no_stack(self):
        """Without stack_manager, get_llm_for_agent falls back to get_llm()."""
        from agentomatic.providers.llm import get_llm, get_llm_for_agent

        result = get_llm_for_agent(agent_name="demo", role="default")
        singleton = get_llm()

        assert result is singleton

    def test_get_named_llm_unknown_provider_falls_to_dummy(self):
        """Unknown provider name triggers fallback to dummy LLM."""
        from agentomatic.providers.llm import get_named_llm

        llm = get_named_llm(name="unknown_prov", provider="nonexistent_xyz")
        assert llm is not None
        assert isinstance(llm, FakeLLM)
        assert llm.provider == "dummy"  # fell back to dummy


# ===================================================================
# 6. load_environment and get_settings_from_dict
# ===================================================================


class TestConfigSettings:
    """Tests for load_environment and get_settings_from_dict."""

    def test_load_environment_no_dotenv(self):
        """load_environment doesn't crash when python-dotenv is missing."""
        from agentomatic.config.settings import load_environment

        with patch.dict("sys.modules", {"dotenv": None}):
            with patch(
                "builtins.__import__",
                side_effect=_import_error_for("dotenv"),
            ):
                # Should not raise
                load_environment()

    def test_load_environment_with_env_file(self, tmp_path):
        """load_environment processes a real .env file without errors."""
        from agentomatic.config.settings import load_environment

        env_file = tmp_path / ".env"
        env_file.write_text("SOME_TEST_VAR=hello\n")

        # Should not raise regardless of whether dotenv is installed
        load_environment(env_file)

    def test_get_settings_from_dict(self):
        """get_settings_from_dict creates PlatformSettings from a dict."""
        from agentomatic.config.settings import PlatformSettings, get_settings_from_dict

        overrides = {
            "app_name": "Test App",
            "app_env": "testing",
            "log_level": "DEBUG",
        }
        settings = get_settings_from_dict(overrides)

        assert isinstance(settings, PlatformSettings)
        assert settings.app_name == "Test App"
        assert settings.app_env == "testing"
        assert settings.log_level == "DEBUG"

    def test_get_settings_from_dict_nested(self):
        """get_settings_from_dict handles nested configuration."""
        from agentomatic.config.settings import get_settings_from_dict

        overrides = {
            "llm": {"provider": "openai", "model": "gpt-4"},
            "features": {"enable_streaming": False},
        }
        settings = get_settings_from_dict(overrides)

        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4"
        assert settings.features.enable_streaming is False

    def test_get_settings_from_dict_defaults(self):
        """Empty overrides still produce valid PlatformSettings with defaults."""
        from agentomatic.config.settings import get_settings_from_dict

        settings = get_settings_from_dict({})

        assert settings.app_name == "Agentomatic Platform"
        assert settings.app_env == "development"
        assert settings.llm.provider == "ollama"
