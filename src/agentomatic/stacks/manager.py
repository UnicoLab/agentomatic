"""Stack manager for loading and resolving multi-environment LLM configurations."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentomatic.config.settings import PlatformSettings

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class LLMStackEntry(BaseModel):
    """Single LLM provider entry within a stack.

    Attributes:
        provider: LLM provider identifier.
        model: Model name / identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for generation.
        api_key: API key; supports ``${ENV_VAR}`` interpolation.
        base_url: Custom base URL for the provider endpoint.
        extra: Arbitrary provider-specific parameters.
    """

    provider: str = Field(
        ...,
        description="LLM provider: ollama | openai | azure | vertex | google_genai",
    )
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)
    api_key: str = Field("", description="API key (supports ${ENV_VAR} interpolation)")
    base_url: str = Field("", description="Custom base URL for the provider")
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific params")


class AuthStackConfig(BaseModel):
    """Authentication configuration for a stack.

    Attributes:
        method: Auth method — ``"api_key"`` or ``"jwt"``.
        api_key: Static API key (when *method* is ``"api_key"``).
        jwks_url: JWKS endpoint URL (when *method* is ``"jwt"``).
        issuer: JWT issuer claim.
        audience: JWT audience claim.
    """

    method: str = Field("api_key", description="Auth method: api_key | jwt")
    api_key: str = Field("", description="Static API key")
    jwks_url: str = Field("", description="JWKS endpoint URL")
    issuer: str = Field("", description="JWT issuer")
    audience: str = Field("", description="JWT audience")


class EmbeddingStackEntry(BaseModel):
    """Embedding provider configuration.

    Attributes:
        provider: Embedding provider identifier.
        model: Embedding model name.
        dimension: Embedding vector dimension.
    """

    provider: str = Field("dummy", description="Embedding provider")
    model: str = Field("nomic-embed-text", description="Embedding model")
    dimension: int = Field(768, ge=1, description="Vector dimension")


class DatabaseStackEntry(BaseModel):
    """Database connection configuration.

    Attributes:
        url: Async database URL (supports ``${ENV_VAR}`` interpolation).
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections beyond *pool_size*.
    """

    url: str = Field(
        "sqlite+aiosqlite:///data/platform.db",
        description="Database URL",
    )
    pool_size: int = Field(10, ge=1, description="Connection pool size")
    max_overflow: int = Field(20, ge=0, description="Max overflow connections")


class FeaturesStackEntry(BaseModel):
    """Feature flags for a stack.

    Attributes:
        enable_streaming: Enable SSE streaming endpoints.
        enable_a2a: Enable A2A protocol.
        enable_metrics: Enable Prometheus metrics.
        enable_rate_limit: Enable rate limiting.
        enable_auth: Enable authentication.
        enable_db: Enable database storage.
    """

    enable_streaming: bool = Field(True, description="Enable SSE streaming")
    enable_a2a: bool = Field(True, description="Enable A2A protocol")
    enable_metrics: bool = Field(False, description="Enable Prometheus metrics")
    enable_rate_limit: bool = Field(False, description="Enable rate limiting")
    enable_auth: bool = Field(False, description="Enable authentication")
    enable_db: bool = Field(False, description="Enable database storage")


class StackConfig(BaseModel):
    """Complete stack configuration.

    Bundles LLM profiles, embedding, database, feature flags, authentication,
    and optional agent-level overrides into a single deployable unit.

    Attributes:
        name: Human-readable stack name.
        description: Optional description of the stack's purpose.
        llm: Named LLM profiles (e.g. ``"default"``, ``"fast"``, ``"judge"``).
        embedding: Embedding provider settings.
        database: Database connection settings.
        features: Feature flags.
        auth: Authentication settings.
        env_file: Optional path to a ``.env`` file to load.
        environment: Inline environment variable overrides.
        agent_overrides: Per-agent LLM overrides keyed by agent name.
    """

    name: str = Field(..., description="Stack name")
    description: str = Field("", description="Stack description")
    llm: dict[str, LLMStackEntry] = Field(
        default_factory=dict,
        description="Named LLM profiles",
    )
    embedding: EmbeddingStackEntry = Field(default_factory=EmbeddingStackEntry)
    database: DatabaseStackEntry = Field(default_factory=DatabaseStackEntry)
    features: FeaturesStackEntry = Field(default_factory=FeaturesStackEntry)
    auth: AuthStackConfig = Field(default_factory=AuthStackConfig)
    env_file: str | None = Field(None, description="Path to .env file")
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Inline environment variable overrides",
    )
    agent_overrides: dict[str, LLMStackEntry] = Field(
        default_factory=dict,
        description="Per-agent LLM overrides",
    )


# ---------------------------------------------------------------------------
# Environment-variable interpolation pattern
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


# ---------------------------------------------------------------------------
# StackManager
# ---------------------------------------------------------------------------


class StackManager:
    """Load, resolve, and manage stack configurations.

    The manager reads YAML stack files from a directory, applies dotenv files,
    interpolates ``${ENV_VAR}`` references, and exposes convenience accessors
    for LLM profiles and platform settings.

    Args:
        stacks_dir: Directory containing ``*.yaml`` stack files.

    Example::

        mgr = StackManager("stacks/")
        stack = mgr.load("local")
        llm = mgr.get_llm_config("default")
    """

    def __init__(self, stacks_dir: str | Path = "stacks/") -> None:
        self._stacks_dir = Path(stacks_dir)
        self._active_stack: StackConfig | None = None

    # -- Constructors -------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> StackManager:
        """Create a StackManager from a single YAML file.

        The file's parent directory is used as the stacks directory, and the
        stack is immediately loaded.

        Args:
            path: Path to a stack YAML file.

        Returns:
            A StackManager with the stack loaded.
        """
        path = Path(path)
        manager = cls(stacks_dir=path.parent)
        stem = path.stem
        manager.load(stem)
        return manager

    # -- Core API -----------------------------------------------------------

    def load(self, stack_name: str = "local") -> StackConfig:
        """Load a stack configuration from YAML.

        If the file is not found, falls back to built-in defaults from
        :mod:`agentomatic.stacks.defaults`.

        Args:
            stack_name: Name of the stack (file stem, without ``.yaml``).

        Returns:
            The loaded and activated :class:`StackConfig`.
        """
        stack_path = self._stacks_dir / f"{stack_name}.yaml"

        try:
            raw = stack_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            logger.info("Loaded stack '{}' from {}", stack_name, stack_path)
        except FileNotFoundError:
            logger.warning(
                "Stack file '{}' not found — trying built-in defaults",
                stack_path,
            )
            data = self._builtin_fallback(stack_name)

        stack = StackConfig.model_validate(data)

        # Apply dotenv and inline env vars
        self.apply_dotenv(stack.env_file)
        for key, value in stack.environment.items():
            os.environ.setdefault(key, value)

        from agentomatic.config.settings import reset_settings

        reset_settings()

        self._active_stack = stack
        return stack

    def get_llm_config(self, name: str = "default") -> LLMStackEntry:
        """Return the named LLM config from the active stack.

        Args:
            name: Profile name (e.g. ``"default"``, ``"fast"``, ``"judge"``).

        Returns:
            The :class:`LLMStackEntry` for the requested profile.

        Raises:
            ValueError: If no stack is loaded or the profile name is missing.
        """
        if self._active_stack is None:
            raise ValueError("No stack loaded — call load() first")
        if name not in self._active_stack.llm:
            available = ", ".join(self._active_stack.llm) or "(none)"
            raise ValueError(
                f"LLM profile '{name}' not found in stack "
                f"'{self._active_stack.name}'. Available: {available}"
            )
        return self._active_stack.llm[name]

    def get_settings(self) -> PlatformSettings:
        """Construct a :class:`PlatformSettings` from the active stack.

        Returns:
            A fully populated ``PlatformSettings`` instance.

        Raises:
            ValueError: If no stack is loaded.
        """
        from agentomatic.config.settings import (
            AuthSettings,
            DatabaseSettings,
            EmbeddingSettings,
            FeatureSettings,
            LLMSettings,
            PlatformSettings,
        )

        if self._active_stack is None:
            raise ValueError("No stack loaded — call load() first")

        stack = self.resolve()

        # Pick the "default" LLM profile (or first available)
        llm_entry: LLMStackEntry | None = stack.llm.get("default")
        if llm_entry is None and stack.llm:
            llm_entry = next(iter(stack.llm.values()))

        llm_kwargs: dict[str, Any] = {}
        if llm_entry is not None:
            llm_kwargs.update(
                provider=llm_entry.provider,
                model=llm_entry.model,
                temperature=llm_entry.temperature,
                max_tokens=llm_entry.max_tokens,
            )
            if llm_entry.provider == "ollama" and llm_entry.base_url:
                llm_kwargs["ollama_base_url"] = llm_entry.base_url
            elif llm_entry.provider == "openai" and llm_entry.api_key:
                llm_kwargs["openai_api_key"] = llm_entry.api_key

        return PlatformSettings(
            llm=LLMSettings(**llm_kwargs),
            embedding=EmbeddingSettings(
                provider=stack.embedding.provider,
                model=stack.embedding.model,
                dimension=stack.embedding.dimension,
            ),
            db=DatabaseSettings(
                url=stack.database.url,
                pool_size=stack.database.pool_size,
                max_overflow=stack.database.max_overflow,
            ),
            features=FeatureSettings(
                enable_streaming=stack.features.enable_streaming,
                enable_a2a=stack.features.enable_a2a,
                enable_metrics=stack.features.enable_metrics,
                enable_rate_limit=stack.features.enable_rate_limit,
                enable_auth=stack.features.enable_auth,
                enable_db=stack.features.enable_db,
            ),
            auth=AuthSettings(api_key=stack.auth.api_key),
        )  # type: ignore[call-arg]

    def apply_dotenv(self, env_file: str | None = None) -> None:
        """Load environment variables from a ``.env`` file if available.

        Uses ``python-dotenv`` when installed; silently skips otherwise.

        Args:
            env_file: Explicit path to a ``.env`` file. Falls back to the
                stack's ``env_file`` field if not provided.
        """
        target = env_file
        if target is None and self._active_stack is not None:
            target = self._active_stack.env_file
        if not target:
            return
        try:
            from dotenv import load_dotenv

            load_dotenv(target, override=False)
            logger.debug("Loaded dotenv from {}", target)
        except ImportError:
            logger.debug("python-dotenv not installed — skipping env file")

    def interpolate_env(self, value: str) -> str:
        """Replace ``${VAR_NAME}`` patterns with environment variable values.

        Missing variables are replaced with an empty string.

        Args:
            value: String potentially containing ``${VAR_NAME}`` references.

        Returns:
            The interpolated string.
        """
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )

    def get_agent_llm_config(self, agent_name: str) -> LLMStackEntry | None:
        """Return an agent-specific LLM override, if one exists.

        Args:
            agent_name: Name of the agent to look up.

        Returns:
            The :class:`LLMStackEntry` override, or ``None`` if the agent
            has no override in the active stack.
        """
        if self._active_stack is None:
            return None
        return self._active_stack.agent_overrides.get(agent_name)

    def resolve(self) -> StackConfig:
        """Walk all string fields and interpolate ``${ENV_VAR}`` references.

        Returns:
            A new :class:`StackConfig` with all environment variables resolved.

        Raises:
            ValueError: If no stack is loaded.
        """
        if self._active_stack is None:
            raise ValueError("No stack loaded — call load() first")

        data = self._active_stack.model_dump(mode="json")
        resolved = self._resolve_recursive(data)
        return StackConfig.model_validate(resolved)

    # -- Private helpers ----------------------------------------------------

    def _resolve_recursive(self, obj: Any) -> Any:
        """Recursively interpolate env vars in a nested data structure."""
        if isinstance(obj, str):
            return self.interpolate_env(obj)
        if isinstance(obj, dict):
            return {k: self._resolve_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_recursive(item) for item in obj]
        return obj

    @staticmethod
    def _builtin_fallback(stack_name: str) -> dict[str, Any]:
        """Return dict data for a built-in stack, or a minimal fallback."""
        from agentomatic.stacks.defaults import BUILTIN_STACKS

        if stack_name in BUILTIN_STACKS:
            stack = BUILTIN_STACKS[stack_name]()
            logger.info("Using built-in '{}' stack defaults", stack_name)
            return stack.model_dump(mode="json")

        logger.warning(
            "No built-in stack '{}' — returning minimal defaults",
            stack_name,
        )
        return {"name": stack_name}
