"""Built-in default stack configurations for local and remote environments."""

from __future__ import annotations

from collections.abc import Callable

import yaml

from agentomatic.stacks.manager import (
    AuthStackConfig,
    DatabaseStackEntry,
    EmbeddingStackEntry,
    FeaturesStackEntry,
    LLMStackEntry,
    StackConfig,
)


def get_default_local_stack() -> StackConfig:
    """Return a StackConfig for local development.

    Uses Ollama for LLM, SQLite for storage, and no authentication.

    Returns:
        A StackConfig pre-configured for local development.
    """
    return StackConfig(
        name="local",
        description="Local development stack with Ollama and SQLite",
        llm={
            "default": LLMStackEntry(
                provider="ollama",
                model="mistral:7b",
                temperature=0.1,
                max_tokens=8192,
                base_url="http://localhost:11434",
            ),
            "fast": LLMStackEntry(
                provider="ollama",
                model="mistral:7b",
                temperature=0.0,
                max_tokens=8192,
                base_url="http://localhost:11434",
            ),
        },
        embedding=EmbeddingStackEntry(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768,
        ),
        database=DatabaseStackEntry(
            url="sqlite+aiosqlite:///data/platform.db",
            pool_size=5,
            max_overflow=10,
        ),
        features=FeaturesStackEntry(
            enable_streaming=True,
            enable_a2a=True,
            enable_metrics=False,
            enable_rate_limit=False,
            enable_auth=False,
            enable_db=False,
        ),
        auth=AuthStackConfig(method="api_key"),
    )


def get_default_remote_stack() -> StackConfig:
    """Return a StackConfig for cloud / remote deployment.

    Uses OpenAI with ``${OPENAI_API_KEY}``, PostgreSQL with
    ``${DATABASE_URL}``, and JWT authentication.

    Returns:
        A StackConfig pre-configured for cloud deployment.
    """
    return StackConfig(
        name="remote",
        description="Cloud deployment stack with OpenAI, PostgreSQL, and JWT auth",
        llm={
            "default": LLMStackEntry(
                provider="openai",
                model="gpt-4o",
                temperature=0.1,
                max_tokens=8192,
                api_key="${OPENAI_API_KEY}",
            ),
            "fast": LLMStackEntry(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=8192,
                api_key="${OPENAI_API_KEY}",
            ),
            "judge": LLMStackEntry(
                provider="openai",
                model="gpt-4o",
                temperature=0.0,
                max_tokens=8192,
                api_key="${OPENAI_API_KEY}",
            ),
        },
        embedding=EmbeddingStackEntry(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        ),
        database=DatabaseStackEntry(
            url="${DATABASE_URL}",
            pool_size=10,
            max_overflow=20,
        ),
        features=FeaturesStackEntry(
            enable_streaming=True,
            enable_a2a=True,
            enable_metrics=True,
            enable_rate_limit=True,
            enable_auth=True,
            enable_db=True,
        ),
        auth=AuthStackConfig(
            method="jwt",
            jwks_url="${JWKS_URL}",
            issuer="${JWT_ISSUER}",
            audience="${JWT_AUDIENCE}",
        ),
    )


def get_default_stack_yaml(stack_name: str) -> str:
    """Return the YAML string representation of a built-in stack.

    Args:
        stack_name: The name of the built-in stack (``"local"`` or ``"remote"``).

    Returns:
        A YAML-formatted string for writing to disk.

    Raises:
        ValueError: If the stack name is not a recognised built-in.
    """
    if stack_name not in BUILTIN_STACKS:
        raise ValueError(
            f"Unknown built-in stack '{stack_name}'. Available: {', '.join(BUILTIN_STACKS)}"
        )
    stack = BUILTIN_STACKS[stack_name]()
    data = stack.model_dump(mode="json")
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


BUILTIN_STACKS: dict[str, Callable[[], StackConfig]] = {
    "local": get_default_local_stack,
    "remote": get_default_remote_stack,
}
