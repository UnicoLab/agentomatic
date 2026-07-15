"""Tests for ``AgentPlatform.run`` reload / multi-worker factory wiring (P1-1).

Modern uvicorn ``sys.exit(1)`` when handed an app instance together with
``reload=True`` or ``workers>1``; it needs an import string. These tests verify
that:

- a folder-based platform serialises a reconstructable config and hands uvicorn
  the ``agentomatic._runtime:create_app`` factory import string;
- a programmatically-configured platform degrades to a single in-process
  instance (app instance, no reload/workers) instead of exiting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from agentomatic import AgentManifest, AgentPlatform
from agentomatic._runtime import FACTORY_CONFIG_ENV, create_app
from agentomatic.storage.memory import MemoryStore


def _folder_platform(tmp_path: Path) -> AgentPlatform:
    """Return a pure folder-based platform (no programmatic state)."""
    agents = tmp_path / "agents"
    agents.mkdir()
    return AgentPlatform(
        agents_dir=agents,
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        ingestion_dir=tmp_path / "ingestion",
        title="Reload Test",
    )


def test_reload_uses_factory_import_string(tmp_path: Path) -> None:
    """reload=True must call uvicorn with a factory import string, not an app."""
    platform = _folder_platform(tmp_path)
    with patch("uvicorn.run") as mock_run:
        platform.run(reload=True)
    assert mock_run.call_count == 1
    args, kwargs = mock_run.call_args
    assert args[0] == "agentomatic._runtime:create_app"
    assert kwargs["factory"] is True
    assert kwargs["reload"] is True
    import os

    assert FACTORY_CONFIG_ENV in os.environ


def test_workers_gt_one_uses_factory_import_string(tmp_path: Path) -> None:
    """workers>1 must also use the factory import string."""
    platform = _folder_platform(tmp_path)
    with patch("uvicorn.run") as mock_run:
        platform.run(workers=4)
    args, kwargs = mock_run.call_args
    assert args[0] == "agentomatic._runtime:create_app"
    assert kwargs["workers"] == 4


def test_no_reload_passes_app_instance(tmp_path: Path) -> None:
    """Default (no reload / single worker) keeps the fast in-process app path."""
    platform = _folder_platform(tmp_path)
    with patch("uvicorn.run") as mock_run:
        platform.run()
    args, kwargs = mock_run.call_args
    # First positional arg is a built FastAPI app, not an import string.
    assert not isinstance(args[0], str)
    assert "factory" not in kwargs


def test_programmatic_platform_degrades_gracefully(tmp_path: Path) -> None:
    """A platform with a custom store cannot use the factory → single instance."""
    platform = _folder_platform(tmp_path)
    platform.store = MemoryStore()

    async def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": "ok"}

    platform.register_agent(
        manifest=AgentManifest(name="x", slug="x"),
        node_fn=echo,
    )
    with patch("uvicorn.run") as mock_run:
        # Must NOT raise / exit — degrades to an app instance without reload.
        platform.run(reload=True, workers=3)
    args, kwargs = mock_run.call_args
    assert not isinstance(args[0], str)
    assert "factory" not in kwargs


def test_create_app_requires_config_env(monkeypatch: Any) -> None:
    """create_app() raises a clear error when the config env is absent."""
    monkeypatch.delenv(FACTORY_CONFIG_ENV, raising=False)
    try:
        create_app()
    except RuntimeError as exc:
        assert FACTORY_CONFIG_ENV in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("create_app should raise without config env")


def test_factory_config_roundtrip_builds_app(tmp_path: Path, monkeypatch: Any) -> None:
    """The serialised config must let create_app() rebuild a working app."""
    import json

    platform = _folder_platform(tmp_path)
    config = platform._factory_config()
    assert config is not None
    monkeypatch.setenv(FACTORY_CONFIG_ENV, json.dumps(config))
    app = create_app()
    assert app.title == "Reload Test"
