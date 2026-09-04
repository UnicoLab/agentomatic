"""Production contracts for safe automatic plugin artifact reloads."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agentomatic import AgentPlatform
from agentomatic.artifacts import ArtifactRegistry
from agentomatic.config.settings import PlatformSettings
from agentomatic.plugins import BaseMLPlugin, PluginAutoReloader, PluginRegistry


class _Input(BaseModel):
    value: str


class _Output(BaseModel):
    artifact_version: str


class ArtifactBackedPlugin(BaseMLPlugin[_Input, _Output]):
    """Test plugin that atomically assigns a model after a successful load."""

    plugin_name = "artifact_backed"

    def __init__(self) -> None:
        super().__init__()
        self.model_version: str | None = None
        self.load_count = 0

    async def load_model(self) -> None:
        artifact_dir = self.artifact_dir()
        version = artifact_dir.name if artifact_dir is not None else None
        if version == "broken":
            self.model_version = "partially-loaded"
            raise RuntimeError("candidate model is invalid")
        self.model_version = version
        self.load_count += 1
        await super().load_model()

    async def predict(self, inputs: _Input) -> _Output:
        assert self.model_version is not None
        return _Output(artifact_version=self.model_version)


def _promote(registry: ArtifactRegistry, version: str) -> None:
    registry.candidate_dir(version)
    registry.register_candidate(version)
    registry.promote(version)


@pytest.mark.asyncio
async def test_autoreloader_updates_promoted_version_and_preserves_last_good_model(
    tmp_path, monkeypatch
) -> None:
    """A bad promotion is tried once without taking healthy plugin traffic down."""
    monkeypatch.setenv("AGENTOMATIC_ARTIFACT_ROOT", str(tmp_path))
    artifacts = ArtifactRegistry(tmp_path)
    _promote(artifacts, "v1")

    plugin = ArtifactBackedPlugin()
    await plugin.load_model()
    registry = PluginRegistry()
    registry._plugins[plugin.plugin_name] = plugin  # noqa: SLF001 - controlled fixture
    watcher = PluginAutoReloader(registry, artifact_root=tmp_path, interval=60)
    await watcher.start()
    try:
        _promote(artifacts, "v2")
        assert await watcher.check_once() is True
        assert plugin.model_version == "v2"
        assert plugin.load_count == 2
        assert (await plugin.invoke(_Input(value="ok"))).artifact_version == "v2"

        _promote(artifacts, "broken")
        assert await watcher.check_once() is True
        assert watcher.last_result == {
            "reloaded": 0,
            "failed": 1,
            "results": [],
            "errors": [{"name": "artifact_backed", "error": "candidate model is invalid"}],
        }
        assert plugin.is_loaded is True
        assert plugin.model_version == "v2"
        assert (await plugin.invoke(_Input(value="still-serving"))).artifact_version == "v2"
        assert await watcher.check_once() is False
    finally:
        await watcher.stop()


def test_platform_watcher_reloads_real_plugin_routes_after_artifact_promotion(
    tmp_path, monkeypatch
) -> None:
    """The production lifespan watcher updates the live REST route without a restart."""
    monkeypatch.setenv("AGENTOMATIC_ARTIFACT_ROOT", str(tmp_path))
    artifacts = ArtifactRegistry(tmp_path)
    _promote(artifacts, "v1")
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        plugin_autoreload=True,
        plugin_autoreload_interval=0.01,
    )
    plugin = ArtifactBackedPlugin()
    platform._plugin_registry._plugins[plugin.plugin_name] = plugin  # noqa: SLF001

    with TestClient(platform.build()) as client:
        assert client.post(
            "/api/v1/plugins/artifact_backed/predict", json={"value": "before"}
        ).json() == {"artifact_version": "v1"}
        _promote(artifacts, "v2")

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            response = client.post(
                "/api/v1/plugins/artifact_backed/predict", json={"value": "after"}
            )
            if response.json() == {"artifact_version": "v2"}:
                break
            time.sleep(0.02)
        else:
            pytest.fail("artifact promotion did not auto-reload the live plugin route")

    assert platform._plugin_autoreloader is None


def test_autoreload_environment_settings_are_applied_to_platform(tmp_path, monkeypatch) -> None:
    """The documented environment flags work without custom application code."""
    monkeypatch.setenv("AGENTOMATIC_PLUGIN_AUTORELOAD", "true")
    monkeypatch.setenv("AGENTOMATIC_PLUGIN_AUTORELOAD_INTERVAL", "2.5")
    settings = PlatformSettings(_env_file=None)
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        settings=settings,
    )

    assert platform._plugin_autoreload is True
    assert platform._plugin_autoreload_interval == 2.5


def test_platform_settings_artifact_root_drives_plugin_and_watcher(tmp_path) -> None:
    """An explicit settings object must not split watcher and plugin artifact roots."""
    artifact_root = tmp_path / "configured-artifacts"
    artifacts = ArtifactRegistry(artifact_root)
    _promote(artifacts, "v1")
    settings = PlatformSettings(
        _env_file=None,
        artifact_root=artifact_root,
        plugin_autoreload=True,
        plugin_autoreload_interval=0.01,
    )
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        settings=settings,
    )
    plugin = ArtifactBackedPlugin()
    platform._plugin_registry._plugins[plugin.plugin_name] = plugin  # noqa: SLF001

    with TestClient(platform.build()) as client:
        assert plugin.artifact_dir() == artifact_root / "v1"
        assert client.post(
            "/api/v1/plugins/artifact_backed/predict", json={"value": "before"}
        ).json() == {"artifact_version": "v1"}

        _promote(artifacts, "v2")
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            response = client.post(
                "/api/v1/plugins/artifact_backed/predict", json={"value": "after"}
            )
            if response.json() == {"artifact_version": "v2"}:
                break
            time.sleep(0.02)
        else:
            pytest.fail("settings-backed artifact promotion was not observed")


@pytest.mark.asyncio
async def test_cancelled_reload_restores_previous_plugin_state() -> None:
    """Cancellation must be as transactional as an ordinary loading failure."""
    plugin = ArtifactBackedPlugin()
    plugin.model_version = "v1"
    plugin.load_count = 1
    plugin.mark_loaded()
    load_started = asyncio.Event()

    async def partial_load() -> None:
        plugin.model_version = "partially-loaded"
        load_started.set()
        await asyncio.Event().wait()

    plugin.load_model = partial_load  # type: ignore[method-assign]
    reload_task = asyncio.create_task(plugin.reload_model())
    await load_started.wait()
    reload_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await reload_task

    assert plugin.model_version == "v1"
    assert plugin.load_count == 1
    assert plugin.is_loaded is True


@pytest.mark.asyncio
async def test_invoke_supports_legacy_plugin_init_without_super() -> None:
    """The new synchronization wrapper must preserve old plugin subclasses."""

    class LegacyInitPlugin(ArtifactBackedPlugin):
        def __init__(self) -> None:
            self.model_version = "legacy"

    plugin = LegacyInitPlugin()
    result = await plugin.invoke(_Input(value="ok"))

    assert result.artifact_version == "legacy"
