"""Regression tests for the release-time embedded Studio wheel check."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path


def _load_verifier():
    script = Path(__file__).parents[1] / "scripts" / "verify_wheel_studio.py"
    spec = importlib.util.spec_from_file_location("verify_wheel_studio", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel_with_studio(wheel: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(f"agentomatic/studio/static/{name}", content)


def test_wheel_studio_verifier_accepts_a_byte_identical_bundle(tmp_path: Path) -> None:
    verifier = _load_verifier()
    static = tmp_path / "static"
    assets = static / "assets"
    assets.mkdir(parents=True)
    (static / "index.html").write_bytes(b"<div id=app></div>")
    (assets / "index.js").write_bytes(b"console.log('studio')")
    wheel = tmp_path / "agentomatic.whl"
    _wheel_with_studio(
        wheel,
        {
            "index.html": (static / "index.html").read_bytes(),
            "assets/index.js": (assets / "index.js").read_bytes(),
        },
    )

    assert verifier.verify_wheel_studio(wheel, static) == []


def test_wheel_studio_verifier_reports_stale_or_changed_assets(tmp_path: Path) -> None:
    verifier = _load_verifier()
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_bytes(b"current")
    wheel = tmp_path / "agentomatic.whl"
    _wheel_with_studio(
        wheel,
        {"index.html": b"old", "assets/stale.js": b"stale"},
    )

    errors = verifier.verify_wheel_studio(wheel, static)

    assert any("differs from source" in error for error in errors)
    assert any("stale Studio files" in error for error in errors)
