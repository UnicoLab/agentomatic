"""Shared test helpers."""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path


def install_plugin_package(
    plugins_dir: Path | str,
    package: str,
    source: str,
) -> contextlib.AbstractContextManager[None]:
    """Write a plugin package and make it importable, hermetically.

    The platform discovers plugins as ``<plugins_dir.name>.<package>.plugin``
    — see ``AgentPlatform.build``, where the prefix defaults to the plugins
    directory's own name — resolved through ``sys.path``. A test that merely
    writes the files therefore depends on ambient interpreter state, and
    passes or fails depending on which tests ran before it.

    This puts the *parent* of ``plugins_dir`` on ``sys.path`` for the duration,
    invalidates importlib's cached directory listings so the freshly written
    files are visible, and evicts the modules again afterwards.

    Args:
        plugins_dir: The platform's plugins directory.
        package: Package name to create inside it.
        source: Contents of the package's ``plugin.py``.

    Returns:
        A context manager that makes the package importable while active.
    """
    plugins_dir = Path(plugins_dir)
    prefix = plugins_dir.name
    root = plugins_dir.parent

    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / "__init__.py").write_text("", encoding="utf-8")
    target = plugins_dir / package
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_text("", encoding="utf-8")
    (target / "plugin.py").write_text(source, encoding="utf-8")

    def _evict() -> None:
        for name in [m for m in sys.modules if m == prefix or m.startswith(f"{prefix}.")]:
            del sys.modules[name]

    @contextlib.contextmanager
    def _importable() -> Iterator[None]:
        sys.path.insert(0, str(root))
        _evict()
        # The files were created after interpreter start, so importlib's cached
        # directory listings would otherwise not see them.
        importlib.invalidate_caches()
        try:
            yield
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(root))
            _evict()

    return _importable()
