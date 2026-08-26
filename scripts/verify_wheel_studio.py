#!/usr/bin/env python3
"""Fail a package build when its embedded Studio differs from the source bundle.

The Studio is built in a separate repository and copied into the Python package
before packaging.  Checking only that a wheel contains *some* files beneath
``agentomatic/studio/static`` is not enough: an old entrypoint can leave the
embedded UI blank or call backend routes that no longer exist.  This verifier
compares every bundled Studio file byte-for-byte with the wheel payload.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_ROOT = REPOSITORY_ROOT / "src" / "agentomatic" / "studio" / "static"
WHEEL_PREFIX = "agentomatic/studio/static/"


def verify_wheel_studio(wheel: Path, static_root: Path = DEFAULT_STATIC_ROOT) -> list[str]:
    """Return consistency errors between a built wheel and its Studio source files."""
    if not wheel.is_file():
        return [f"wheel does not exist: {wheel}"]
    if not static_root.is_dir():
        return [f"Studio static directory does not exist: {static_root}"]

    source_files = sorted(path for path in static_root.rglob("*") if path.is_file())
    if not source_files:
        return [f"Studio static directory is empty: {static_root}"]
    expected = {
        WHEEL_PREFIX + path.relative_to(static_root).as_posix(): path for path in source_files
    }

    try:
        with zipfile.ZipFile(wheel) as archive:
            actual = {name for name in archive.namelist() if name.startswith(WHEEL_PREFIX)}
            errors: list[str] = []
            missing = sorted(set(expected) - actual)
            unexpected = sorted(actual - set(expected))
            if missing:
                errors.append(f"wheel is missing Studio files: {', '.join(missing)}")
            if unexpected:
                errors.append(f"wheel contains stale Studio files: {', '.join(unexpected)}")
            for member, source in expected.items():
                if member in actual and archive.read(member) != source.read_bytes():
                    errors.append(f"wheel Studio asset differs from source: {member}")
            return errors
    except zipfile.BadZipFile:
        return [f"not a valid wheel archive: {wheel}"]


def main(argv: list[str] | None = None) -> int:
    """Run the wheel/Studio integrity verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Wheel to inspect, e.g. dist/agentomatic-*.whl")
    parser.add_argument(
        "--static-root",
        type=Path,
        default=DEFAULT_STATIC_ROOT,
        help="Studio static source directory (defaults to the package source)",
    )
    args = parser.parse_args(argv)

    errors = verify_wheel_studio(args.wheel, args.static_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    count = sum(1 for path in args.static_root.rglob("*") if path.is_file())
    print(f"Studio wheel integrity OK: {args.wheel.name} ({count} files, byte-identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
