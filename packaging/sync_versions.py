"""Sync version metadata across pyproject.toml, gui/package.json,
gui/src-tauri/tauri.conf.json, gui/src-tauri/Cargo.toml, and the
pureframe-desktop entry in gui/src-tauri/Cargo.lock.

Source of truth: pyproject.toml [project] version.

PEP 440 forms supported:
    0.1.0b14         (beta)
    0.1.0rc1         (release candidate)
    0.1.0            (stable)

Translates to:
    npm SemVer (gui/package.json):       0.1.0-beta.14 / 0.1.0-rc.1 / 0.1.0
    Cargo SemVer (Cargo.toml + lock):    0.1.0-beta.14 / 0.1.0-rc.1 / 0.1.0
    Tauri 3-part numeric (tauri.conf):   0.1.14 / 0.1.1001 / 0.1.0
    Linux package version (rpm/deb):     0.1.0~beta14 / 0.1.0~rc1 / 0.1.0

Usage:
    python packaging/sync_versions.py [--check]

With --check: exit nonzero if any file is out of sync (CI guard).
Without flags: rewrite the downstream files in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
GUI_PKG = ROOT / "gui" / "package.json"
TAURI_CONF = ROOT / "gui" / "src-tauri" / "tauri.conf.json"
CARGO = ROOT / "gui" / "src-tauri" / "Cargo.toml"
CARGO_LOCK = ROOT / "gui" / "src-tauri" / "Cargo.lock"
CARGO_LOCK_PACKAGE = "pureframe-desktop"


_PEP440_RE = re.compile(
    r"^(?P<release>\d+\.\d+\.\d+)"
    r"(?:(?P<pre_kind>a|b|rc)(?P<pre_num>\d+))?$"
)


def parse_pep440(v: str) -> tuple[tuple[int, int, int], str | None, int | None]:
    """Split a PEP 440 version into (release, pre_kind, pre_num)."""
    m = _PEP440_RE.match(v)
    if not m:
        raise SystemExit(f"unsupported pyproject version '{v}'")
    parts = tuple(int(x) for x in m.group("release").split("."))
    if len(parts) != 3:
        raise SystemExit(f"need 3-part release, got '{v}'")
    return (
        parts,
        m.group("pre_kind"),
        int(m.group("pre_num")) if m.group("pre_num") else None,
    )  # type: ignore[return-value]


def npm_version(v: str) -> str:
    release, kind, num = parse_pep440(v)
    base = ".".join(str(x) for x in release)
    if kind is None:
        return base
    label = {"a": "alpha", "b": "beta", "rc": "rc"}[kind]
    return f"{base}-{label}.{num}"


def tauri_version(v: str) -> str:
    """Tauri/MSI requires strictly numeric 3-part Major.Minor.Build.

    We encode the prerelease number in the patch field so beta builds bump.
    """
    (major, minor, patch), kind, num = parse_pep440(v)
    if kind is None:
        return f"{major}.{minor}.{patch}"
    # Reserve patch ranges per prerelease kind:
    #   alpha:  patch + 1..99
    #   beta:   patch + 1..999
    #   rc:     patch + 1001..1999
    offset = {"a": 0, "b": 0, "rc": 1000}[kind]
    return f"{major}.{minor}.{patch + offset + num}"


def read_pyproject_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["version"]


def update_json_version(path: Path, new: str) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") == new:
        return False
    data["version"] = new
    # Preserve trailing newline.
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def update_cargo_version(path: Path, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{new}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"failed to rewrite version in {path}")
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def _cargo_lock_version(path: Path, package: str) -> str | None:
    """Version of *package* in a Cargo.lock file, or None if absent."""
    in_package = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            in_package = False
        elif stripped == f'name = "{package}"':
            in_package = True
        elif in_package and stripped.startswith("version = "):
            parts = stripped.split('"')
            return parts[1] if len(parts) > 1 else None
    return None


def update_cargo_lock(path: Path, package: str, new: str) -> bool:
    """Rewrite *package*'s version inside Cargo.lock.

    cargo regenerates the lock on build, but the release version-guard runs
    against the committed file, so the entry is kept in step here.
    """
    current = _cargo_lock_version(path, package)
    if current is None or current == new:
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_package = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[[package]]":
            in_package = False
        elif stripped == f'name = "{package}"':
            in_package = True
        elif in_package and stripped.startswith("version = "):
            indent = line[: len(line) - len(line.lstrip())]
            ending = line[len(line.rstrip("\r\n")) :]
            lines[i] = f'{indent}version = "{new}"{ending}'
            path.write_text("".join(lines), encoding="utf-8")
            return True
    return False


def check_versions() -> int:
    """Exit 0 if all downstream files match; 1 otherwise."""
    src = read_pyproject_version()
    expected_npm = npm_version(src)
    expected_tauri = tauri_version(src)

    drift: list[str] = []

    pkg = json.loads(GUI_PKG.read_text(encoding="utf-8"))
    if pkg.get("version") != expected_npm:
        drift.append(f"gui/package.json: {pkg.get('version')} != {expected_npm}")

    tauri = json.loads(TAURI_CONF.read_text(encoding="utf-8"))
    if tauri.get("version") != expected_tauri:
        drift.append(
            f"gui/src-tauri/tauri.conf.json: {tauri.get('version')} != {expected_tauri}"
        )

    cargo_text = CARGO.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo_text, flags=re.MULTILINE)
    cargo_version = m.group(1) if m else None
    if cargo_version != expected_npm:
        drift.append(f"gui/src-tauri/Cargo.toml: {cargo_version} != {expected_npm}")

    lock_version = _cargo_lock_version(CARGO_LOCK, CARGO_LOCK_PACKAGE)
    if lock_version is not None and lock_version != expected_npm:
        drift.append(
            f"gui/src-tauri/Cargo.lock ({CARGO_LOCK_PACKAGE}): "
            f"{lock_version} != {expected_npm}"
        )

    if drift:
        print("Version drift detected (source: pyproject.toml):", file=sys.stderr)
        for line in drift:
            print(f"  - {line}", file=sys.stderr)
        print(
            "Run `python packaging/sync_versions.py` to fix.",
            file=sys.stderr,
        )
        return 1
    print(f"All versions in sync with pyproject ({src}).")
    return 0


def sync_versions() -> None:
    src = read_pyproject_version()
    npm = npm_version(src)
    tauri = tauri_version(src)

    changed = False
    if update_json_version(GUI_PKG, npm):
        print(f"gui/package.json -> {npm}")
        changed = True
    if update_json_version(TAURI_CONF, tauri):
        print(f"gui/src-tauri/tauri.conf.json -> {tauri}")
        changed = True
    if update_cargo_version(CARGO, npm):
        print(f"gui/src-tauri/Cargo.toml -> {npm}")
        changed = True
    if update_cargo_lock(CARGO_LOCK, CARGO_LOCK_PACKAGE, npm):
        print(f"gui/src-tauri/Cargo.lock ({CARGO_LOCK_PACKAGE}) -> {npm}")
        changed = True
    if not changed:
        print(f"All versions already in sync ({src}).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if any downstream file is out of sync",
    )
    args = parser.parse_args()
    if args.check:
        return check_versions()
    sync_versions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
