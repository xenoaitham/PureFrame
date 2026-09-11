"""Tests for the version-sync helper that fans pyproject.toml's version
into the gui package.json, tauri.conf.json, and Cargo.toml."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# packaging/ is not a real package; import the module by path.
PACKAGING_DIR = Path(__file__).resolve().parent.parent / "packaging"
sys.path.insert(0, str(PACKAGING_DIR))
import sync_versions  # type: ignore[import-not-found]  # noqa: E402

# --- pure version translators -------------------------------------------------


@pytest.mark.parametrize(
    "pep440, npm",
    [
        ("0.1.0", "0.1.0"),
        ("0.1.0b14", "0.1.0-beta.14"),
        ("0.1.0b1", "0.1.0-beta.1"),
        ("0.2.3a4", "0.2.3-alpha.4"),
        ("1.0.0rc2", "1.0.0-rc.2"),
        ("2.4.7", "2.4.7"),
    ],
)
def test_npm_version_translation(pep440: str, npm: str) -> None:
    """PEP 440 prereleases (a/b/rc) translate to SemVer alpha/beta/rc dash-form."""
    assert sync_versions.npm_version(pep440) == npm


@pytest.mark.parametrize(
    "pep440, tauri",
    [
        ("0.1.0", "0.1.0"),
        ("0.1.0b14", "0.1.14"),
        ("0.1.0b1", "0.1.1"),
        ("0.1.0a5", "0.1.5"),
        ("0.1.0rc1", "0.1.1001"),
        ("2.4.7", "2.4.7"),
    ],
)
def test_tauri_version_translation(pep440: str, tauri: str) -> None:
    """Tauri/MSI demands strictly numeric Major.Minor.Build with rc reserved
    in the >1000 patch range so an rc never compares lower than a beta."""
    assert sync_versions.tauri_version(pep440) == tauri


def test_tauri_rc_outranks_beta_for_same_release() -> None:
    """rc must compare higher than any beta within the same release line.

    This is the property that justifies the +1000 offset reservation.
    """
    beta_max = sync_versions.tauri_version("0.1.0b999")
    rc_min = sync_versions.tauri_version("0.1.0rc1")
    # Compare numerically by tuple of integers to mirror MSI installer ordering.
    assert tuple(int(x) for x in rc_min.split(".")) > tuple(
        int(x) for x in beta_max.split(".")
    )


def test_unsupported_version_raises() -> None:
    """Dev/local/postN suffixes are rejected - releases must be canonical."""
    with pytest.raises(SystemExit):
        sync_versions.npm_version("0.1.0.dev1")


# --- side-effecting helpers ---------------------------------------------------


def test_update_json_version_rewrites_value(tmp_path: Path) -> None:
    p = tmp_path / "pkg.json"
    p.write_text(json.dumps({"name": "x", "version": "0.0.1"}) + "\n", encoding="utf-8")

    changed = sync_versions.update_json_version(p, "1.2.3")

    assert changed is True
    assert json.loads(p.read_text(encoding="utf-8"))["version"] == "1.2.3"


def test_update_json_version_noop_when_already_correct(tmp_path: Path) -> None:
    p = tmp_path / "pkg.json"
    p.write_text(
        json.dumps({"name": "x", "version": "1.2.3"}, indent=2) + "\n",
        encoding="utf-8",
    )
    before = p.read_text(encoding="utf-8")

    changed = sync_versions.update_json_version(p, "1.2.3")

    assert changed is False
    # Idempotent: file untouched byte-for-byte.
    assert p.read_text(encoding="utf-8") == before


def test_update_cargo_version_only_touches_package_version(tmp_path: Path) -> None:
    """Must not rewrite version specifiers inside [dependencies]."""
    p = tmp_path / "Cargo.toml"
    p.write_text(
        "[package]\n"
        'name = "x"\n'
        'version = "0.0.1"\n'
        "\n"
        "[dependencies]\n"
        'serde = { version = "1.0", features = ["derive"] }\n',
        encoding="utf-8",
    )

    sync_versions.update_cargo_version(p, "9.9.9")

    text = p.read_text(encoding="utf-8")
    assert '\nversion = "9.9.9"\n' in text
    # Dependency version pinning is untouched.
    assert 'serde = { version = "1.0"' in text


def test_update_cargo_version_errors_when_no_match(tmp_path: Path) -> None:
    p = tmp_path / "Cargo.toml"
    p.write_text('[package]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        sync_versions.update_cargo_version(p, "1.0.0")
