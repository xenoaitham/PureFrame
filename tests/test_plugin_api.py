"""Contract tests for the plugin API (``pureframe.plugin_api``).

The fixture detector lives in this module so ``EntryPoint.load()`` can
import it by name (``tests.test_plugin_api:FixtureDetector``) without
installing anything - the same resolution path a real entry point takes.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

import numpy as np

from pureframe.hardware import ProfileSettings
from pureframe.pipeline.detect.nudity import Detection
from pureframe.plugin_api import (
    DEFAULT_LABEL_THRESHOLD,
    PluginRegistration,
    discover,
    validate_plugin,
)


class FixtureDetector:
    """Minimal contract-honoring detector used as the registry fixture."""

    label_categories = {
        "motion_blob": "WEAPON_VISIBLE",
        "second_blob": "WEAPON_VISIBLE",
    }
    label_thresholds = {"motion_blob": 0.6}

    def __init__(self, settings: ProfileSettings):
        self.settings = settings
        self.calls = 0

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        self.calls += 1
        h, w = frames_bgr[0].shape[:2] if frames_bgr else (0, 0)
        box = (w // 4, h // 4, w // 2, h // 2)
        return [
            [Detection(label="motion_blob", score=0.9, box=box)] for _ in frames_bgr
        ]

    def unload(self) -> None:
        self.settings = None


class NotADetector:
    def __init__(
        self, settings, extra_required
    ):  # pragma: no cover - signature check only
        pass


def _entry_point(name: str, target: str) -> EntryPoint:
    return EntryPoint(name, target, "pureframe.plugins")


def _patched_discover(monkeypatch, entry_points_list):
    from pureframe import plugin_api

    monkeypatch.setattr(
        plugin_api,
        "entry_points",
        lambda group=None: entry_points_list,
    )
    return discover()


class TestValidatePlugin:
    def test_fixture_detector_satisfies_contract(self):
        assert validate_plugin(FixtureDetector) == []

    def test_missing_methods_reported(self):
        class Bare:
            pass

        problems = validate_plugin(Bare)
        assert any("detect_batch" in p for p in problems)
        assert any("unload" in p for p in problems)

    def test_non_class_reported(self):
        problems = validate_plugin(lambda settings: None)
        assert problems == ["entry point resolved to <lambda>, not a class"] or any(
            "not a class" in p for p in problems
        )

    def test_multi_argument_init_reported(self):
        problems = validate_plugin(NotADetector)
        assert any("__init__" in p for p in problems)


class TestDiscover:
    def test_fixture_plugin_registers(self, monkeypatch):
        registry = _patched_discover(
            monkeypatch,
            [_entry_point("fixture", "tests.test_plugin_api:FixtureDetector")],
        )
        assert set(registry) == {"fixture"}
        reg = registry["fixture"]
        assert reg.cls is FixtureDetector
        assert reg.label_categories == {
            "motion_blob": "WEAPON_VISIBLE",
            "second_blob": "WEAPON_VISIBLE",
        }
        assert reg.label_thresholds == {"motion_blob": 0.6}
        assert reg.category_names() == {"WEAPON_VISIBLE"}

    def test_no_plugins_means_empty_registry(self, monkeypatch):
        assert _patched_discover(monkeypatch, []) == {}

    def test_unimportable_plugin_skipped(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            registry = _patched_discover(
                monkeypatch,
                [_entry_point("broken", "tests.test_plugin_api:DoesNotExist")],
            )
        assert registry == {}
        assert any("Skipping PureFrame plugin" in r.message for r in caplog.records)

    def test_contract_violation_skipped(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            registry = _patched_discover(
                monkeypatch,
                [_entry_point("bad", "tests.test_plugin_api:NotADetector")],
            )
        assert registry == {}
        assert any("breaks the contract" in r.message for r in caplog.records)

    def test_bad_label_map_skipped(self, monkeypatch):
        class BadMap(FixtureDetector):
            label_categories = {"motion_blob": ""}

        registry = _patched_discover(
            monkeypatch, [_entry_point("badmap", "tests.test_plugin_api:BadMap")]
        )
        assert registry == {}

    def test_non_numeric_threshold_skipped(self, monkeypatch):
        class BadThreshold(FixtureDetector):
            label_thresholds = {"motion_blob": "high"}

        registry = _patched_discover(
            monkeypatch,
            [_entry_point("badthr", "tests.test_plugin_api:BadThreshold")],
        )
        assert registry == {}


class TestPluginRegistration:
    def test_threshold_for_uses_default_for_unknown_labels(self):
        reg = PluginRegistration(
            name="x", cls=FixtureDetector, label_categories={"a": "CAT"}
        )
        assert reg.threshold_for("a") == DEFAULT_LABEL_THRESHOLD

    def test_threshold_for_clamps(self):
        reg = PluginRegistration(
            name="x",
            cls=FixtureDetector,
            label_categories={"a": "CAT"},
            label_thresholds={"a": 1.7, "b": -0.2},
        )
        assert reg.threshold_for("a") == 1.0
        assert reg.threshold_for("b") == 0.0


class TestFixtureDetectorBehavior:
    def test_detect_batch_returns_detection_per_frame(self):
        detector = FixtureDetector(settings=None)
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
        results = detector.detect_batch(frames)
        assert len(results) == 3
        assert all(len(dets) == 1 for dets in results)
        assert results[0][0].label == "motion_blob"
        assert results[0][0].box == (4, 4, 8, 8)
        assert detector.calls == 1

    def test_real_entry_point_group_name(self):
        from pureframe.plugin_api import ENTRY_POINT_GROUP

        assert ENTRY_POINT_GROUP == "pureframe.plugins"


def test_discover_reads_the_real_entry_point_group(monkeypatch):
    """discover() must query the ``pureframe.plugins`` group of the real
    metadata registry (empty in this environment - no plugin installed)."""
    from pureframe import plugin_api

    seen = {}

    def fake_entry_points(*, group=None):
        seen["group"] = group
        return []

    monkeypatch.setattr(plugin_api, "entry_points", fake_entry_points)
    assert discover() == {}
    assert seen["group"] == "pureframe.plugins"


def test_entry_point_to_a_class_registers():
    ep = _entry_point("shape", "tests.test_plugin_api:FixtureDetector")
    assert ep.load() is FixtureDetector


def test_entry_point_to_a_non_class_is_skipped(monkeypatch):
    registry = _patched_discover(
        monkeypatch,
        [
            _entry_point(
                "shape", "tests.test_plugin_api:FixtureDetector.label_categories"
            )
        ],
    )
    assert registry == {}
