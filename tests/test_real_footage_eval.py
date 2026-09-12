"""Real-footage evaluation harness (pureframe.eval.real_footage).

The interval math is checked in isolation; one integration test plans
the three-shot fixture through the real CLI (with the nudity detector
mocked silent and the fixture weapon plugin enabled, as elsewhere in
the suite) and scores it against hand marks. The clip: dark grey
0-4 s | testsrc2 4-6 s (marker 4.5-5.5 s) | light grey 6-10 s.
"""

from __future__ import annotations

import json

import pytest

from pureframe.eval.real_footage import (
    FlaggedRange,
    MarkedRange,
    _overlap_seconds,
    _subtract,
    _union,
    format_report,
    load_marks,
    run_real_eval,
    score_ranges,
)
from pureframe.pipeline.detect.nudity import Detection
from pureframe.plugin_api import PluginRegistration
from tests.conftest import frame_has_marker

WEAPON = "WEAPON_VISIBLE"
BOX = (60, 40, 260, 200)


class MarkerDetector:
    """Fixture plugin: flags the marker frame with a weapon box."""

    label_categories = {"pistol": WEAPON}
    label_thresholds = {"pistol": 0.6}

    def __init__(self, settings):
        self.settings = settings

    def detect_batch(self, frames_bgr):
        return [
            [Detection(label="pistol", score=0.9, box=BOX)]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    def unload(self):
        pass


def _registration() -> PluginRegistration:
    return PluginRegistration(
        name="weapons",
        cls=MarkerDetector,
        label_categories={"pistol": WEAPON},
        label_thresholds={"pistol": 0.6},
    )


def _patched_registry(monkeypatch, registration):
    from pureframe import plugin_api

    monkeypatch.setattr(
        plugin_api, "discover", lambda: {registration.name: registration}
    )


class TestIntervalMath:
    def test_union_merges_overlaps_and_adjacent(self):
        assert _union([]) == []
        assert _union([(0.0, 1.0)]) == [(0.0, 1.0)]
        assert _union([(0.0, 2.0), (1.0, 3.0)]) == [(0.0, 3.0)]
        assert _union([(0.0, 1.0), (1.0, 2.0)]) == [(0.0, 2.0)]
        assert _union([(4.0, 5.0), (0.0, 1.0)]) == [(0.0, 1.0), (4.0, 5.0)]

    def test_overlap(self):
        assert _overlap_seconds([(0.0, 2.0)], [(1.0, 3.0)]) == pytest.approx(1.0)
        assert _overlap_seconds([(0.0, 1.0)], [(2.0, 3.0)]) == 0.0
        assert _overlap_seconds([(0.0, 4.0)], [(1.0, 2.0), (3.0, 4.0)]) == (
            pytest.approx(2.0)
        )

    def test_subtract(self):
        assert _subtract([(0.0, 4.0)], [(1.0, 2.0)]) == [(0.0, 1.0), (2.0, 4.0)]
        assert _subtract([(0.0, 4.0)], [(0.0, 4.0)]) == []
        assert _subtract([(0.0, 1.0)], [(2.0, 3.0)]) == [(0.0, 1.0)]


class TestLoadMarks:
    def test_valid_marks_with_and_without_category(self, tmp_path):
        path = tmp_path / "marks.json"
        path.write_text(
            json.dumps(
                {
                    "ranges": [
                        {"start": 1.0, "end": 2.5, "category": "NUDITY_EXPLICIT"},
                        {"start": 4.0, "end": 5.0},
                    ]
                }
            ),
            encoding="utf-8",
        )
        marks = load_marks(path)
        assert marks[0] == MarkedRange(1.0, 2.5, "NUDITY_EXPLICIT")
        assert marks[1] == MarkedRange(4.0, 5.0, "ANY")

    def test_rejects_bad_shapes(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"ranges": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_marks(empty)

        inverted = tmp_path / "inverted.json"
        inverted.write_text(
            json.dumps({"ranges": [{"start": 5.0, "end": 1.0}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_marks(inverted)


class TestScoreRanges:
    def test_perfect_match(self):
        marks = [MarkedRange(4.0, 6.0, "NUDITY_EXPLICIT")]
        flagged = [FlaggedRange(4.0, 6.0, "NUDITY_EXPLICIT")]
        (score,) = score_ranges(marks, flagged, tolerance=0.0)
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0

    def test_partial_recall_on_half_marked_shot(self):
        marks = [MarkedRange(4.0, 5.0, "ANY")]
        flagged = [FlaggedRange(4.0, 6.0, "NUDITY_EXPLICIT")]
        scores = score_ranges(marks, flagged, tolerance=0.0)
        # The ANY mark folds into the concrete category; no empty ANY row.
        assert {s.category for s in scores} == {"NUDITY_EXPLICIT"}
        (score,) = scores
        assert score.recall == pytest.approx(1.0)
        assert score.precision == pytest.approx(0.5)

    def test_tolerance_absorbs_boundary_snap(self):
        marks = [MarkedRange(4.2, 5.8, "ANY")]
        flagged = [FlaggedRange(4.0, 6.0, "KISS_INTENSE")]
        (score,) = score_ranges(marks, flagged, tolerance=0.5)
        assert score.precision == pytest.approx(1.0)
        assert score.recall == pytest.approx(1.0)
        # Reported expected time stays the marks as written.
        assert score.expected_seconds == pytest.approx(1.6)

    def test_missed_and_extra_ranges_are_reported(self):
        marks = [MarkedRange(4.0, 6.0, "ANY")]
        flagged = [FlaggedRange(4.0, 5.0, "ANY"), FlaggedRange(8.0, 9.0, "ANY")]
        (score,) = score_ranges(marks, flagged, tolerance=0.0)
        assert score.missed == [(5.0, 6.0)]
        assert score.extra == [(8.0, 9.0)]
        assert score.recall == pytest.approx(0.5)


class TestEndToEndOnFixture:
    @pytest.fixture
    def plugin_plan_ranges(self, three_shot_video, tmp_path, monkeypatch):
        _patched_registry(monkeypatch, _registration())
        import pureframe.cli
        from pureframe.pipeline.detect.nudity import NudityDetector

        monkeypatch.setattr(
            NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
        )
        original = pureframe.cli.get_settings

        def dense(profile, **kwargs):
            s = original(profile)
            s.sample_keyframes_per_shot = 10
            return s

        monkeypatch.setattr(pureframe.cli, "get_settings", dense)

        # Point the evaluator at the fixture plugin through the same
        # pass-through channel a user would use.
        extra = ["--enable-plugin", "weapons", "--no-clip", "--no-audio"]
        marks_path = tmp_path / "marks.json"
        marks_path.write_text(
            json.dumps(
                {"ranges": [{"start": 4.0, "end": 6.0, "category": "WEAPON_VISIBLE"}]}
            ),
            encoding="utf-8",
        )
        scores, summary = run_real_eval(
            three_shot_video,
            marks_path,
            tolerance=0.5,
            profile="CPU",
            extra_args=extra,
            jsonl_path=tmp_path / "results.jsonl",
        )
        return scores, summary, tmp_path / "results.jsonl"

    def test_scores_and_jsonl_record(
        self, plugin_plan_ranges, monkeypatch, three_shot_video
    ):
        scores, summary, jsonl_path = plugin_plan_ranges
        weapon = next(s for s in scores if s.category == "WEAPON_VISIBLE")
        # The middle shot (4-6 s) is flagged by the plugin and exactly
        # marked: a clean hit, and the outer shots stay unflagged.
        assert weapon.precision == pytest.approx(1.0)
        assert weapon.recall == pytest.approx(1.0)
        assert weapon.expected_seconds == pytest.approx(2.0)

        # Privacy-safe record: the clip is a hash, never a path.
        record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["kind"] == "real-eval"
        assert "three_shot" not in json.dumps(record)
        assert record["file"]["sha256"]
        assert record["marks_sha256"]
        assert "WEAPON_VISIBLE" in record["per_category"]

    def test_report_is_human_readable(self, plugin_plan_ranges):
        scores, summary, _ = plugin_plan_ranges
        text = format_report(scores, summary)
        assert "Real-footage evaluation" in text
        assert "WEAPON_VISIBLE" in text
        assert "precision" in text
