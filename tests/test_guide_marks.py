"""Parental-guide marks (--guide): local target windows for the planner.

A marks JSON carries {"ranges": [{start, end, category?}]} in seconds -
the same schema the real-footage evaluator reads. PureFrame never fetches
guides itself; the file is authored from a saved guide page or an API
export. Inside a marked window the nudity threshold scales by
guide_threshold_factor (hint mode), and guide_mode "window" also turns
detector-silent guide shots into whole-shot full-frame blur (GUIDE_BOX)
so guide evidence can censor what the models missed - loudly, for review.

The three-shot fixture runs 0-4 s (dark), 4-6 s (busy), 6-10 s (light);
the fixture guide window 4.0-6.0 covers exactly the middle shot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pureframe.cli import generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.detect.nudity import Detection, NudityDetector
from pureframe.pipeline.shots import Action, Category
from tests.conftest import frame_has_marker


def _config(three_shot_video, tmp_path, **kwargs) -> Config:
    kwargs.setdefault("no_clip", True)
    kwargs.setdefault("no_audio", True)
    return Config.from_cli(
        input_path=three_shot_video,
        output_path=tmp_path / "out.mp4",
        profile=HardwareProfile.CPU,
        **kwargs,
    )


def _dense_settings(monkeypatch):
    import pureframe.cli

    original = pureframe.cli.get_settings

    def dense(profile, **kwargs):
        s = original(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense)


def _write_marks(tmp_path: Path, start: float, end: float) -> Path:
    path = tmp_path / "guide.json"
    path.write_text(
        json.dumps({"ranges": [{"start": start, "end": end}]}), encoding="utf-8"
    )
    return path


class TestConfigHashAndValidation:
    def test_guide_settings_change_the_hash(self, three_shot_video, tmp_path):
        plain = _config(three_shot_video, tmp_path)
        guided = _config(
            three_shot_video, tmp_path, guide_path=str(_write_marks(tmp_path, 4, 6))
        )
        assert guided.config_hash != plain.config_hash

    def test_bogus_mode_rejected(self, three_shot_video, tmp_path):
        with pytest.raises(Exception):
            _config(three_shot_video, tmp_path, guide_mode="bogus")


class TestHintMode:
    def test_threshold_boost_flags_a_marginal_shot(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        """A 0.50-score detection misses the default 0.55 bar everywhere -
        except inside the guide window, where the 0.7 factor applies."""
        marks = _write_marks(tmp_path, 4.0, 6.0)

        def marginal_on_busy(self, frames):
            return [
                [
                    Detection(
                        label="FEMALE_GENITALIA_EXPOSED",
                        score=0.50,
                        box=(60, 40, 260, 200),
                    )
                ]
                if frame_has_marker(f)
                else []
                for f in frames
            ]

        monkeypatch.setattr(NudityDetector, "detect_batch", marginal_on_busy)
        _dense_settings(monkeypatch)

        # Without the guide: the 0.50 score stays under the 0.55 bar.
        plain = generate_plan(_config(three_shot_video, tmp_path))
        assert all(v.action == Action.NONE for v in plain.verdicts)

        # With the guide window over the middle shot: 0.55 * 0.7 = 0.385.
        guided = generate_plan(
            _config(
                three_shot_video,
                tmp_path,
                guide_path=str(marks),
                guide_threshold_factor=0.7,
            )
        )
        flagged = [v for v in guided.verdicts if v.action != Action.NONE]
        assert len(flagged) == 1
        assert flagged[0].category == Category.NUDITY_EXPLICIT
        assert flagged[0].shot_index == 1
        # The boost is visible in the verdict: 0.55 * 0.7 -> 0.39 after
        # formatting - and the densify pass keeps the marginal detections'
        # boxes (it must scale its own bar by the same factor, or the
        # verdict would carry no blur boxes at all).
        assert "threshold: 0.39" in flagged[0].reasoning
        assert flagged[0].boxes, "boosted detections must densify into boxes"

    def test_hint_mode_never_blurs_detector_silent_shots(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        monkeypatch.setattr(
            NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
        )
        guided = generate_plan(
            _config(
                three_shot_video,
                tmp_path,
                guide_path=str(_write_marks(tmp_path, 4.0, 6.0)),
                guide_mode="hint",
            )
        )
        assert all(v.action == Action.NONE for v in guided.verdicts)


class TestWindowMode:
    def test_detector_silent_guide_shot_blurrs_whole_shot(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        monkeypatch.setattr(
            NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
        )
        guided = generate_plan(
            _config(
                three_shot_video,
                tmp_path,
                guide_path=str(_write_marks(tmp_path, 4.0, 6.0)),
                guide_mode="window",
            )
        )
        verdicts = {v.shot_index: v for v in guided.verdicts}
        assert verdicts[1].action == Action.FULL_FRAME_BLUR
        assert verdicts[1].category == Category.GUIDE_BOX
        assert "Guide-marked" in verdicts[1].reasoning
        # Shots outside the window stay untouched.
        assert verdicts[0].action == Action.NONE
        assert verdicts[2].action == Action.NONE

    def test_window_mode_keeps_stronger_detector_verdicts(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        """When the detector does flag the shot, its own verdict (with
        tracked boxes) wins - the guide only fills the gaps."""

        def strong_on_busy(self, frames):
            return [
                [
                    Detection(
                        label="FEMALE_GENITALIA_EXPOSED",
                        score=0.9,
                        box=(60, 40, 260, 200),
                    )
                ]
                if frame_has_marker(f)
                else []
                for f in frames
            ]

        monkeypatch.setattr(NudityDetector, "detect_batch", strong_on_busy)
        _dense_settings(monkeypatch)
        guided = generate_plan(
            _config(
                three_shot_video,
                tmp_path,
                guide_path=str(_write_marks(tmp_path, 4.0, 6.0)),
                guide_mode="window",
            )
        )
        flagged = [v for v in guided.verdicts if v.action != Action.NONE]
        assert len(flagged) == 1
        assert flagged[0].category == Category.NUDITY_EXPLICIT
        assert flagged[0].boxes, "the detector verdict keeps its tracked boxes"

    def test_marks_outside_the_clip_touch_nothing(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        monkeypatch.setattr(
            NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
        )
        guided = generate_plan(
            _config(
                three_shot_video,
                tmp_path,
                guide_path=str(_write_marks(tmp_path, 40.0, 50.0)),
                guide_mode="window",
            )
        )
        assert all(v.action == Action.NONE for v in guided.verdicts)
