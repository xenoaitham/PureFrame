"""Extreme close-ups: quadrant-zoom multi-scale inference.

A close-up fills the frame with a body part; the detector, trained on
body-scale views, reads nothing at full-frame scale. When a shot is
fully detector-silent but the CLIP scene reads sexual, the pipeline
re-runs the detector on exact center-crop quadrants (a 2x zoom) of the
keyframes it already has and merges anything above the shot's bar.

The mocked session is honest about the failure mode: score scales with
the region's height relative to the frame handed to it, which is
exactly what a close-up breaks and the zoom restores.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from pureframe.cli import generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.detect.nudity import NudityDetector
from pureframe.pipeline.second_pass import closeup_scan
from pureframe.pipeline.shots import Action, Category, Shot

FPS = 15
SKIN_LABEL = "FEMALE_GENITALIA_EXPOSED"
TAN = "0xD2B48C"  # ffmpeg RGB; reads as skin to the mask below

# 30x30 region at (40, 30): relative height 0.125 on the full 320x240
# frame, 0.25 once its quadrant is upscaled back to 320x240. Permanent,
# so the per-shot keyframes always land on it while staying silent.
RECT_FILTER = f"drawbox=x=40:y=30:w=30:h=30:color={TAN}@1:t=fill"


@pytest.fixture(scope="session")
def closeup_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("closeup") / "closeup.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0xC0C0C0:size=320x240:rate={FPS}:duration=6",
            "-vf",
            RECT_FILTER,
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-g",
            "15",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


def _skin_detect(frames):
    """A close-up reads as nothing at full-frame scale: below 20% of the
    frame's height the session reports no detection at all (not a weak
    one), and at or above it the region is body-scale and scores 0.85."""
    batch = []
    for f in frames:
        b, g, r = (f[:, :, i].astype(np.int16) for i in range(3))
        mask = (r > 120) & (r - g > 15) & (g - b > 15)
        if int(mask.sum()) < 30:
            batch.append([])
            continue
        ys, xs = np.nonzero(mask)
        box = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        rel_h = (box[3] - box[1]) / f.shape[0]
        if rel_h < 0.20:
            batch.append([])
            continue
        from pureframe.pipeline.detect.nudity import Detection

        batch.append([Detection(label=SKIN_LABEL, score=0.85, box=box)])
    return batch


def _fixed_shots() -> list[Shot]:
    return [Shot(index=0, start_frame=0, end_frame=90, start_time=0.0, end_time=6.0)]


class _SexualScene:
    """CLIP stand-in: the scene reads as a sexual situation."""

    def __init__(self, settings):
        self.enabled = True

    def classify_shot(self, frame):
        from pureframe.pipeline.detect.scene_clip import ShotContext

        return ShotContext(
            explicit_act_score=0.7,
            implied_sex_score=0.1,
            kissing_score=0.0,
            safe_score=0.05,
        )

    def unload(self):
        pass


class _NeutralScene(_SexualScene):
    def classify_shot(self, frame):
        from pureframe.pipeline.detect.scene_clip import ShotContext

        return ShotContext(
            explicit_act_score=0.05,
            implied_sex_score=0.05,
            kissing_score=0.0,
            safe_score=0.9,
        )


class TestCloseupScanUnit:
    def test_zoom_lifts_the_region_over_the_bar(self):
        from pureframe.pipeline.detect.tiling import tile_grid

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[30:60, 40:70] = (140, 180, 210)  # BGR tan, inside one quadrant

        class Session:
            def detect_batch(self, frames):
                return _skin_detect(frames)

        detector = Session()
        results = closeup_scan({10: frame}, detector, threshold=0.55)
        assert 10 in results
        dets = results[10]
        assert dets and all(d.score >= 0.55 for d in dets)
        # The box lands in the top-left quadrant, where the region lives.
        assert all(d.box[0] < 160 and d.box[1] < 120 for d in dets)
        # Quadrants are exact (no overlap).
        assert len(tile_grid(320, 240, grid=(2, 2), overlap=0.0)) == 4

    def test_silent_frame_stays_silent(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[30:60, 40:70] = (128, 128, 128)  # grey: no skin

        class Session:
            def detect_batch(self, frames):
                return _skin_detect(frames)

        assert closeup_scan({0: frame}, Session(), threshold=0.55) == {}


class TestPlanPipeline:
    def _plan(self, video, tmp_path, monkeypatch, scene_cls):
        tally = {"frames": 0}

        def detect_batch(self, frames):
            tally["frames"] += len(frames)
            return _skin_detect(frames)

        monkeypatch.setattr(NudityDetector, "detect_batch", detect_batch)
        import pureframe.cli

        monkeypatch.setattr(
            pureframe.cli, "detect_shots", lambda *a, **k: _fixed_shots()
        )
        monkeypatch.setattr(pureframe.cli, "SceneClassifier", scene_cls)
        config = Config.from_cli(
            input_path=video,
            output_path=tmp_path / "out.mp4",
            profile=HardwareProfile.CPU,
            no_clip=False,
            no_audio=True,
        )
        return generate_plan(config), tally

    def test_silent_sexual_scene_flags_via_closeup(
        self, closeup_video, tmp_path, monkeypatch
    ):
        plan, _ = self._plan(closeup_video, tmp_path, monkeypatch, _SexualScene)
        flagged = [v for v in plan.verdicts if v.action != Action.NONE]
        assert len(flagged) == 1
        assert flagged[0].category == Category.NUDITY_EXPLICIT
        assert flagged[0].boxes, "the close-up verdict must carry boxes"
        # The zoomed region's box sits in the top-left quadrant.
        assert any(b.x1 < 160 and b.y1 < 120 for b in flagged[0].boxes), (
            "expected a box where the zoomed region lives"
        )

    def test_silent_neutral_scene_never_runs_the_pass(
        self, closeup_video, tmp_path, monkeypatch
    ):
        plan, tally = self._plan(closeup_video, tmp_path, monkeypatch, _NeutralScene)
        assert all(v.action == Action.NONE for v in plan.verdicts)
        # 2 keyframes only: first pass, no rescan, no quadrant zoom.
        assert tally["frames"] == 2
