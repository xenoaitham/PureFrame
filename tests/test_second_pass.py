"""Second-pass rescan: dense stride plus tiled zoom catches what the first
pass sampled over.

The fixture is one continuous 10 s grey clip (15 fps, 320x240) carrying a
5-frame (1/3 s) flash of a large tan rectangle at 6.4-6.7 s and a
persistent 30x30 tan rectangle in the lower right from 6 s on. Shot
boundaries are pinned by stubbing detect_shots so the keyframe math is
deterministic: the CPU profile's 2 keyframes per shot land on frames
0/59, 60/89 and 90/149 - none of them inside the flash, and the small
rectangle scores 0.30 (under the 0.55 bar, over the 0.25 rescan floor).
The old path flags nothing; the rescan's dense stride (the profile's
densify stride, 5) lands on flash frame 100, and the tiled zoom lifts
the small rectangle over the bar on every other sampled frame.

The mocked detector is honest about the failure mode it simulates:
score scales with the skin region's height relative to the frame it was
handed, which is exactly what full-frame sampling loses and tiling
restores.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from pureframe.cli import generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.detect.nudity import Detection, NudityDetector
from pureframe.pipeline.shots import Action, Category, Shot

FPS = 15
SKIN_LABEL = "FEMALE_GENITALIA_EXPOSED"

# tan in ffmpeg RGB syntax; reads as skin to the mask below after BGR decode
TAN = "0xD2B48C"
FLASH_FILTER = (
    f"drawbox=x=40:y=40:w=200:h=150:color={TAN}@1:t=fill:enable='between(t,6.4,6.7)'"
)
SMALL_FILTER = (
    f"drawbox=x=240:y=180:w=30:h=30:color={TAN}@1:t=fill:enable='between(t,6,10)'"
)


@pytest.fixture(scope="session")
def second_pass_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("second_pass") / "second_pass.mp4"
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
            f"color=c=0xC0C0C0:size=320x240:rate={FPS}:duration=10",
            "-vf",
            f"{FLASH_FILTER},{SMALL_FILTER}",
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


def _fixed_shots() -> list[Shot]:
    return [
        Shot(index=0, start_frame=0, end_frame=60, start_time=0.0, end_time=4.0),
        Shot(index=1, start_frame=60, end_frame=90, start_time=4.0, end_time=6.0),
        Shot(index=2, start_frame=90, end_frame=150, start_time=6.0, end_time=10.0),
    ]


def _skin_detect(frames):
    """Score = f(skin region height / frame height): the small-object model.

    A region under 15% of the frame's height scores 0.30 (real but under
    the medium bar); at or over 15% it scores 0.80. Tiling quadruples a
    region's relative height, which is the catch mechanism under test.
    """
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
        score = 0.80 if rel_h >= 0.15 else 0.30
        batch.append([Detection(label=SKIN_LABEL, score=score, box=box)])
    return batch


def _install_skin_mock(monkeypatch) -> dict:
    """Patch NudityDetector.detect_batch; return {"frames": count} tally."""
    tally = {"frames": 0}

    def detect_batch(self, frames):
        tally["frames"] += len(frames)
        return _skin_detect(frames)

    monkeypatch.setattr(NudityDetector, "detect_batch", detect_batch)
    return tally


def _config(video, tmp_path, **kwargs) -> Config:
    kwargs.setdefault("no_clip", True)
    kwargs.setdefault("no_audio", True)
    return Config.from_cli(
        input_path=video,
        output_path=tmp_path / "out.mp4",
        profile=HardwareProfile.CPU,
        **kwargs,
    )


def _plan_with_skin(video, tmp_path, monkeypatch, **kwargs):
    _install_skin_mock(monkeypatch)
    import pureframe.cli

    monkeypatch.setattr(pureframe.cli, "detect_shots", lambda *a, **k: _fixed_shots())
    return generate_plan(_config(video, tmp_path, **kwargs))


class TestSecondPass:
    def test_disabled_flags_nothing(self, second_pass_video, tmp_path, monkeypatch):
        plan = _plan_with_skin(
            second_pass_video, tmp_path, monkeypatch, second_pass=False
        )
        assert all(v.action == Action.NONE for v in plan.verdicts)

    def test_rescan_catches_flash_and_small_region(
        self, second_pass_video, tmp_path, monkeypatch
    ):
        plan = _plan_with_skin(second_pass_video, tmp_path, monkeypatch)
        flagged = [v for v in plan.verdicts if v.action != Action.NONE]
        assert len(flagged) == 1
        verdict = flagged[0]
        assert verdict.shot_index == 2
        assert verdict.category == Category.NUDITY_EXPLICIT
        assert verdict.boxes, "the rescan verdict must carry blur boxes"

        # The small background rectangle survives into the plan's boxes
        # (padding and tracking give a few pixels of slack).
        small = [
            b for b in verdict.boxes if abs(b.x1 - 240) < 30 and abs(b.y1 - 180) < 30
        ]
        assert small, "the tiled pass's small-region box is missing"

        # The flash region lands too: the dense stride sampled a flash
        # frame and its large box reached the plan.
        flash = [b for b in verdict.boxes if b.x1 < 70 and b.y1 < 70 and b.x2 > 200]
        assert flash, "the dense-stride flash box is missing"

    def test_rescan_costs_nothing_without_candidates(
        self, second_pass_video, tmp_path, monkeypatch
    ):
        import pureframe.cli

        monkeypatch.setattr(
            pureframe.cli, "detect_shots", lambda *a, **k: _fixed_shots()
        )

        tally = _install_skin_mock(monkeypatch)
        generate_plan(_config(second_pass_video, tmp_path))
        with_rescan = tally["frames"]

        tally = _install_skin_mock(monkeypatch)
        generate_plan(_config(second_pass_video, tmp_path, second_pass=False))
        without_rescan = tally["frames"]

        assert without_rescan == 6  # 2 keyframes per shot x 3 shots, nothing else
        assert with_rescan > without_rescan


class TestConfig:
    def test_second_pass_flag_changes_the_hash(self, three_shot_video, tmp_path):
        on = _config(three_shot_video, tmp_path)
        off = _config(three_shot_video, tmp_path, second_pass=False)
        assert on.config_hash != off.config_hash
