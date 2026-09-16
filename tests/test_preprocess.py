"""Low-light / low-contrast preprocessing: the dark-scene detection branch.

Unit tests pin the trigger math (dark frames, low-contrast grayscale,
untouched normal content) and the stretch's lift. The clip fixtures make
the pipeline claim concrete: a dark explicit clip and a grayscale
explicit clip that the mocked detector cannot see on raw pixels but
flags once the normalization branch runs - with the branch disabled the
plan flags nothing, which is the fails-without-it proof.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from pureframe.cli import generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.detect import preprocess
from pureframe.pipeline.detect.nudity import NudityDetector
from pureframe.pipeline.detect.preprocess import (
    needs_normalization,
    normalize_for_detection,
)
from pureframe.pipeline.shots import Action

FPS = 15
# The mock's visibility bar: a normalized dark clip lifts mean luma from
# ~38 to ~86, a normalized flat gray clip lifts luma std from ~8 to ~70.
DARK_VISIBLE_MEAN = 55
GRAY_VISIBLE_STD = 40


def _gray_frame(base: int, region: int, w: int = 320, h: int = 240) -> np.ndarray:
    frame = np.full((h, w, 3), base, dtype=np.uint8)
    frame[70:200, 100:250] = region
    rng = np.random.default_rng(7)
    noise = rng.integers(-8, 9, (h, w), dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)


class TestTriggerMath:
    def test_dark_frame_triggers(self):
        dark = _gray_frame(30, 70)
        assert needs_normalization(dark) is True

    def test_bright_colored_frame_never_triggers(self):
        frame = np.full((240, 320, 3), (60, 80, 200), dtype=np.uint8)
        frame[70:200, 100:250] = (40, 120, 220)
        assert needs_normalization(frame) is False

    def test_low_contrast_grayscale_triggers(self):
        flat = _gray_frame(128, 145)
        assert needs_normalization(flat) is True

    def test_full_range_grayscale_does_not_trigger(self):
        frame = _gray_frame(40, 200)
        assert needs_normalization(frame) is False

    def test_channel_deviation_gates_the_contrast_branch(self):
        # Same compressed luma histogram, but colored: the low-contrast
        # branch must stay off (only the dark branch may apply).
        frame = _gray_frame(110, 160).copy()
        frame[:, :, 0] = np.clip(frame[:, :, 0].astype(int) + 40, 0, 255).astype(
            np.uint8
        )
        assert (
            preprocess.channel_deviation(frame) > preprocess.GRAYSCALE_CHANNEL_DEVIATION
        )
        assert preprocess.is_low_contrast_grayscale(frame) is False


class TestStretch:
    def test_dark_frame_gains_luma_and_contrast(self):
        dark = _gray_frame(30, 70)
        out = normalize_for_detection(dark)
        g_in = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
        g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        assert g_out.mean() > g_in.mean() * 1.5
        assert g_out.std() > g_in.std() * 2

    def test_grayscale_stays_grayscale_and_gains_contrast(self):
        flat = _gray_frame(128, 145)
        out = normalize_for_detection(flat)
        dev = preprocess.channel_deviation(out)
        assert dev <= preprocess.GRAYSCALE_CHANNEL_DEVIATION + 2
        assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).std() > GRAY_VISIBLE_STD

    def test_flat_frame_passes_through(self):
        flat = np.full((240, 320, 3), 128, dtype=np.uint8)
        assert normalize_for_detection(flat) is flat

    def test_geometry_is_untouched(self):
        frame = _gray_frame(30, 70)
        out = normalize_for_detection(frame)
        assert out.shape == frame.shape
        # A linear map cannot move the bright region's bounding box.
        for img in (frame, out):
            mask = img[:, :, 0] > img[:, :, 0].mean()
            ys, xs = np.nonzero(mask)
            assert 100 <= xs.min() <= 100 + 3
            assert 70 <= ys.min() <= 70 + 3


LOW_CONTRAST_LIFT_MIN = 40.0


def _build_gray_clip(path: Path, base: int, region: int) -> Path:
    """60-frame 320x240 clip: flat gray with a lighter region and shared noise."""
    tmp = path.with_suffix(".tmp.mp4")
    w, h = 320, 240
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for _ in range(60):
        frame = np.full((h, w, 3), base, dtype=np.uint8)
        frame[70:200, 100:250] = region
        noise = rng.integers(-8, 9, (h, w), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise[:, :, None], 0, 255).astype(
            np.uint8
        )
        writer.write(frame)
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    tmp.unlink()
    return path


@pytest.fixture(scope="session")
def dark_clip(tmp_path_factory):
    # mean luma ~38: dark branch. Post-stretch mean lands ~86.
    return _build_gray_clip(tmp_path_factory.mktemp("dark") / "dark.mp4", 30, 70)


@pytest.fixture(scope="session")
def grayscale_clip(tmp_path_factory):
    # mean luma ~130, std ~8: grayscale + low-contrast branch.
    # Post-stretch std lands ~70.
    return _build_gray_clip(tmp_path_factory.mktemp("gray") / "gray.mp4", 128, 145)


class _NormalizedVisibleSession:
    """Stands in for the NudeNet session inside the real detect_batch.

    The normalization branch runs inside NudityDetector.detect_batch, so
    the mock replaces the inner model session, not the batch method:
    raw clip pixels stay invisible (dark clip mean ~38, gray clip std
    ~8); once the branch stretches a frame past the visibility bar the
    session reports the region, in NudeNet's own dict+xywh format.
    """

    def __init__(self, mode: str):
        self.mode = mode

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        visible = (
            gray.mean() > DARK_VISIBLE_MEAN
            if self.mode == "dark"
            else gray.std() > GRAY_VISIBLE_STD
        )
        if not visible:
            return []
        return [
            {
                "class": "FEMALE_GENITALIA_EXPOSED",
                "score": 0.9,
                "box": [100, 70, 150, 130],
            }
        ]


def _install_normalized_only_mock(monkeypatch, mode: str) -> None:
    fake = _NormalizedVisibleSession(mode)
    monkeypatch.setattr(
        NudityDetector, "_load", lambda self: setattr(self, "detector", fake)
    )


def _plan(clip, tmp_path, monkeypatch, **kwargs):
    kwargs.setdefault("no_clip", True)
    kwargs.setdefault("no_audio", True)
    return generate_plan(
        Config.from_cli(
            input_path=clip,
            output_path=tmp_path / "out.mp4",
            profile=HardwareProfile.CPU,
            **kwargs,
        )
    )


class TestPipeline:
    @pytest.mark.parametrize(
        "fixture_name,mode", [("dark_clip", "dark"), ("grayscale_clip", "gray")]
    )
    def test_dark_and_grayscale_clips_flag_after_normalization(
        self, fixture_name, mode, request, tmp_path, monkeypatch
    ):
        clip = request.getfixturevalue(fixture_name)
        _install_normalized_only_mock(monkeypatch, mode)
        plan = _plan(clip, tmp_path, monkeypatch)
        flagged = [v for v in plan.verdicts if v.action != Action.NONE]
        assert flagged, f"the {mode} clip must flag once normalization runs"
        assert flagged[0].boxes, "the flagged verdict must carry boxes"

    @pytest.mark.parametrize(
        "fixture_name,mode", [("dark_clip", "dark"), ("grayscale_clip", "gray")]
    )
    def test_clips_stay_unflagged_without_the_branch(
        self, fixture_name, mode, request, tmp_path, monkeypatch
    ):
        clip = request.getfixturevalue(fixture_name)
        _install_normalized_only_mock(monkeypatch, mode)
        monkeypatch.setattr(
            "pureframe.pipeline.detect.nudity.needs_normalization", lambda f: False
        )
        plan = _plan(clip, tmp_path, monkeypatch)
        assert all(v.action == Action.NONE for v in plan.verdicts), (
            "without the branch the mock cannot see the raw pixels"
        )
