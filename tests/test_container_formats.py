"""Container-format support: MKV, WebM and AVI end to end.

The README promises "any MP4, MKV, AVI, or WebM". These tests run the full
``process`` flow (probe → scene detect → plan → render → mux) over tiny
lavfi-generated clips in each non-MP4 container. NudeNet is mocked so the
suite stays fast and CI-safe: each clip carries a small magenta marker and
the mock fires exactly on marker frames.

Two clips per container/codec pair:

* a single-shot clip — the whole shot is flagged, so the render takes the
  full re-encode path and the final mux into the input's container;
* the shared three-shot clip (grey | pattern+marker | grey) — only the
  middle shot is flagged, so the smart renderer stream-copies the outer
  shots, re-encodes the middle one and concatenates, all inside the input's
  container.

Before the source-matched encoder landed, every re-encode was H.264: WebM
refused it outright ("Only VP8 or VP9 or AV1 video … supported") and AVI
rejected the MP4-style bitstream ("no startcode found"), so ``process``
failed on both containers. The keyframe probe also trusted the decoder's
``-skip_frame nokey``, which VP8/VP9 ignore — every frame came back as a
keyframe, so WebM copy cuts would have landed off-keyframe.
"""

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from tests.conftest import (
    THREE_SHOT_FPS,
    THREE_SHOT_FRAMES,
    THREE_SHOT_MARKER_FILTER,
    frame_has_marker,
    generate_three_shot_clip,
)

# Censor region in native pixel coords — what the mocked detector returns.
BOX = (60, 40, 260, 200)
# The 40×40 marker sits in the middle of BOX; the band above it (inside
# BOX, outside the marker) keeps testsrc2 detail for the blur check.
BLUR_ROI = np.s_[50:90, 70:250]

# name → (container, source-clip encoder flags, smart path expected).
# Fast settings only; clips are 320×240. AVI/H.264 carries no presentation
# timestamps, so its keyframe map is unreadable and it takes the full
# re-encode fallback by design.
CASES = {
    "mkv-h264": ("mkv", ["-c:v", "libx264", "-crf", "28"], True),
    "webm-vp9": (
        "webm",
        ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "40"]
        + ["-deadline", "realtime", "-cpu-used", "8"],
        True,
    ),
    "webm-vp8": (
        "webm",
        ["-c:v", "libvpx", "-b:v", "400k", "-deadline", "realtime"],
        True,
    ),
    "avi-mpeg4": ("avi", ["-c:v", "mpeg4", "-qscale:v", "6"], True),
    "avi-h264": ("avi", ["-c:v", "libx264", "-crf", "28"], False),
}


def _generate_single_shot_clip(path: Path, codec_args: list[str]) -> None:
    """4 s of testsrc2 with the marker between 1 s and 2 s (frames 15–29)."""
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
            f"testsrc2=duration=4:size=320x240:rate={THREE_SHOT_FPS}",
            "-vf",
            THREE_SHOT_MARKER_FILTER.replace("between(t,4.5,5.5)", "between(t,1,2)"),
            *codec_args,
            "-g",
            "15",  # keyframe every second
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def _ffprobe_stream(path: Path, entries: str, count_frames: bool = False) -> str:
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *(["-count_frames"] if count_frames else []),
            "-show_entries",
            f"stream={entries}",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return res.stdout.strip()


def _nb_frames(path: Path) -> int:
    return int(_ffprobe_stream(path, "nb_read_frames", count_frames=True))


def _video_codec(path: Path) -> str:
    return _ffprobe_stream(path, "codec_name")


def _roi_lap_var(path: Path, frame_idx: int) -> float:
    """Laplacian variance of BLUR_ROI at *frame_idx*, read sequentially.

    Sequential reads rather than ``CAP_PROP_POS_FRAMES``: seeking in AVI is
    index-based and lands off by a frame or two for H.264 with B-frames.
    """
    cap = cv2.VideoCapture(str(path))
    frame = None
    for _ in range(frame_idx + 1):
        ok, frame = cap.read()
        assert ok, f"could not read frame {frame_idx} from {path}"
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray[BLUR_ROI], cv2.CV_64F).var())


def _run_process(clip: Path, out_path: Path, monkeypatch) -> None:
    import pureframe.cli
    from pureframe.pipeline.detect.nudity import Detection, NudityDetector

    def mocked_detect_batch(self, frames_bgr):
        return [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.99, box=BOX)]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    monkeypatch.setattr(NudityDetector, "detect_batch", mocked_detect_batch)

    # The CPU profile samples 2 keyframes per shot (first/last frame), which
    # would miss a 1 s marker window entirely; sample densely enough to hit it.
    original_get_settings = pureframe.cli.get_settings

    def dense_settings(profile, **kwargs):
        s = original_get_settings(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense_settings)

    config = Config.from_cli(
        input_path=clip,
        output_path=out_path,
        profile=HardwareProfile.CPU,
        no_clip=True,
        no_audio=True,
    )
    pureframe.cli.process_file(config)


def _assert_rendered_in_place(
    clip: Path, out_path: Path, total_frames: int, blurred_frame: int
) -> None:
    # The render produced a playable file in the SAME container and codec…
    assert out_path.exists() and out_path.stat().st_size > 0
    assert _video_codec(out_path) == _video_codec(clip)

    # …with the exact input frame count (no duplicated/truncated GOPs)…
    assert _nb_frames(out_path) == _nb_frames(clip) == total_frames

    # …and the censor blur actually landed inside the flagged window.
    in_var = _roi_lap_var(clip, blurred_frame)
    out_var = _roi_lap_var(out_path, blurred_frame)
    assert in_var > 20, f"test ROI has no detail to blur ({in_var=:.1f})"
    assert out_var < in_var * 0.35, (
        f"expected blur inside the flagged ROI ({out_var=:.1f} vs {in_var=:.1f})"
    )


@pytest.fixture(scope="session")
def single_shot_clip(request, tmp_path_factory):
    ext, codec_args, _ = CASES[request.param]
    clip = tmp_path_factory.mktemp("container") / f"{request.param}.{ext}"
    _generate_single_shot_clip(clip, codec_args)
    return clip


@pytest.fixture(scope="session")
def three_shot_clip(request, tmp_path_factory):
    ext, codec_args, _ = CASES[request.param]
    clip = tmp_path_factory.mktemp("container") / f"{request.param}-3shot.{ext}"
    return generate_three_shot_clip(clip, codec_args)


@pytest.mark.parametrize("single_shot_clip", list(CASES), indirect=True)
def test_process_full_reencode_keeps_container(single_shot_clip, tmp_path, monkeypatch):
    out_path = tmp_path / f"out{single_shot_clip.suffix}"
    _run_process(single_shot_clip, out_path, monkeypatch)
    _assert_rendered_in_place(
        single_shot_clip, out_path, 4 * THREE_SHOT_FPS, blurred_frame=20
    )


@pytest.mark.parametrize("three_shot_clip", list(CASES), indirect=True)
def test_process_smart_render_keeps_container(
    three_shot_clip, tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="pureframe")
    out_path = tmp_path / f"out{three_shot_clip.suffix}"
    _run_process(three_shot_clip, out_path, monkeypatch)
    _assert_rendered_in_place(
        three_shot_clip, out_path, THREE_SHOT_FRAMES, blurred_frame=75
    )

    smart_expected = CASES[three_shot_clip.stem.removesuffix("-3shot")][2]
    if smart_expected:
        assert "Smart render:" in caplog.text
        assert "falling back" not in caplog.text
    else:
        assert "falling back to full re-encode" in caplog.text
