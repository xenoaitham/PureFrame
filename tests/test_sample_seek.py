"""Seek-based frame extraction must return the same frames as a full decode.

extract_frames now input-seeks to the requested window instead of decoding
the video from frame 0 for every call (a near-quadratic cost across a
movie's shots). These tests pin that the seek path returns frames
pixel-equal to a sequential decode on a constant-framerate clip.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from pureframe.pipeline.sample import extract_frames
from pureframe.utils.ffmpeg import extract_metadata, probe

W, H, FPS, FRAMES = 128, 96, 25, 90


@pytest.fixture(scope="module")
def unique_color_clip(tmp_path_factory) -> Path:
    """CFR clip where every frame has a unique color (channel values encode
    the frame index), so a mismatched frame cannot pass by luck."""
    raw = np.zeros((FRAMES, H, W, 3), dtype=np.uint8)
    for i in range(FRAMES):
        raw[i, :, :, 0] = (i * 2) % 256  # B
        raw[i, :, :, 1] = (255 - i * 2) % 256  # G
        raw[i, :, :, 2] = (i * 3) % 256  # R
    clip = tmp_path_factory.mktemp("seekclip") / "colors.mp4"

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "pipe:",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )
    proc.stdin.write(raw.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, "synthetic clip generation failed"
    return clip


@pytest.fixture(scope="module")
def ground_truth(unique_color_clip: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(unique_color_clip))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == FRAMES, f"expected {FRAMES} frames, got {len(frames)}"
    return frames


def _assert_matches(clip, ground_truth, indices):
    got = extract_frames(clip, indices, downscale_max_edge=240)
    assert sorted(got.keys()) == sorted(indices)
    for idx in indices:
        diff = np.abs(got[idx].astype(int) - ground_truth[idx].astype(int))
        assert diff.max() <= 3, f"frame {idx} differs (max channel diff {diff.max()})"


def test_extraction_matches_sequential_decode(unique_color_clip, ground_truth):
    # 0/1 exercise the no-seek path; scattered later indices exercise seeks.
    _assert_matches(unique_color_clip, ground_truth, [0, 1, 2, 30, 45, 60, 89])


def test_extraction_late_window_only(unique_color_clip, ground_truth):
    _assert_matches(unique_color_clip, ground_truth, [70, 75, 80, 88])


def test_extraction_reuses_provided_metadata(unique_color_clip, ground_truth):
    meta = extract_metadata(probe(unique_color_clip))
    got = extract_frames(
        unique_color_clip, [10, 40, 77], downscale_max_edge=240, meta=meta
    )
    assert sorted(got.keys()) == [10, 40, 77]
    for idx in (10, 40, 77):
        diff = np.abs(got[idx].astype(int) - ground_truth[idx].astype(int))
        assert diff.max() <= 3


def test_extraction_empty_indices(unique_color_clip):
    assert extract_frames(unique_color_clip, [], downscale_max_edge=240) == {}
