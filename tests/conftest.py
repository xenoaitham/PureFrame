import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def mock_store(monkeypatch, tmp_path):
    import pureframe.cli
    from pureframe.checkpoint import CheckpointStore

    db_path = tmp_path / "test_checkpoints.db"
    store = CheckpointStore(db_path)
    monkeypatch.setattr(pureframe.cli, "get_store", lambda: store)
    return store


@pytest.fixture(scope="session")
def fixtures_dir():
    d = Path(__file__).parent / "fixtures"
    d.mkdir(exist_ok=True, parents=True)
    return d


@pytest.fixture(scope="session")
def synthetic_video(fixtures_dir):
    out_path = fixtures_dir / "synthetic_explicit.mp4"
    if out_path.exists():
        return out_path

    # Generate a 15-second 24fps 720p video
    # Frames 100-250 will have an ellipse that represents explicit content.
    # A 1 kHz sine wave audio track.

    import cv2
    import numpy as np

    temp_vid = fixtures_dir / "temp_synth.mp4"
    w, h = 1280, 720
    fps = 24
    total_frames = 15 * fps

    out = cv2.VideoWriter(str(temp_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(total_frames):
        frame = np.full((h, w, 3), 128, dtype=np.uint8)  # solid grey

        if 100 <= i <= 250:
            # Flesh-toned ellipse
            # Let's say flesh tone is BGR: 140, 180, 210
            # NudeNet expects realistic things, so a synthetic shape might not be detected!
            # Wait, NudeNet is very specific. A simple ellipse might NOT be detected by NudeNet.
            # But the prompt says: "Generate synthetic test clips programmatically - solid color frames with shapes that look like skin tones in specific bounding boxes - and verify the pipeline detects and censors them correctly."
            # Actually, NudeNet will likely NOT detect a random ellipse.
            # So for test_e2e, we'll need to mock NudityDetector OR force it.
            # I will draw something skin-colored but if it fails to detect, I might need to mock NudityDetector in the test.
            # Let's draw an ellipse.
            cv2.ellipse(frame, (300, 250), (100, 100), 0, 0, 360, (140, 180, 210), -1)

        # Draw frame number for debug
        cv2.putText(
            frame,
            f"Frame {i}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        out.write(frame)
    out.release()

    # Add audio and re-encode to proper format.
    # -nostdin prevents ffmpeg from waiting on stdin (Windows CI hangs without it).
    # capture_output + 120s timeout keeps a misbehaving ffmpeg from hanging the suite.
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=15",
            "-i",
            str(temp_vid),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )

    os.remove(temp_vid)
    return out_path


# Three-shot clip geometry shared by the multi-shot tests: dark grey 4 s |
# testsrc2 pattern 2 s | light grey 4 s at 15 fps, 320×240. Hard cuts at
# frames 60 and 90; a 40×40 magenta marker at (140,100) is drawn between
# 4.5 s and 5.5 s (frames 68–82), inside the middle shot only.
THREE_SHOT_FPS = 15
THREE_SHOT_FRAMES = 150
THREE_SHOT_MARKER_FRAMES = range(68, 83)
THREE_SHOT_MARKER_FILTER = (
    "drawbox=x=140:y=100:w=40:h=40:color=magenta@1:t=fill:enable='between(t,4.5,5.5)'"
)


def generate_three_shot_clip(path: Path, codec_args: list[str]) -> Path:
    size = "320x240"
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
            f"color=c=0x303030:size={size}:rate={THREE_SHOT_FPS}:duration=4",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=duration=2:size={size}:rate={THREE_SHOT_FPS}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0xC0C0C0:size={size}:rate={THREE_SHOT_FPS}:duration=4",
            "-filter_complex",
            f"[0:v][1:v][2:v]concat=n=3:v=1:a=0[cat];[cat]{THREE_SHOT_MARKER_FILTER}[v]",
            "-map",
            "[v]",
            *codec_args,
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


def frame_has_marker(frame_bgr) -> bool:
    """True when the magenta marker is at the pixel the tests agree on."""
    b, g, r = (int(v) for v in frame_bgr[120, 160])
    return b > 180 and r > 180 and g < 90


@pytest.fixture(scope="session")
def three_shot_video(tmp_path_factory):
    clip = tmp_path_factory.mktemp("three_shot") / "three_shot.mp4"
    return generate_three_shot_clip(clip, ["-c:v", "libx264", "-crf", "28"])
