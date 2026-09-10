"""`pureframe preview --before-after`: paired original/censored PNGs.

Each flagged shot gets a full-resolution pair - the untouched frame and the
same frame with the plan's censoring applied through the same overlay
callback the real renderer uses - embedded side-by-side in the HTML report.
Lets a user verify blur placement per shot without rendering the video, and
gives the GUI loadable images for its before/after view.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from pureframe.cli import app, generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from tests.conftest import frame_has_marker

runner = CliRunner()

BOX = (60, 40, 260, 200)
# Inside the mock detector's box, above the marker: testsrc2 detail that the
# blur must flatten.
ROI = np.s_[60:100, 80:240]


def _make_plan(three_shot_video, tmp_path, monkeypatch) -> Path:
    from pureframe.pipeline.detect.nudity import Detection, NudityDetector

    def mocked_detect_batch(self, frames_bgr):
        return [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.99, box=BOX)]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    monkeypatch.setattr(NudityDetector, "detect_batch", mocked_detect_batch)

    import pureframe.cli

    original = pureframe.cli.get_settings

    def dense(profile, **kwargs):
        s = original(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense)

    config = Config.from_cli(
        input_path=three_shot_video,
        output_path=tmp_path / "unused.mp4",
        profile=HardwareProfile.CPU,
        no_clip=True,
        no_audio=True,
    )
    plan = generate_plan(config)
    plan_path = tmp_path / "three_shot.mp4.censorplan.json"
    plan.serialize(plan_path)
    return plan_path


def _lap_var(img_path: Path) -> float:
    img = cv2.imread(str(img_path))
    assert img is not None, f"unreadable image {img_path}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray[ROI], cv2.CV_64F).var())


def test_before_after_pairs_are_written_and_blurred(
    three_shot_video, tmp_path, monkeypatch
):
    plan_path = _make_plan(three_shot_video, tmp_path, monkeypatch)
    html = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["preview", str(plan_path), "--output", str(html), "--before-after"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    frames_dir = three_shot_video.with_name(three_shot_video.name + ".preview_frames")
    before = frames_dir / "shot_001_before.png"
    after = frames_dir / "shot_001_after.png"
    assert before.exists() and after.exists()

    # Only the flagged shot gets a pair.
    assert not (frames_dir / "shot_000_before.png").exists()

    # The "after" frame is genuinely censored in the flagged ROI.
    in_var = _lap_var(before)
    out_var = _lap_var(after)
    assert in_var > 20, f"before-frame ROI has no detail ({in_var=:.1f})"
    assert out_var < in_var * 0.35, (
        f"after-frame is not blurred ({out_var=:.1f} vs {in_var=:.1f})"
    )

    # And the report references the pair with loadable relative paths.
    html_text = html.read_text(encoding="utf-8")
    assert f"{os.path.relpath(before, html.parent).replace(os.sep, '/')}" in html_text
    for src in (before, after):
        assert (html.parent / os.path.relpath(src, html.parent)).exists()


def test_before_after_custom_frames_dir(three_shot_video, tmp_path, monkeypatch):
    plan_path = _make_plan(three_shot_video, tmp_path, monkeypatch)
    custom = tmp_path / "my_frames"

    result = runner.invoke(
        app,
        [
            "preview",
            str(plan_path),
            "--output",
            str(tmp_path / "r.html"),
            "--before-after",
            "--frames-dir",
            str(custom),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (custom / "shot_001_before.png").exists()
    assert (custom / "shot_001_after.png").exists()


def test_before_after_missing_video_fails_cleanly(tmp_path):
    import json

    plan_path = tmp_path / "ghost.censorplan.json"
    plan_path.write_text(
        json.dumps(
            {
                "pureframe_version": "0",
                "plan_version": 1,
                "input_metadata": {
                    "width": 16,
                    "height": 16,
                    "fps": "15/1",
                    "duration_seconds": 1.0,
                    "total_frames": 15,
                    "has_audio": False,
                    "audio_streams": [],
                    "subtitle_streams": [],
                    "container": "mp4",
                    "video_codec": "h264",
                    "pixel_format": "yuv420p",
                    "color_space": "unknown",
                    "is_hdr": False,
                },
                "config_snapshot": {"input_path": str(tmp_path / "ghost.mp4")},
                "shots": [
                    {
                        "index": 0,
                        "start_frame": 0,
                        "end_frame": 15,
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "frames": {},
                    }
                ],
                "verdicts": [
                    {
                        "shot_index": 0,
                        "category": "NUDITY_EXPLICIT",
                        "action": "BLACK_BOX",
                        "confidence": 0.9,
                        "boxes": None,
                        "reasoning": "test",
                    }
                ],
                "total_censored_frames": 15,
                "total_blur_frames": 0,
                "generated_at": "2026-09-10T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["preview", str(plan_path), "--before-after"], catch_exceptions=False
    )
    assert result.exit_code == 2
    assert "does not exist" in " ".join(result.output.split())
