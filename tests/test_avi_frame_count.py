"""AVI frame-count metadata lies: probe must report the real packet count.

An MPEG-4 AVI carries a stream duration one tick longer than the frames
it holds (nb_frames 61 for 60 packets on the fixture shape), so the
plan's keyframe sampler requests a phantom tail frame. Extraction drops
it, and a profile sampling two keyframes per shot hands pairwise
detectors a single frame - the motion-blob plugin then sees nothing to
difference and the shot silently passes as SAFE. The WebM that
regressed alongside it in the critic pass keeps its real count, so the
same clip flagged through the plugin on WebM and passed on AVI.

The end-to-end guard runs the real planner on a single-shot AVI with
the example motion-blob plugin (discovery patched, per the suite's
plugin-test pattern) and demands the PLUGIN_BOX verdict the safe
rendering used to swallow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pureframe.pipeline.probe import probe_video

EXAMPLES_SRC = (
    Path(__file__).parent.parent / "examples" / "pureframe-plugins-examples" / "src"
)


@pytest.fixture(scope="module")
def single_shot_avi(tmp_path_factory):
    """4 s of always-moving testsrc2 as MPEG-4 in AVI at 15 fps.

    Muxed with an audio track under -shortest, the shape that makes the
    AVI stream duration run one tick past its frames (nb_frames 61 for
    60 packets) - the phantom tail frame the probe must not trust.
    """
    path = tmp_path_factory.mktemp("avi") / "single_shot.avi"
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
            "testsrc2=duration=4:size=320x240:rate=15",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=4",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
        shell=False,
    )
    return path


def _real_packet_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
    )
    return int(result.stdout.strip())


def test_probe_reports_real_packet_count(single_shot_avi):
    meta = probe_video(single_shot_avi)
    real = _real_packet_count(single_shot_avi)
    assert meta.total_frames == real
    assert meta.total_frames >= 55, "fixture expects a ~60 frame clip"


def test_motionblob_flags_a_single_shot_avi(
    single_shot_avi, tmp_path, monkeypatch, mock_store
):
    sys.path.insert(0, str(EXAMPLES_SRC))
    try:
        from pureframe_plugins_examples.motionblob import MotionBlobDetector
    finally:
        sys.path.remove(str(EXAMPLES_SRC))

    from pureframe import plugin_api
    from pureframe.cli import generate_plan
    from pureframe.config import Config
    from pureframe.hardware import HardwareProfile
    from pureframe.pipeline.detect.nudity import NudityDetector
    from pureframe.pipeline.shots import Action, Category
    from pureframe.plugin_api import PluginRegistration

    registration = PluginRegistration(
        name="motionblob",
        cls=MotionBlobDetector,
        label_categories={"motion_blob": "MOTION_VISIBLE"},
        label_thresholds={"motion_blob": 0.45},
    )
    monkeypatch.setattr(plugin_api, "discover", lambda: {"motionblob": registration})
    monkeypatch.setattr(
        NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
    )

    config = Config.from_cli(
        input_path=single_shot_avi,
        output_path=tmp_path / "out.avi",
        profile=HardwareProfile.CPU,
        no_clip=True,
        no_audio=True,
        enabled_plugins=["motionblob"],
    )
    plan = generate_plan(config)

    assert plan.shots[0].end_frame <= plan.input_metadata.total_frames
    flagged = [v for v in plan.verdicts if v.action != Action.NONE]
    assert len(flagged) == 1, (
        "the moving single-shot AVI must flag through the plugin; a SAFE "
        "verdict here means the phantom tail frame starved the detector again"
    )
    assert flagged[0].category == Category.PLUGIN_BOX
    assert flagged[0].boxes, "the plugin verdict must carry tracked boxes"
