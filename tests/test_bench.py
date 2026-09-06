"""Tests for the `pureframe bench` harness (no models loaded)."""

import json
import subprocess

from pureframe.bench import (
    _parse_flagged,
    _parse_timers_payload,
    capture_environment,
    generate_bench_clip,
    report_to_markdown,
)


def test_generate_bench_clip_is_valid_media(tmp_path):
    clip = generate_bench_clip(
        tmp_path / "clip.mp4", duration=2.0, width=320, height=240
    )
    assert clip.exists() and clip.stat().st_size > 0

    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(clip),
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    data = json.loads(out.stdout)
    assert abs(float(data["format"]["duration"]) - 2.0) < 0.5
    types = {s["codec_type"] for s in data["streams"]}
    assert "video" in types
    assert "audio" in types, "silent audio track must exist so audio classify is timed"


def test_parse_timers_payload():
    payload = {"phases": {"render": {"seconds": 1.5, "calls": 1}}, "flagged_shots": 3}
    line = f"PUREFRAME_TIMERS {json.dumps(payload)}"
    parsed = _parse_timers_payload(f"noise\n{line}\nmore noise")
    assert parsed == payload
    assert _parse_flagged(parsed, "") == 3


def test_parse_timers_payload_missing():
    assert _parse_timers_payload("no timers here") is None
    assert _parse_flagged(None, "Flagged 7 shots for censoring.") == 7


def test_capture_environment_fields():
    env = capture_environment()
    for key in ("pureframe", "python", "platform", "cpu_count", "ffmpeg"):
        assert key in env
    assert isinstance(env["cpu_count"], int)


def test_report_to_markdown_rows():
    report = {
        "environment": {
            "pureframe": "0.1.0b16",
            "platform": "test",
            "cpu_count": 4,
            "ffmpeg": "ffmpeg v7",
        },
        "clip": "/tmp/bench_clip.mp4",
        "profiles": {
            "CPU": {
                "total_seconds_median": 12.5,
                "flagged_median": 2,
                "phase_seconds_median": {"render": 5.0, "extract": 3.0},
            }
        },
    }
    md = report_to_markdown(report)
    assert "| bench_clip.mp4 | CPU | 12.5s | 2 |" in md
    assert "render 5.0s" in md
