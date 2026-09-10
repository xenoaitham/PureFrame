"""`pureframe bench --real`: per-phase benchmarks of a user-supplied file.

The synthetic bench numbers are reproducible but not representative of real
content; this mode times the full `process` flow on a file the user owns
and appends privacy-safe JSON lines (file identified by SHA-256 + basic
metadata, never the path) to a local JSONL, so runs accumulate and the
result can be shared without leaking filenames.
"""

import json

import pytest
from typer.testing import CliRunner

from pureframe.bench import run_benchmark_real, summarize_real_records
from tests.conftest import generate_three_shot_clip

runner = CliRunner()


@pytest.fixture
def bench_clip(tmp_path):
    clip = tmp_path / "mine.mp4"
    return generate_three_shot_clip(clip, ["-c:v", "libx264", "-crf", "28"])


def test_records_appended_and_privacy_safe(bench_clip, tmp_path, monkeypatch):
    import pureframe.cli
    from pureframe.pipeline.detect.nudity import Detection, NudityDetector

    def mocked_detect_batch(self, frames_bgr):
        return [
            [
                Detection(
                    label="FEMALE_BREAST_EXPOSED", score=0.99, box=(60, 40, 260, 200)
                )
            ]
        ] * len(frames_bgr)

    monkeypatch.setattr(NudityDetector, "detect_batch", mocked_detect_batch)

    original = pureframe.cli.get_settings

    def dense(profile, **kwargs):
        s = original(profile, **kwargs)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense)

    jsonl = tmp_path / "records.jsonl"
    records = run_benchmark_real(bench_clip, profiles=["CPU"], reps=1, jsonl_path=jsonl)

    assert len(records) == 1
    record = records[0]
    assert record["kind"] == "real-bench"
    assert record["profile"] == "CPU"
    assert record["total_seconds"] > 0
    assert record["flagged_shots"] >= 1
    assert record["phase_seconds"]["render"] > 0

    # Privacy: the JSON identifies the file by hash + metadata, never a path.
    assert str(bench_clip) not in json.dumps(record)
    assert "file" in record and len(record["file"]["sha256"]) == 64

    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "real-bench"


def test_appends_across_runs(bench_clip, tmp_path):
    jsonl = tmp_path / "records.jsonl"
    run_benchmark_real(bench_clip, profiles=["CPU"], reps=1, jsonl_path=jsonl)
    records = run_benchmark_real(bench_clip, profiles=["CPU"], reps=2, jsonl_path=jsonl)

    assert len(records) == 2
    assert len(jsonl.read_text().strip().splitlines()) == 3

    summary = summarize_real_records(records)
    assert "| CPU | 2 |" in summary
    assert "sha256" in summary


def test_cli_bench_real(bench_clip, tmp_path):
    from pureframe.cli import app

    jsonl = tmp_path / "records.jsonl"
    result = runner.invoke(
        app,
        [
            "bench",
            "--real",
            str(bench_clip),
            "--profiles",
            "CPU",
            "--jsonl",
            str(jsonl),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert jsonl.exists()
    assert "Records appended" in result.output
    # Synthetic-only flags stay synthetic-only: no clip artifacts appear.
    assert not (tmp_path / "pureframe_bench_clip_1280x720.mp4").exists()
