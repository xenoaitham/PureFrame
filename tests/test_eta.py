"""ETA estimates: calibration math, formatting, and CLI surfacing."""

import pytest
from typer.testing import CliRunner

from pureframe.cli import app
from pureframe.eta import (
    ANALYSIS_SPF,
    COPY_SPF,
    REENCODE_SPF,
    estimate_analysis_seconds,
    estimate_render_seconds,
    format_duration,
)
from pureframe.hardware import HardwareProfile

runner = CliRunner()


class TestFormatDuration:
    def test_seconds_under_a_minute(self):
        assert format_duration(0) == "≈ 0 s"
        assert format_duration(3.2) == "≈ 3 s"
        assert format_duration(59.9) == "≈ 60 s"

    def test_minutes(self):
        assert format_duration(60) == "≈ 1.0 min"
        assert format_duration(212) == "≈ 3.5 min"

    def test_hours(self):
        assert format_duration(3600) == "≈ 1 h 00 min"
        assert format_duration(4529) == "≈ 1 h 15 min"

    def test_negative_clamps_to_zero(self):
        assert format_duration(-5) == "≈ 0 s"


class TestEstimates:
    def test_analysis_is_per_frame_constant(self):
        assert estimate_analysis_seconds(HardwareProfile.MEDIUM, 900) == pytest.approx(
            ANALYSIS_SPF[HardwareProfile.MEDIUM] * 900
        )

    def test_none_profile_falls_back_to_cpu(self):
        assert estimate_analysis_seconds(None, 100) == pytest.approx(
            ANALYSIS_SPF[HardwareProfile.CPU] * 100
        )

    def test_render_splits_flagged_and_clean(self):
        # 900 frames, 90 flagged: 90 re-encode + 810 copy.
        expected = REENCODE_SPF[HardwareProfile.HIGH] * 90 + COPY_SPF * 810
        assert estimate_render_seconds(HardwareProfile.HIGH, 900, 90) == pytest.approx(
            expected
        )

    def test_render_fully_flagged_is_a_full_reencode(self):
        assert estimate_render_seconds(HardwareProfile.LOW, 900, 900) == pytest.approx(
            REENCODE_SPF[HardwareProfile.LOW] * 900
        )

    def test_render_clamps_out_of_range_flagged(self):
        assert estimate_render_seconds(HardwareProfile.HIGH, 100, 500) == pytest.approx(
            REENCODE_SPF[HardwareProfile.HIGH] * 100
        )
        assert estimate_render_seconds(HardwareProfile.HIGH, 100, -20) == pytest.approx(
            COPY_SPF * 100
        )

    def test_calibrations_match_the_published_bench_magnitudes(self):
        # 30 s @ 30 fps reference clip: analysis ≈ 3 s on CPU-ish, and a
        # fully-flagged GPU render ≈ 4 s. Guards against silent constant
        # drift; real medians live in BENCHMARKS.md.
        assert estimate_analysis_seconds(HardwareProfile.CPU, 900) < 5
        assert 3.5 < estimate_render_seconds(HardwareProfile.LOW, 900, 900) < 5


def test_plan_prints_analysis_estimate(synthetic_video, tmp_path):
    result = runner.invoke(
        app,
        [
            "plan",
            str(synthetic_video),
            "--output",
            str(tmp_path / "p.json"),
            "--no-clip",
            "--no-audio",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "Analysis estimate:" in result.output
    assert "≈" in result.output
