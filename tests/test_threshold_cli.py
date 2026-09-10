"""Per-category threshold flags: CLI wiring and the densify/fuse alignment.

``--threshold-nudity/--threshold-clip/--threshold-audio`` (and ``--threshold``
as the nudity alias) replace the strictness preset's value for that
category; ``--thresholds file.json`` does the same from a file, with flags
winning. Before this, ``--threshold`` was silently ignored unless
``--strictness custom`` was also passed.

The plan loop must also filter densified boxes at the *effective* nudity
threshold fuse() flags on. It used the raw default (0.55) instead, so a
shot flagged at 0.45 under ``--strictness high`` (preset 0.35) came out as
a BLACK_BOX verdict with no boxes — nothing rendered.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pureframe.cli import app, generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.shots import Action
from tests.conftest import frame_has_marker

runner = CliRunner()


def _captured_config(argv):
    """Run *argv* with plan generation stubbed; return the Config it received."""
    with (
        patch("pureframe.cli.generate_plan") as gen,
        patch("pureframe.cli.process_file") as proc,
    ):
        gen.return_value = MagicMock()
        result = runner.invoke(app, argv, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        call = gen.call_args or proc.call_args
        return call.args[0]


class TestCliFlags:
    def test_threshold_alias_sets_nudity_override(self, synthetic_video):
        config = _captured_config(["plan", str(synthetic_video), "--threshold", "0.4"])
        assert config.threshold_overrides == {"nudity": 0.4}
        assert config.get_effective_thresholds()[0] == pytest.approx(0.4)

    def test_no_flags_means_no_overrides(self, synthetic_video):
        config = _captured_config(["plan", str(synthetic_video)])
        assert config.threshold_overrides == {}

    def test_explicit_nudity_flag_wins_over_alias(self, synthetic_video):
        config = _captured_config(
            [
                "process",
                str(synthetic_video),
                "--threshold",
                "0.4",
                "--threshold-nudity",
                "0.3",
            ]
        )
        assert config.threshold_overrides == {"nudity": 0.3}

    def test_each_category_flag(self, synthetic_video):
        config = _captured_config(
            [
                "process",
                str(synthetic_video),
                "--threshold-nudity",
                "0.3",
                "--threshold-clip",
                "0.6",
                "--threshold-audio",
                "0.7",
                "--strictness",
                "high",
            ]
        )
        assert config.threshold_overrides == {"nudity": 0.3, "clip": 0.6, "audio": 0.7}
        assert config.get_effective_thresholds() == pytest.approx((0.3, 0.6, 0.7))

    def test_thresholds_file_merged_with_flags_winning(self, synthetic_video, tmp_path):
        f = tmp_path / "thresholds.json"
        f.write_text('{"nudity": 0.45, "clip": 0.65}', encoding="utf-8")
        config = _captured_config(
            [
                "plan",
                str(synthetic_video),
                "--thresholds",
                str(f),
                "--threshold-nudity",
                "0.25",
            ]
        )
        assert config.threshold_overrides == {"nudity": 0.25, "clip": 0.65}

    def test_out_of_range_flag_rejected_by_typer(self, synthetic_video):
        result = runner.invoke(
            app, ["plan", str(synthetic_video), "--threshold", "1.5"]
        )
        assert result.exit_code != 0
        assert "1.5" in result.output

    def test_bad_thresholds_file_is_a_clean_cli_error(self, synthetic_video, tmp_path):
        f = tmp_path / "thresholds.json"
        f.write_text("[0.4]", encoding="utf-8")
        result = runner.invoke(
            app, ["plan", str(synthetic_video), "--thresholds", str(f)]
        )
        assert result.exit_code == 2
        # Rich wraps the error panel; macOS's longer /private/var paths move
        # the wrap point, so match on whitespace-collapsed text.
        assert "expected a JSON object" in " ".join(result.output.split())

    def test_unknown_category_in_file_is_a_clean_cli_error(
        self, synthetic_video, tmp_path
    ):
        f = tmp_path / "thresholds.json"
        f.write_text('{"kiss": 0.4}', encoding="utf-8")
        result = runner.invoke(
            app, ["process", str(synthetic_video), "--thresholds", str(f)]
        )
        assert result.exit_code == 2
        assert "unknown threshold category" in " ".join(result.output.split())

    def test_zero_threshold_is_a_clean_cli_error(self, synthetic_video):
        result = runner.invoke(
            app, ["plan", str(synthetic_video), "--threshold-clip", "0"]
        )
        assert result.exit_code == 2
        assert "must be in (0, 1]" in " ".join(result.output.split())


def test_densify_keeps_boxes_flagged_by_a_lower_effective_threshold(
    three_shot_video, tmp_path, monkeypatch
):
    """A 0.45-score detection under --strictness high must keep its boxes."""
    import pureframe.cli
    from pureframe.pipeline.detect.nudity import Detection, NudityDetector

    def mocked_detect_batch(self, frames_bgr):
        return [
            [
                Detection(
                    label="FEMALE_BREAST_EXPOSED", score=0.45, box=(60, 40, 260, 200)
                )
            ]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    monkeypatch.setattr(NudityDetector, "detect_batch", mocked_detect_batch)

    original = pureframe.cli.get_settings

    def dense(profile):
        s = original(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense)

    config = Config.from_cli(
        input_path=three_shot_video,
        output_path=tmp_path / "out.mp4",
        profile=HardwareProfile.CPU,
        strictness="high",  # nudity preset 0.35 < 0.45 < raw default 0.55
        no_clip=True,
        no_audio=True,
    )
    plan = generate_plan(config)

    flagged = [v for v in plan.verdicts if v.action != Action.NONE]
    assert [v.shot_index for v in flagged] == [1]
    assert flagged[0].boxes, "flagged shot lost its boxes to the raw 0.55 filter"
