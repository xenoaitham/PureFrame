"""Plugin CLI surface and pipeline wiring (slice 3).

``pureframe plugins list`` shows what is discovered; ``--enable-plugin``
(repeatable, plan/process only) opts a video run into a plugin's
categories; the enabled set folds into config_hash so cache semantics
hold; and generate_plan runs plugin detectors on the same keyframes as
the nudity detector, with PLUGIN_BOX verdicts densified through the
plugin that flagged them.
"""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from pureframe.cli import _plugin_runtimes, app, generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.shots import Action, Category
from pureframe.plugin_api import PluginRegistration
from tests.conftest import frame_has_marker

runner = CliRunner()

WEAPON = "WEAPON_VISIBLE"
BOX = (60, 40, 260, 200)


class WeaponDetector:
    """Fixture plugin: flags the marker frame with a weapon box."""

    label_categories = {"pistol": WEAPON}
    label_thresholds = {"pistol": 0.6}

    def __init__(self, settings):
        self.settings = settings
        self.unloaded = False

    def detect_batch(self, frames_bgr):
        return [
            [Detection(label="pistol", score=0.9, box=BOX)]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    def unload(self):
        self.unloaded = True


def _fixture_registration(cls=WeaponDetector) -> PluginRegistration:
    return PluginRegistration(
        name="weapons",
        cls=cls,
        label_categories={"pistol": WEAPON},
        label_thresholds={"pistol": 0.6},
    )


def _patched_registry(monkeypatch, registration):
    from pureframe import plugin_api

    monkeypatch.setattr(
        plugin_api, "discover", lambda: {registration.name: registration}
    )


class TestPluginsListCommand:
    def test_empty_registry_prints_hint(self, monkeypatch):
        from pureframe import plugin_api

        monkeypatch.setattr(plugin_api, "discover", lambda: {})
        result = runner.invoke(app, ["plugins", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No plugins discovered" in result.output

    def test_lists_names_and_categories(self, monkeypatch):
        _patched_registry(monkeypatch, _fixture_registration())
        result = runner.invoke(app, ["plugins", "list"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "weapons" in result.output
        assert WEAPON in result.output
        assert "pistol" in result.output

    def test_list_succeeds_with_real_registry(self):
        # The real environment has no plugins installed; the command must
        # still exit cleanly.
        result = runner.invoke(app, ["plugins", "list"], catch_exceptions=False)
        assert result.exit_code == 0


class TestEnablePluginFlag:
    def test_unknown_name_rejected(self, synthetic_video):
        result = runner.invoke(
            app,
            ["plan", str(synthetic_video), "--enable-plugin", "nope"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "unknown plugin(s): nope" in result.output

    def test_known_name_reaches_config(self, synthetic_video, monkeypatch):
        _patched_registry(monkeypatch, _fixture_registration())
        with (
            patch("pureframe.cli.generate_plan") as gen,
            patch("pureframe.cli.process_file") as proc,
        ):
            gen.return_value = MagicMock()
            proc.return_value = MagicMock()
            result = runner.invoke(
                app,
                [
                    "plan",
                    str(synthetic_video),
                    "--enable-plugin",
                    "weapons",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            config = gen.call_args.args[0]
        assert config.enabled_plugins == ["weapons"]

    def test_flag_absent_leaves_set_empty(self, synthetic_video):
        with (
            patch("pureframe.cli.generate_plan") as gen,
            patch("pureframe.cli.process_file") as proc,
        ):
            gen.return_value = MagicMock()
            proc.return_value = MagicMock()
            result = runner.invoke(
                app, ["plan", str(synthetic_video)], catch_exceptions=False
            )
            assert result.exit_code == 0, result.output
            config = gen.call_args.args[0]
        assert config.enabled_plugins == []


class TestConfigHash:
    def test_enabled_plugins_change_the_hash(self, synthetic_video):
        base = Config(input_path=synthetic_video, output_path=synthetic_video)
        with_plugin = Config(
            input_path=synthetic_video,
            output_path=synthetic_video,
            enabled_plugins=["weapons"],
        )
        assert base.config_hash != with_plugin.config_hash

    def test_order_normalized_in_hash(self, synthetic_video):
        a = Config(
            input_path=synthetic_video,
            output_path=synthetic_video,
            enabled_plugins=["weapons", "other"],
        )
        b = Config(
            input_path=synthetic_video,
            output_path=synthetic_video,
            enabled_plugins=["other", "weapons"],
        )
        assert a.config_hash == b.config_hash


class TestPluginRuntimes:
    def test_empty_when_disabled(self, tmp_path):
        config = Config(input_path=tmp_path / "x.mp4")
        assert _plugin_runtimes(config, settings=None) == []

    def test_unknown_name_raises(self, tmp_path):
        config = Config(input_path=tmp_path / "x.mp4", enabled_plugins=["ghost"])
        import pytest

        with pytest.raises(ValueError, match="unknown plugin 'ghost'"):
            _plugin_runtimes(config, settings=None)

    def test_constructs_enabled_instances_in_order(self, tmp_path, monkeypatch):
        _patched_registry(monkeypatch, _fixture_registration())
        config = Config(input_path=tmp_path / "x.mp4", enabled_plugins=["weapons"])
        runtimes = _plugin_runtimes(config, settings=None)
        assert [reg.name for reg, _ in runtimes] == ["weapons"]
        assert isinstance(runtimes[0][1], WeaponDetector)


class TestPlanIntegration:
    def _run_plan(self, three_shot_video, tmp_path, monkeypatch):
        """generate_plan with a silent nudity detector and the fixture
        weapon plugin enabled."""
        import pureframe.cli
        from pureframe.pipeline.detect.nudity import NudityDetector

        monkeypatch.setattr(
            NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
        )
        # Sample densely so the marker frame is among the keyframes.
        original = pureframe.cli.get_settings

        def dense(profile, **kwargs):
            s = original(profile)
            s.sample_keyframes_per_shot = 10
            return s

        monkeypatch.setattr(pureframe.cli, "get_settings", dense)
        _patched_registry(monkeypatch, _fixture_registration())

        config = Config.from_cli(
            input_path=three_shot_video,
            output_path=tmp_path / "out.mp4",
            profile=HardwareProfile.CPU,
            no_clip=True,
            no_audio=True,
            enabled_plugins=["weapons"],
        )
        return generate_plan(config)

    def test_plugin_flags_only_its_shot(self, three_shot_video, tmp_path, monkeypatch):
        plan = self._run_plan(three_shot_video, tmp_path, monkeypatch)

        flagged = [v for v in plan.verdicts if v.action != Action.NONE]
        assert len(flagged) == 1
        verdict = flagged[0]
        assert verdict.category == Category.PLUGIN_BOX
        assert verdict.plugin_category == WEAPON
        assert verdict.action == Action.BLACK_BOX

    def test_plugin_verdict_carries_tracked_boxes(
        self, three_shot_video, tmp_path, monkeypatch
    ):
        plan = self._run_plan(three_shot_video, tmp_path, monkeypatch)

        flagged = [v for v in plan.verdicts if v.action != Action.NONE][0]
        assert flagged.boxes, "plugin verdict must densify into tracked boxes"
        middle = plan.shots[flagged.shot_index]
        box_frames = {b.frame_idx for b in flagged.boxes}
        assert box_frames <= set(range(middle.start_frame, middle.end_frame))
        for box in flagged.boxes:
            assert (box.x2 - box.x1) > 0 and (box.y2 - box.y1) > 0

    def test_unflagged_shots_stay_safe(self, three_shot_video, tmp_path, monkeypatch):
        plan = self._run_plan(three_shot_video, tmp_path, monkeypatch)
        safe = [v for v in plan.verdicts if v.action == Action.NONE]
        assert {v.shot_index for v in safe} == {0, 2}

    def test_disabled_plugin_changes_nothing(
        self, three_shot_video, tmp_path, monkeypatch
    ):
        from pureframe.pipeline.detect.nudity import NudityDetector

        monkeypatch.setattr(
            NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
        )
        # Registry knows the plugin, but the run does not enable it.
        _patched_registry(monkeypatch, _fixture_registration())

        config = Config.from_cli(
            input_path=three_shot_video,
            output_path=tmp_path / "out.mp4",
            profile=HardwareProfile.CPU,
            no_clip=True,
            no_audio=True,
        )
        plan = generate_plan(config)
        assert all(v.action == Action.NONE for v in plan.verdicts)
