"""Tests for the plugin box-provider branch of ``fuse()`` and the
per-category threshold integration for plugin categories."""

from __future__ import annotations

import pytest

from pureframe.config import Config
from pureframe.pipeline.detect.audio import AudioContext
from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.detect.scene_clip import ShotContext
from pureframe.pipeline.fuse import fuse
from pureframe.pipeline.shots import Action, Category, Shot

WEAPON = "WEAPON_VISIBLE"
LONG_SHOT = Shot(index=0, start_frame=0, end_frame=100, start_time=0.0, end_time=4.0)
SHORT_SHOT = Shot(index=0, start_frame=0, end_frame=50, start_time=0.0, end_time=2.0)

NEUTRAL_AUDIO = AudioContext(
    moaning_score=0, sexual_audio_score=0, music_score=0, speech_score=0
)


def scene(**overrides) -> ShotContext:
    values = dict(
        explicit_act_score=0, implied_sex_score=0, kissing_score=0, safe_score=1
    )
    values.update(overrides)
    return ShotContext(**values)


def weapon_dets(score: float, per_frame: int = 1) -> dict[str, list[Detection]]:
    frames = [
        [Detection(label="pistol", score=score, box=(10, 10, 40, 40))]
        for _ in range(per_frame)
    ]
    return {WEAPON: frames}


@pytest.fixture
def weapon_registry(monkeypatch):
    """Make the fixture plugin's WEAPON_VISIBLE category discoverable so
    the threshold_overrides validator accepts it."""
    from pureframe import plugin_api
    from pureframe.plugin_api import PluginRegistration

    def fake_discover():
        return {
            "fixture": PluginRegistration(
                name="fixture",
                cls=object,
                label_categories={"motion_blob": WEAPON},
                label_thresholds={"motion_blob": 0.6},
            )
        }

    monkeypatch.setattr(plugin_api, "discover", fake_discover)
    return fake_discover


def test_plugin_detection_flags_black_box():
    verdict = fuse(
        LONG_SHOT,
        [],
        scene(),
        NEUTRAL_AUDIO,
        Config(input_path="d", output_path="d"),
        plugin_detections=weapon_dets(0.9),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert verdict.category == Category.PLUGIN_BOX
    assert verdict.action == Action.BLACK_BOX
    assert verdict.plugin_category == WEAPON
    assert verdict.confidence == 0.9
    assert verdict.boxes is None  # boxes attach in the densify pass


def test_below_threshold_stays_safe():
    verdict = fuse(
        LONG_SHOT,
        [],
        scene(),
        NEUTRAL_AUDIO,
        Config(input_path="d", output_path="d"),
        plugin_detections=weapon_dets(0.59),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert verdict.category == Category.SAFE
    assert verdict.action == Action.NONE
    assert verdict.plugin_category is None


def test_never_gates_on_audio():
    """A box provider fires on visual evidence alone: silence must not
    block it and loud sexual-audio must not be required."""
    loud_audio = AudioContext(
        moaning_score=0.9, sexual_audio_score=0.9, music_score=0, speech_score=0
    )
    config = Config(input_path="d", output_path="d")
    silent = fuse(
        LONG_SHOT,
        [],
        scene(),
        NEUTRAL_AUDIO,
        config,
        plugin_detections=weapon_dets(0.9),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    loud = fuse(
        LONG_SHOT,
        [],
        scene(),
        loud_audio,
        config,
        plugin_detections=weapon_dets(0.9),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert silent.category == loud.category == Category.PLUGIN_BOX
    assert silent.action == loud.action == Action.BLACK_BOX


def test_never_downgrades_nudity_verdict():
    dets = [[Detection(label="FEMALE_BREAST_EXPOSED", score=0.99, box=(0, 0, 10, 10))]]
    verdict = fuse(
        LONG_SHOT,
        dets,
        scene(),
        NEUTRAL_AUDIO,
        Config(input_path="d", output_path="d"),
        plugin_detections=weapon_dets(0.9),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert verdict.category == Category.NUDITY_EXPLICIT
    assert verdict.plugin_category is None


def test_upgrades_kiss_light_to_black_box():
    verdict = fuse(
        SHORT_SHOT,  # 2.0 s duration -> light kiss on its own
        [],
        scene(kissing_score=0.6, safe_score=0),
        NEUTRAL_AUDIO,
        Config(input_path="d", output_path="d"),
        plugin_detections=weapon_dets(0.9),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert verdict.category == Category.PLUGIN_BOX
    assert verdict.action == Action.BLACK_BOX


def test_plugin_branch_outranks_kiss_branches():
    """The plugin branch runs before the kiss branches: a 4.0 s shot that
    would be KISS_INTENSE on its own keeps its censoring when a plugin
    also fires, just attributed to the plugin's box category."""
    verdict = fuse(
        LONG_SHOT,  # 4.0 s duration -> intense kiss on its own
        [],
        scene(kissing_score=0.6, safe_score=0),
        NEUTRAL_AUDIO,
        Config(input_path="d", output_path="d"),
        plugin_detections=weapon_dets(0.9),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert verdict.category == Category.PLUGIN_BOX
    assert verdict.action == Action.BLACK_BOX


def test_kiss_survives_when_plugin_does_not_fire():
    verdict = fuse(
        LONG_SHOT,
        [],
        scene(kissing_score=0.6, safe_score=0),
        NEUTRAL_AUDIO,
        Config(input_path="d", output_path="d"),
        plugin_detections=weapon_dets(0.3),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert verdict.category == Category.KISS_INTENSE


def test_threshold_override_lowers_plugin_threshold(weapon_registry):
    config = Config(
        input_path="d",
        output_path="d",
        threshold_overrides={WEAPON: 0.3},
    )
    verdict = fuse(
        LONG_SHOT,
        [],
        scene(),
        NEUTRAL_AUDIO,
        config,
        plugin_detections=weapon_dets(0.35),
        plugin_threshold_bases={WEAPON: 0.6},
    )
    assert verdict.category == Category.PLUGIN_BOX


def test_strict_mode_eases_plugin_threshold():
    """The 0.85 strict factor lowers thresholds exactly as it does for
    nudity: 0.55 stays safe at the 0.64 base without --strict and fires
    with it."""
    config = Config(input_path="d", output_path="d")
    verdict_55 = fuse(
        LONG_SHOT,
        [],
        scene(),
        NEUTRAL_AUDIO,
        config,
        plugin_detections=weapon_dets(0.55),
        plugin_threshold_bases={WEAPON: 0.64},
    )
    verdict_55_strict = fuse(
        LONG_SHOT,
        [],
        scene(),
        NEUTRAL_AUDIO,
        config,
        strict_mode=True,
        plugin_detections=weapon_dets(0.55),
        plugin_threshold_bases={WEAPON: 0.64},
    )
    assert verdict_55.category == Category.SAFE
    assert verdict_55_strict.category == Category.PLUGIN_BOX


def test_no_plugin_detections_changes_nothing():
    verdict = fuse(
        LONG_SHOT, [], scene(), NEUTRAL_AUDIO, Config(input_path="d", output_path="d")
    )
    assert verdict.category == Category.SAFE


def test_content_type_multiplier_applies_to_plugin_base():
    config = Config(input_path="d", output_path="d", content_type="anime")
    effective = config.get_effective_plugin_thresholds({WEAPON: 0.6})
    # anime multiplier 1.4
    assert effective[WEAPON] == pytest.approx(0.84)


def test_effective_plugin_thresholds_respect_overrides():
    config = Config(input_path="d", output_path="d", content_type="animation")
    effective = config.get_effective_plugin_thresholds({WEAPON: 0.6, "OTHER": 0.9})
    assert effective[WEAPON] == pytest.approx(0.78)  # 0.6 * 1.3
    assert effective["OTHER"] == pytest.approx(0.99)  # 0.9 * 1.3, capped


def test_effective_plugin_thresholds_empty_without_categories():
    config = Config(input_path="d", output_path="d")
    assert config.get_effective_plugin_thresholds({}) == {}


def test_empty_overrides_validator_short_circuits_without_plugins():
    config = Config(input_path="d", output_path="d")
    assert config.threshold_overrides == {}


def test_unknown_category_still_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="unknown threshold category"):
        Config(
            input_path="d",
            output_path="d",
            threshold_overrides={"not_a_category": 0.5},
        )


def test_plugin_category_accepted_by_validator(weapon_registry):
    config = Config(
        input_path="d",
        output_path="d",
        threshold_overrides={WEAPON: 0.4},
    )
    assert config.threshold_overrides == {WEAPON: 0.4}


def test_range_validation_applies_to_plugin_categories(weapon_registry):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must be in"):
        Config(
            input_path="d",
            output_path="d",
            threshold_overrides={WEAPON: 1.5},
        )


def test_plugin_box_category_serializes_round_trip():
    """Plans stay portable: a PLUGIN_BOX verdict serializes and reloads
    through the plain pydantic model, on this machine without any plugin
    installed."""
    from pureframe.pipeline.shots import ShotVerdict

    verdict = ShotVerdict(
        shot_index=3,
        category=Category.PLUGIN_BOX,
        action=Action.BLACK_BOX,
        confidence=0.82,
        plugin_category=WEAPON,
        reasoning="plugin",
    )
    data = verdict.model_dump_json()
    restored = ShotVerdict.model_validate_json(data)
    assert restored.category == Category.PLUGIN_BOX
    assert restored.plugin_category == WEAPON
