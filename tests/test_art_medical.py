"""Art and medical content profiles plus their scene-context gates.

``--content-type art`` and ``--content-type medical`` raise whole-run
bars (1.5x / 1.4x on top of the strictness preset); the museum/gallery
and medical/clinical CLIP contexts refine the same raise per shot, so a
documentary mixing paintings or surgery with real explicit scenes keeps
its strict bar exactly where the sexual context shows up.
"""

import os
import tempfile
from pathlib import Path

import pytest

from pureframe.config import Config, ContentType, Strictness
from pureframe.pipeline.detect.audio import AudioContext
from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.detect.scene_clip import PROMPT_SETS, ShotContext
from pureframe.pipeline.fuse import SCENE_CONTEXT_CONFIDENCE, fuse, scene_context_factor
from pureframe.pipeline.shots import Action, Category, Shot


@pytest.fixture
def dummy_file():
    tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tf.close()
    yield Path(tf.name)
    os.unlink(tf.name)


@pytest.fixture
def basic_shot():
    return Shot(index=0, start_frame=0, end_frame=240, start_time=0.0, end_time=10.0)


@pytest.fixture
def safe_audio_ctx():
    return AudioContext(
        moaning_score=0.0,
        sexual_audio_score=0.0,
        music_score=0.0,
        speech_score=0.0,
    )


def ctx(museum=0.0, medical=0.0, **kwargs) -> ShotContext:
    return ShotContext(
        explicit_act_score=kwargs.get("explicit_act", 0.0),
        implied_sex_score=kwargs.get("implied_sex", 0.0),
        kissing_score=kwargs.get("kissing", 0.0),
        safe_score=kwargs.get("safe", 0.9),
        museum_gallery_score=museum,
        medical_score=medical,
    )


def mid_detection(score=0.65):
    return [
        [Detection(label="FEMALE_BREAST_EXPOSED", score=score, box=(0, 0, 100, 100))]
    ]


class TestProfiles:
    def test_art_multiplier(self, dummy_file):
        config = Config(
            input_path=dummy_file,
            content_type=ContentType.ART,
            strictness=Strictness.MEDIUM,
        )
        nudity, _, _ = config.get_effective_thresholds()
        assert nudity == pytest.approx(0.55 * 1.5)

    def test_medical_multiplier(self, dummy_file):
        config = Config(
            input_path=dummy_file,
            content_type=ContentType.MEDICAL,
            strictness=Strictness.MEDIUM,
        )
        nudity, _, _ = config.get_effective_thresholds()
        assert nudity == pytest.approx(0.55 * 1.4)

    def test_content_type_changes_the_hash(self, dummy_file):
        live = Config(input_path=dummy_file)
        art = Config(input_path=dummy_file, content_type=ContentType.ART)
        assert live.config_hash != art.config_hash

    def test_art_alone_spares_mid_score_paintings(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        # 0.55 * 1.5 = 0.825: a 0.65 painted-figure detection passes.
        config = Config(input_path=dummy_file, content_type=ContentType.ART)
        verdict = fuse(basic_shot, mid_detection(), ctx(), safe_audio_ctx, config)
        assert verdict.action == Action.NONE

    def test_live_action_still_flags_the_same_detection(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        verdict = fuse(basic_shot, mid_detection(), ctx(), safe_audio_ctx, config)
        assert verdict.action == Action.BLACK_BOX


class TestSceneGates:
    def test_museum_context_raises(self, dummy_file):
        config = Config(input_path=dummy_file)
        factor = scene_context_factor(
            ctx(museum=SCENE_CONTEXT_CONFIDENCE + 0.1), config
        )
        assert factor == 1.4

    def test_medical_context_raises(self, dummy_file):
        config = Config(input_path=dummy_file)
        factor = scene_context_factor(
            ctx(medical=SCENE_CONTEXT_CONFIDENCE + 0.1), config
        )
        assert factor == 1.4

    def test_museum_gated_shot_stays_unflagged_at_defaults(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        verdict = fuse(
            basic_shot,
            mid_detection(),
            ctx(museum=SCENE_CONTEXT_CONFIDENCE + 0.1),
            safe_audio_ctx,
            config,
        )
        assert verdict.action == Action.NONE

    def test_medical_gated_shot_stays_unflagged_at_defaults(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        verdict = fuse(
            basic_shot,
            mid_detection(),
            ctx(medical=SCENE_CONTEXT_CONFIDENCE + 0.1),
            safe_audio_ctx,
            config,
        )
        assert verdict.action == Action.NONE

    def test_real_nudity_in_the_same_documentary_still_flags(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        # The doc cuts to an actual explicit scene: strong score clears
        # even the raised bar.
        config = Config(input_path=dummy_file)
        verdict = fuse(
            basic_shot,
            mid_detection(score=0.95),
            ctx(museum=SCENE_CONTEXT_CONFIDENCE + 0.1),
            safe_audio_ctx,
            config,
        )
        assert verdict.action == Action.BLACK_BOX

    def test_sexual_context_switches_the_museum_gate_off(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        scene = ctx(museum=SCENE_CONTEXT_CONFIDENCE + 0.1, explicit_act=0.7, safe=0.0)
        factor = scene_context_factor(scene, config)
        assert factor == 1.0
        audio = AudioContext(
            moaning_score=0.9, sexual_audio_score=0.7, music_score=0.0, speech_score=0.0
        )
        verdict = fuse(basic_shot, mid_detection(score=0.30), scene, audio, config)
        assert verdict.category == Category.SEXUAL_ACT_VISIBLE


class TestPrompts:
    def test_context_categories_exist(self):
        assert "museum_gallery" in PROMPT_SETS
        assert "medical_clinical" in PROMPT_SETS
        assert len(PROMPT_SETS["museum_gallery"]) >= 3
        assert len(PROMPT_SETS["medical_clinical"]) >= 3


class TestCorpus:
    def test_art_and_medical_entries_exist(self):
        from pureframe.eval import SYNTHETIC_SCENARIOS

        ids = {s["id"] for s in SYNTHETIC_SCENARIOS}
        assert "ART-001" in ids
        assert "MED-001" in ids

    def test_new_frame_types_render(self):
        from pureframe.eval import _generate_synthetic_frame

        for frame_type in ("art_painting", "medical_surgical"):
            frame = _generate_synthetic_frame(frame_type)
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == "uint8"
