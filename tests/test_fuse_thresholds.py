"""Tests for the fusion engine with effective thresholds."""

import os
import tempfile
from pathlib import Path

import pytest

from pureframe.config import Config, ContentType, Strictness
from pureframe.pipeline.detect.audio import AudioContext
from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.detect.scene_clip import ShotContext
from pureframe.pipeline.fuse import NUDITY_EXPLICIT_LABELS, NUDITY_PARTIAL_LABELS, fuse
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
def safe_scene_ctx():
    return ShotContext(
        safe_score=0.95,
        explicit_act_score=0.0,
        implied_sex_score=0.0,
        kissing_score=0.0,
    )


@pytest.fixture
def safe_audio_ctx():
    return AudioContext(
        moaning_score=0.0,
        sexual_audio_score=0.0,
        music_score=0.0,
        speech_score=0.0,
    )


class TestFuseWithContentTypes:
    def test_live_action_detects_nudity(
        self, dummy_file, basic_shot, safe_scene_ctx, safe_audio_ctx
    ):
        config = Config(
            input_path=dummy_file,
            content_type=ContentType.LIVE_ACTION,
            strictness=Strictness.MEDIUM,
        )
        dets = [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.70, box=(0, 0, 100, 100))]
        ]

        verdict = fuse(basic_shot, dets, safe_scene_ctx, safe_audio_ctx, config)
        assert verdict.action == Action.BLACK_BOX
        assert verdict.category == Category.NUDITY_EXPLICIT

    def test_anime_misses_moderate_nudity(
        self, dummy_file, basic_shot, safe_scene_ctx, safe_audio_ctx
    ):
        """Anime content type has 1.4x multiplier, so 0.55 * 1.4 = 0.77 threshold.
        A 0.70 score should NOT trigger detection."""
        config = Config(
            input_path=dummy_file,
            content_type=ContentType.ANIME,
            strictness=Strictness.MEDIUM,
        )
        dets = [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.70, box=(0, 0, 100, 100))]
        ]

        verdict = fuse(basic_shot, dets, safe_scene_ctx, safe_audio_ctx, config)
        assert verdict.action == Action.NONE
        assert verdict.category == Category.SAFE

    def test_anime_catches_high_confidence(
        self, dummy_file, basic_shot, safe_scene_ctx, safe_audio_ctx
    ):
        """Even with anime's high threshold, 0.90 should trigger."""
        config = Config(
            input_path=dummy_file,
            content_type=ContentType.ANIME,
            strictness=Strictness.MEDIUM,
        )
        dets = [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.90, box=(0, 0, 100, 100))]
        ]

        verdict = fuse(basic_shot, dets, safe_scene_ctx, safe_audio_ctx, config)
        assert verdict.action == Action.BLACK_BOX

    def test_low_light_catches_lower_confidence(
        self, dummy_file, basic_shot, safe_scene_ctx, safe_audio_ctx
    ):
        """Low-light has 0.85x multiplier, so 0.55 * 0.85 = 0.4675 threshold.
        A 0.50 score should trigger."""
        config = Config(
            input_path=dummy_file,
            content_type=ContentType.LOW_LIGHT,
            strictness=Strictness.MEDIUM,
        )
        dets = [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.50, box=(0, 0, 100, 100))]
        ]

        verdict = fuse(basic_shot, dets, safe_scene_ctx, safe_audio_ctx, config)
        assert verdict.action == Action.BLACK_BOX


class TestFuseWithStrictness:
    def test_high_strictness_catches_moderate_score(
        self, dummy_file, basic_shot, safe_scene_ctx, safe_audio_ctx
    ):
        """High strictness preset: nudity threshold = 0.35. Score 0.40 should trigger."""
        config = Config(input_path=dummy_file, strictness=Strictness.HIGH)
        dets = [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.40, box=(0, 0, 100, 100))]
        ]

        verdict = fuse(basic_shot, dets, safe_scene_ctx, safe_audio_ctx, config)
        assert verdict.action == Action.BLACK_BOX

    def test_low_strictness_misses_moderate_score(
        self, dummy_file, basic_shot, safe_scene_ctx, safe_audio_ctx
    ):
        """Low strictness preset: nudity threshold = 0.75. Score 0.60 should NOT trigger."""
        config = Config(input_path=dummy_file, strictness=Strictness.LOW)
        dets = [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.60, box=(0, 0, 100, 100))]
        ]

        verdict = fuse(basic_shot, dets, safe_scene_ctx, safe_audio_ctx, config)
        assert verdict.action == Action.NONE

    def test_safe_scene_always_safe(
        self, dummy_file, basic_shot, safe_scene_ctx, safe_audio_ctx
    ):
        """No detections should always return SAFE regardless of strictness."""
        for s in Strictness:
            config = Config(input_path=dummy_file, strictness=s)
            verdict = fuse(basic_shot, [[]], safe_scene_ctx, safe_audio_ctx, config)
            assert verdict.action == Action.NONE
            assert verdict.category == Category.SAFE


class TestNudityLabels:
    def test_explicit_labels_complete(self):
        expected = {
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "ANUS_EXPOSED",
        }
        assert NUDITY_EXPLICIT_LABELS == expected

    def test_partial_labels_exist(self):
        assert len(NUDITY_PARTIAL_LABELS) > 0
        assert "FEMALE_BREAST_COVERED" in NUDITY_PARTIAL_LABELS

    def test_no_overlap_between_sets(self):
        assert len(NUDITY_EXPLICIT_LABELS & NUDITY_PARTIAL_LABELS) == 0
