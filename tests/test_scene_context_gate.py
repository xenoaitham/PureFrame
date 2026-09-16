"""Scene-context gate: confident beach/pool and gym/sports contexts raise
the nudity bar so swimwear and shirtless athletes stop flagging.

The gate only ever raises (never lowers) a threshold, and a confident
sexual scene signal switches it off - the recall guard the fuse tests
pin per context pair. The plan-pipeline test proves the same factor
reaches the rescan bar, so verdict and box bars cannot disagree.
"""

import os
import tempfile
from pathlib import Path

import pytest

from pureframe.config import Config
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


def ctx(
    explicit_act=0.0,
    implied_sex=0.0,
    kissing=0.0,
    safe=0.9,
    beach_pool=0.0,
    sports=0.0,
) -> ShotContext:
    return ShotContext(
        explicit_act_score=explicit_act,
        implied_sex_score=implied_sex,
        kissing_score=kissing,
        safe_score=safe,
        beach_pool_score=beach_pool,
        sports_score=sports,
    )


def beach_ctx(**kwargs) -> ShotContext:
    return ctx(beach_pool=SCENE_CONTEXT_CONFIDENCE + 0.1, **kwargs)


def sports_ctx(**kwargs) -> ShotContext:
    return ctx(sports=SCENE_CONTEXT_CONFIDENCE + 0.1, **kwargs)


def weak_swimwear_dets(score=0.65):
    """Below the gated medium bar (0.55 * 1.4 = 0.77), above the raw 0.55."""
    return [[Detection(label="BUTTOCKS_EXPOSED", score=score, box=(0, 0, 100, 100))]]


class TestHelper:
    def test_confident_beach_raises(self, dummy_file):
        config = Config(input_path=dummy_file)
        assert scene_context_factor(beach_ctx(), config) == 1.4

    def test_confident_sports_raises(self, dummy_file):
        config = Config(input_path=dummy_file)
        assert scene_context_factor(sports_ctx(), config) == 1.4

    def test_weak_context_leaves_the_bar_alone(self, dummy_file):
        config = Config(input_path=dummy_file)
        assert scene_context_factor(ctx(beach_pool=0.5), config) == 1.0

    def test_confident_explicit_act_switches_the_gate_off(self, dummy_file):
        config = Config(input_path=dummy_file)
        assert scene_context_factor(beach_ctx(explicit_act=0.6), config) == 1.0

    def test_confident_implied_sex_switches_the_gate_off(self, dummy_file):
        config = Config(input_path=dummy_file)
        assert scene_context_factor(beach_ctx(implied_sex=0.6), config) == 1.0

    def test_disabled_gate(self, dummy_file):
        config = Config(input_path=dummy_file, scene_context_factor=1.0)
        assert scene_context_factor(beach_ctx(), config) == 1.0


class TestFusePerContextPair:
    def test_beach_swimwear_stays_unflagged(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        verdict = fuse(
            basic_shot, weak_swimwear_dets(), beach_ctx(), safe_audio_ctx, config
        )
        assert verdict.action == Action.NONE

    def test_sports_shirtless_stays_unflagged(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        verdict = fuse(
            basic_shot, weak_swimwear_dets(), sports_ctx(), safe_audio_ctx, config
        )
        assert verdict.action == Action.NONE

    def test_same_score_flags_without_the_context(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        verdict = fuse(basic_shot, weak_swimwear_dets(), ctx(), safe_audio_ctx, config)
        assert verdict.action == Action.BLACK_BOX
        assert verdict.category == Category.NUDITY_EXPLICIT

    def test_strong_nudity_flags_even_on_a_beach(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        # The gate raises the bar; it does not blank real nudity.
        config = Config(input_path=dummy_file)
        verdict = fuse(
            basic_shot,
            weak_swimwear_dets(score=0.95),
            beach_ctx(),
            safe_audio_ctx,
            config,
        )
        assert verdict.action == Action.BLACK_BOX

    def test_sexual_act_recall_is_never_touched(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        # Sub-bar visual detections (0.30 < 0.55) so the sexual-act branch
        # is what must fire, not the nudity branch.
        config = Config(input_path=dummy_file)
        scene = beach_ctx(explicit_act=0.7)
        audio = AudioContext(
            moaning_score=0.9, sexual_audio_score=0.7, music_score=0.0, speech_score=0.0
        )
        verdict = fuse(basic_shot, weak_swimwear_dets(score=0.30), scene, audio, config)
        assert verdict.category == Category.SEXUAL_ACT_VISIBLE

    def test_implied_sex_recall_is_never_touched(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        config = Config(input_path=dummy_file)
        scene = beach_ctx(implied_sex=0.7)
        audio = AudioContext(
            moaning_score=0.7, sexual_audio_score=0.0, music_score=0.0, speech_score=0.0
        )
        verdict = fuse(basic_shot, weak_swimwear_dets(score=0.30), scene, audio, config)
        assert verdict.category == Category.SEXUAL_CONTEXT_NO_NUDITY

    def test_kiss_branch_is_unaffected(self, dummy_file, safe_audio_ctx):
        shot = Shot(index=0, start_frame=0, end_frame=240, start_time=0.0, end_time=4.0)
        config = Config(input_path=dummy_file)
        verdict = fuse(
            shot,
            weak_swimwear_dets(),
            beach_ctx(kissing=0.8, safe=0.1),
            safe_audio_ctx,
            config,
        )
        assert verdict.category == Category.KISS_INTENSE

    def test_guide_and_context_factors_compose(
        self, dummy_file, basic_shot, safe_audio_ctx
    ):
        # 0.55 * 0.7 (guide) * 1.4 (context) = 0.539: a 0.50 detection
        # misses the composed bar but clears the guide-only one (0.385).
        config = Config(input_path=dummy_file)
        scene = beach_ctx(safe=0.0)
        dets = weak_swimwear_dets(score=0.50)
        both = fuse(
            basic_shot, dets, scene, safe_audio_ctx, config, guide_threshold_factor=0.7
        )
        assert both.action == Action.NONE
        guide_only = fuse(
            basic_shot,
            dets,
            ctx(safe=0.0),
            safe_audio_ctx,
            config,
            guide_threshold_factor=0.7,
        )
        assert guide_only.action == Action.BLACK_BOX


class TestConfig:
    def test_factor_changes_the_hash(self, dummy_file):
        plain = Config(input_path=dummy_file)
        raised = Config(input_path=dummy_file, scene_context_factor=1.8)
        assert plain.config_hash != raised.config_hash

    def test_out_of_range_factor_rejected(self, dummy_file):
        with pytest.raises(Exception):
            Config(input_path=dummy_file, scene_context_factor=2.5)
        with pytest.raises(Exception):
            Config(input_path=dummy_file, scene_context_factor=0.5)


class TestPrompts:
    def test_context_categories_exist(self):
        assert "beach_pool" in PROMPT_SETS
        assert "sports_gym" in PROMPT_SETS
        assert len(PROMPT_SETS["beach_pool"]) >= 3
        assert len(PROMPT_SETS["sports_gym"]) >= 3


class TestPlanPipeline:
    def test_rescan_bar_carries_the_context_factor(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        """A beach-classified shot with a 0.65 detection: the fuse bar sits
        at 0.77, the verdict stays SAFE, and the rescan (candidate: 0.65 is
        over the 0.25 floor) must be handed the same raised bar."""
        import pureframe.cli
        from pureframe.hardware import HardwareProfile
        from pureframe.pipeline.detect.nudity import NudityDetector

        captured = {}

        def fake_rescan(shot, video_path, detector, settings, meta, threshold, **kw):
            captured["threshold"] = threshold
            return {}

        monkeypatch.setattr(pureframe.cli, "rescan_shot", fake_rescan)
        monkeypatch.setattr(
            NudityDetector,
            "detect_batch",
            lambda self, frames: [
                [Detection(label="BUTTOCKS_EXPOSED", score=0.65, box=(10, 10, 60, 60))]
                for _ in frames
            ],
        )

        class BeachScene:
            def __init__(self, settings):
                self.enabled = True

            def classify_shot(self, frame):
                return beach_ctx(safe=0.05)

            def unload(self):
                pass

        monkeypatch.setattr(pureframe.cli, "SceneClassifier", BeachScene)

        config = Config.from_cli(
            input_path=three_shot_video,
            output_path=tmp_path / "out.mp4",
            profile=HardwareProfile.CPU,
            no_audio=True,
        )
        plan = pureframe.cli.generate_plan(config)

        assert captured["threshold"] == pytest.approx(0.55 * 1.4)
        assert all(v.action == Action.NONE for v in plan.verdicts)

    def test_disabling_the_gate_recovers_the_flag(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        import pureframe.cli
        from pureframe.hardware import HardwareProfile
        from pureframe.pipeline.detect.nudity import NudityDetector

        monkeypatch.setattr(
            NudityDetector,
            "detect_batch",
            lambda self, frames: [
                [Detection(label="BUTTOCKS_EXPOSED", score=0.65, box=(10, 10, 60, 60))]
                for _ in frames
            ],
        )

        class BeachScene:
            def __init__(self, settings):
                self.enabled = True

            def classify_shot(self, frame):
                return beach_ctx(safe=0.05)

            def unload(self):
                pass

        monkeypatch.setattr(pureframe.cli, "SceneClassifier", BeachScene)

        config = Config.from_cli(
            input_path=three_shot_video,
            output_path=tmp_path / "out.mp4",
            profile=HardwareProfile.CPU,
            no_audio=True,
            scene_context_factor=1.0,
        )
        plan = pureframe.cli.generate_plan(config)
        flagged = [v for v in plan.verdicts if v.action != Action.NONE]
        assert flagged, "the same detection flags once the gate is off"
        assert flagged[0].category == Category.NUDITY_EXPLICIT
        assert flagged[0].boxes
