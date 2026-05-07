"""Tests for the render overlay callback and plan frame actions."""

from pureframe.pipeline.render.plan import CensorPlan
from pureframe.pipeline.shots import Shot, ShotVerdict, Action, Category
from pureframe.utils.ffmpeg import VideoMetadata
from fractions import Fraction
from datetime import datetime, timezone


class TestCensorPlanFrameActions:
    def _make_plan(self, verdicts, shots):
        return CensorPlan(
            pureframe_version="test",
            plan_version=1,
            input_metadata=VideoMetadata(
                width=1280,
                height=720,
                fps=Fraction(24, 1),
                duration_seconds=10.0,
                total_frames=240,
                has_audio=False,
                audio_streams=[],
                subtitle_streams=[],
                container="mp4",
                video_codec="h264",
                pixel_format="yuv420p",
                color_space="bt709",
                is_hdr=False,
            ),
            config_snapshot={},
            shots=shots,
            verdicts=verdicts,
            total_censored_frames=0,
            total_blur_frames=0,
            generated_at=datetime.now(timezone.utc),
        )

    def test_safe_shots_produce_empty_actions(self):
        shots = [Shot(index=0, start_frame=0, end_frame=100, start_time=0.0, end_time=4.0)]
        verdicts = [
            ShotVerdict(
                shot_index=0,
                category=Category.SAFE,
                action=Action.NONE,
                confidence=0.95,
                boxes=None,
                reasoning="Safe",
            )
        ]
        plan = self._make_plan(verdicts, shots)
        actions = plan.build_frame_actions()
        assert len(actions) == 0

    def test_black_box_covers_full_shot(self):
        shots = [Shot(index=0, start_frame=10, end_frame=50, start_time=0.4, end_time=2.0)]
        verdicts = [
            ShotVerdict(
                shot_index=0,
                category=Category.NUDITY_EXPLICIT,
                action=Action.BLACK_BOX,
                confidence=0.90,
                boxes=None,
                reasoning="Nudity",
            )
        ]
        plan = self._make_plan(verdicts, shots)
        actions = plan.build_frame_actions()

        # Should cover frames 10 to 49
        for f in range(10, 50):
            assert f in actions
            assert actions[f]["action"] == Action.BLACK_BOX

    def test_full_frame_blur_covers_shot(self):
        shots = [Shot(index=0, start_frame=0, end_frame=30, start_time=0.0, end_time=1.25)]
        verdicts = [
            ShotVerdict(
                shot_index=0,
                category=Category.SEXUAL_CONTEXT_NO_NUDITY,
                action=Action.FULL_FRAME_BLUR,
                confidence=0.70,
                boxes=None,
                reasoning="Implied",
            )
        ]
        plan = self._make_plan(verdicts, shots)
        actions = plan.build_frame_actions()

        for f in range(0, 30):
            assert actions[f]["action"] == Action.FULL_FRAME_BLUR

    def test_full_blur_overrides_black_box(self):
        """If two verdicts overlap and one is FULL_FRAME_BLUR, it takes priority."""
        shots = [
            Shot(index=0, start_frame=0, end_frame=50, start_time=0.0, end_time=2.0),
            Shot(index=1, start_frame=30, end_frame=80, start_time=1.25, end_time=3.3),
        ]
        verdicts = [
            ShotVerdict(
                shot_index=0,
                category=Category.NUDITY_EXPLICIT,
                action=Action.BLACK_BOX,
                confidence=0.90,
                boxes=None,
                reasoning="Box",
            ),
            ShotVerdict(
                shot_index=1,
                category=Category.SEXUAL_CONTEXT_NO_NUDITY,
                action=Action.FULL_FRAME_BLUR,
                confidence=0.70,
                boxes=None,
                reasoning="Blur",
            ),
        ]
        plan = self._make_plan(verdicts, shots)
        actions = plan.build_frame_actions()

        # Overlap region (30-49): FULL_FRAME_BLUR should win
        for f in range(30, 50):
            assert actions[f]["action"] == Action.FULL_FRAME_BLUR

    def test_whitelisted_shot_excluded(self):
        shots = [
            Shot(index=0, start_frame=0, end_frame=100, start_time=0.0, end_time=4.0),
            Shot(index=1, start_frame=100, end_frame=200, start_time=4.0, end_time=8.0),
        ]
        verdicts = [
            ShotVerdict(
                shot_index=0,
                category=Category.NUDITY_EXPLICIT,
                action=Action.NONE,  # Whitelisted
                confidence=0.90,
                boxes=None,
                reasoning="Whitelisted",
            ),
            ShotVerdict(
                shot_index=1,
                category=Category.NUDITY_EXPLICIT,
                action=Action.BLACK_BOX,
                confidence=0.85,
                boxes=None,
                reasoning="Active",
            ),
        ]
        plan = self._make_plan(verdicts, shots)
        actions = plan.build_frame_actions()

        # Shot 0 should not be in actions (whitelisted)
        for f in range(0, 100):
            assert f not in actions

        # Shot 1 should be in actions
        for f in range(100, 200):
            assert f in actions

    def test_multiple_shots_independent(self):
        shots = [
            Shot(index=0, start_frame=0, end_frame=50, start_time=0.0, end_time=2.0),
            Shot(index=1, start_frame=100, end_frame=150, start_time=4.0, end_time=6.0),
        ]
        verdicts = [
            ShotVerdict(
                shot_index=0,
                category=Category.NUDITY_EXPLICIT,
                action=Action.BLACK_BOX,
                confidence=0.90,
                boxes=None,
                reasoning="Shot 0",
            ),
            ShotVerdict(
                shot_index=1,
                category=Category.SAFE,
                action=Action.NONE,
                confidence=0.95,
                boxes=None,
                reasoning="Shot 1 safe",
            ),
        ]
        plan = self._make_plan(verdicts, shots)
        actions = plan.build_frame_actions()

        # Shot 0 censored, shot 1 not
        assert 0 in actions
        assert 49 in actions
        assert 50 not in actions
        assert 100 not in actions


class TestPlanSerialization:
    def test_roundtrip(self, tmp_path):
        shots = [Shot(index=0, start_frame=0, end_frame=100, start_time=0.0, end_time=4.0)]
        verdicts = [
            ShotVerdict(
                shot_index=0,
                category=Category.SAFE,
                action=Action.NONE,
                confidence=0.95,
                boxes=None,
                reasoning="Safe",
            )
        ]
        plan = CensorPlan(
            pureframe_version="0.1.0b5",
            plan_version=1,
            input_metadata=VideoMetadata(
                width=1920,
                height=1080,
                fps=Fraction(30, 1),
                duration_seconds=4.0,
                total_frames=120,
                has_audio=True,
                audio_streams=[],
                subtitle_streams=[],
                container="mp4",
                video_codec="h264",
                pixel_format="yuv420p",
                color_space="bt709",
                is_hdr=False,
            ),
            config_snapshot={"nudity_threshold": 0.55, "content_type": "live-action"},
            shots=shots,
            verdicts=verdicts,
            total_censored_frames=0,
            total_blur_frames=0,
            generated_at=datetime.now(timezone.utc),
        )

        path = tmp_path / "test_plan.json"
        plan.serialize(path)

        loaded = CensorPlan.load(path)
        assert loaded.pureframe_version == "0.1.0b5"
        assert loaded.plan_version == 1
        assert len(loaded.shots) == 1
        assert len(loaded.verdicts) == 1
        assert loaded.verdicts[0].action == Action.NONE
        assert loaded.config_snapshot["content_type"] == "live-action"
        assert loaded.input_metadata.width == 1920
