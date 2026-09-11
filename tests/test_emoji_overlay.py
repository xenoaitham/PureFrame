"""Emoji overlay censor style (BlurMode.EMOJI).

One emoji covers the box center, sized from the box: the explicit
``emoji_char`` override wins, else the per-category default, else the
generic marker. Without an emoji font on the system the overlay falls
back to a solid box - a region must never render as untouched pixels
just because a font was missing.
"""

import cv2
import numpy as np

from pureframe.config import (
    DEFAULT_EMOJI,
    EMOJI_BY_CATEGORY,
    BlurMode,
    Config,
)
from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.render.overlay import (
    _apply_emoji,
    _load_emoji_font,
    build_overlay_callback,
    resolve_emoji,
)
from pureframe.pipeline.shots import Action, Category, Shot, ShotVerdict


def _config(**kwargs) -> Config:
    kwargs.setdefault("profile", HardwareProfile.CPU)
    return Config(input_path="d", output_path="d", **kwargs)


def _frame(w=320, h=240):
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _actions(box=(100, 80, 220, 160), category="NUDITY_EXPLICIT"):
    return {
        0: {"action": Action.BLACK_BOX, "boxes": [box], "category": category},
    }


class TestResolveEmoji:
    def test_explicit_override_wins(self):
        config = _config(blur_mode=BlurMode.EMOJI, emoji_char="🌟")
        assert resolve_emoji(config, "KISS_INTENSE") == "🌟"

    def test_category_default(self):
        config = _config(blur_mode=BlurMode.EMOJI)
        assert (
            resolve_emoji(config, "KISS_INTENSE") == EMOJI_BY_CATEGORY["KISS_INTENSE"]
        )

    def test_unknown_category_uses_default(self):
        config = _config(blur_mode=BlurMode.EMOJI)
        assert resolve_emoji(config, "SOMETHING_NEW") == DEFAULT_EMOJI
        assert resolve_emoji(config, None) == DEFAULT_EMOJI


class TestFrameActionCategory:
    def test_black_box_frames_carry_category(self):
        from datetime import UTC, datetime
        from fractions import Fraction

        from pureframe.pipeline.probe import VideoMetadata
        from pureframe.pipeline.render.plan import CensorPlan

        shot = Shot(
            index=0,
            start_frame=0,
            end_frame=10,
            start_time=0.0,
            end_time=1.0,
        )
        verdict = ShotVerdict(
            shot_index=0,
            category=Category.NUDITY_EXPLICIT,
            action=Action.BLACK_BOX,
            confidence=0.9,
            reasoning="",
        )
        plan = CensorPlan(
            pureframe_version="test",
            plan_version=1,
            input_metadata=VideoMetadata(
                width=64,
                height=64,
                fps=Fraction(30, 1),
                duration_seconds=1.0,
                total_frames=10,
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
            shots=[shot],
            verdicts=[verdict],
            total_censored_frames=10,
            total_blur_frames=0,
            generated_at=datetime.now(UTC),
        )
        actions = plan.build_frame_actions()
        assert len(actions) == 10
        assert all(a["category"] == "NUDITY_EXPLICIT" for a in actions.values())

    def test_full_frame_frames_stay_uncategorized(self):
        from datetime import UTC, datetime
        from fractions import Fraction

        from pureframe.pipeline.probe import VideoMetadata
        from pureframe.pipeline.render.plan import CensorPlan

        shot = Shot(
            index=0,
            start_frame=0,
            end_frame=10,
            start_time=0.0,
            end_time=1.0,
        )
        verdict = ShotVerdict(
            shot_index=0,
            category=Category.SEXUAL_CONTEXT_NO_NUDITY,
            action=Action.FULL_FRAME_BLUR,
            confidence=0.9,
            reasoning="",
        )
        plan = CensorPlan(
            pureframe_version="test",
            plan_version=1,
            input_metadata=VideoMetadata(
                width=64,
                height=64,
                fps=Fraction(30, 1),
                duration_seconds=1.0,
                total_frames=10,
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
            shots=[shot],
            verdicts=[verdict],
            total_censored_frames=10,
            total_blur_frames=10,
            generated_at=datetime.now(UTC),
        )
        actions = plan.build_frame_actions()
        assert all("category" not in a for a in actions.values())


class TestApplyEmoji:
    def test_region_is_claimed_with_font(self):
        """With an emoji font available, the box center carries new ink in
        a color the original did not have at that density."""
        if _load_emoji_font() is None:
            import pytest

            pytest.skip("no emoji font on this system; fallback covered below")
        frame = _frame()
        original = frame.copy()
        box = (100, 80, 220, 160)
        _apply_emoji(frame, box, DEFAULT_EMOJI)
        ch, cw = 120, 160  # box center (h, w)
        patch = frame[ch - 10 : ch + 10, cw - 10 : cw + 10]
        orig_patch = original[ch - 10 : ch + 10, cw - 10 : cw + 10]
        assert not np.array_equal(patch, orig_patch)

    def test_missing_font_falls_back_to_solid(self, monkeypatch):
        import pureframe.pipeline.render.overlay as overlay

        # Patch the tile (not the font loader): the loader is lru_cached and
        # an earlier test may have warmed it with a real font.
        monkeypatch.setattr(overlay, "_render_emoji_tile", lambda char: None)
        frame = _frame()
        box = (100, 80, 220, 160)
        _apply_emoji(frame, box, DEFAULT_EMOJI)
        region = frame[80:160, 100:220]
        assert np.all(region == 0), "font-less systems must still censor"

    def test_emoji_does_not_touch_pixels_outside_the_box(self):
        if _load_emoji_font() is None:
            import pytest

            pytest.skip("no emoji font on this system")
        frame = _frame()
        original = frame.copy()
        _apply_emoji(frame, (100, 80, 220, 160), DEFAULT_EMOJI)
        np.testing.assert_array_equal(frame[:, :90], original[:, :90])
        np.testing.assert_array_equal(frame[:, 230:], original[:, 230:])


class TestCallbackEmojiMode:
    def test_emoji_mode_changes_flagged_region(self):
        config = _config(blur_mode=BlurMode.EMOJI)
        settings = get_settings(config.profile)
        callback = build_overlay_callback(_actions(), config, settings)
        frame = _frame()
        original = frame.copy()
        out = callback(0, frame)
        center = out[110:130, 150:170]
        assert not np.array_equal(center, original[110:130, 150:170])
        np.testing.assert_array_equal(out[:, :90], original[:, :90])

    def test_unflagged_frames_pass_through(self):
        config = _config(blur_mode=BlurMode.EMOJI)
        settings = get_settings(config.profile)
        callback = build_overlay_callback(_actions(), config, settings)
        frame = _frame()
        original = frame.copy()
        np.testing.assert_array_equal(callback(5, frame), original)

    def test_explicit_char_uses_override(self):
        """Two different override characters must produce different
        renderings of the same box."""
        if _load_emoji_font() is None:
            import pytest

            pytest.skip("no emoji font on this system")
        outputs = []
        for char in ("🚫", "💋"):
            config = _config(blur_mode=BlurMode.EMOJI, emoji_char=char)
            settings = get_settings(config.profile)
            callback = build_overlay_callback(_actions(), config, settings)
            frame = _frame()
            outputs.append(callback(0, frame.copy()))
        a = outputs[0][110:130, 150:170].astype(int)
        b = outputs[1][110:130, 150:170].astype(int)
        assert np.abs(a - b).mean() > 1.0


class TestEmojiRenderVideo:
    """One e2e-style render check: the emoji style lands in a real render
    through the shared overlay path, and the container stays intact."""

    def test_emoji_render_changes_the_flagged_region(
        self, synthetic_video, tmp_path, monkeypatch
    ):
        from tests.test_e2e import _run_pipeline

        out_path = _run_pipeline(
            tmp_path, monkeypatch, synthetic_video, BlurMode.EMOJI, "out_emoji.mp4"
        )
        assert out_path.exists()

        def center_patch(path):
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
            ret, frame = cap.read()
            cap.release()
            assert ret
            return frame[120:380, 170:430].astype(int)

        # The same ROI the BLUR e2e guards: with EMOJI it must be visibly
        # rewritten, not blurred into smoothness.
        assert (
            np.abs(center_patch(out_path) - center_patch(synthetic_video)).mean() > 5.0
        )
