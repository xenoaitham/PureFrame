"""Tests for apply_censoring - end-to-end render with overlay callback."""

from pureframe.config import Config
from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.render.apply import apply_censoring
from pureframe.pipeline.shots import Action


class TestApplyCensoring:
    def test_apply_no_actions_copies_video(self, tmp_path, synthetic_video):
        """With no frame actions, output should be produced (essentially a copy)."""
        output = tmp_path / "output.mp4"
        config = Config(input_path=synthetic_video, output_path=output)
        settings = get_settings(HardwareProfile.CPU)

        apply_censoring(
            input_path=synthetic_video,
            output_path=output,
            frame_actions={},
            config=config,
            profile_settings=settings,
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_apply_full_blur(self, tmp_path, synthetic_video):
        """Test rendering with FULL_FRAME_BLUR on all frames."""
        output = tmp_path / "blurred.mp4"
        config = Config(input_path=synthetic_video, output_path=output)
        settings = get_settings(HardwareProfile.CPU)

        # Blur first 30 frames
        frame_actions = {
            i: {"action": Action.FULL_FRAME_BLUR, "boxes": []} for i in range(30)
        }

        apply_censoring(
            input_path=synthetic_video,
            output_path=output,
            frame_actions=frame_actions,
            config=config,
            profile_settings=settings,
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_apply_black_box(self, tmp_path, synthetic_video):
        """Test rendering with BLACK_BOX on specific frames."""
        output = tmp_path / "boxed.mp4"
        config = Config(input_path=synthetic_video, output_path=output)
        settings = get_settings(HardwareProfile.CPU)

        # Black box on frames 0-9
        frame_actions = {
            i: {
                "action": Action.BLACK_BOX,
                "boxes": [(50, 50, 150, 150)],
            }
            for i in range(10)
        }

        apply_censoring(
            input_path=synthetic_video,
            output_path=output,
            frame_actions=frame_actions,
            config=config,
            profile_settings=settings,
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_apply_black_box_empty_boxes(self, tmp_path, synthetic_video):
        """BLACK_BOX with no boxes should not crash."""
        output = tmp_path / "no_boxes.mp4"
        config = Config(input_path=synthetic_video, output_path=output)
        settings = get_settings(HardwareProfile.CPU)

        frame_actions = {
            0: {"action": Action.BLACK_BOX, "boxes": []},
        }

        apply_censoring(
            input_path=synthetic_video,
            output_path=output,
            frame_actions=frame_actions,
            config=config,
            profile_settings=settings,
        )
        assert output.exists()

    def test_apply_mixed_actions(self, tmp_path, synthetic_video):
        """Test mix of FULL_FRAME_BLUR and BLACK_BOX across frames."""
        output = tmp_path / "mixed.mp4"
        config = Config(input_path=synthetic_video, output_path=output)
        settings = get_settings(HardwareProfile.CPU)

        frame_actions = {}
        # First 5 frames: blur
        for i in range(5):
            frame_actions[i] = {"action": Action.FULL_FRAME_BLUR, "boxes": []}
        # Next 5 frames: black box
        for i in range(5, 10):
            frame_actions[i] = {
                "action": Action.BLACK_BOX,
                "boxes": [(20, 20, 100, 100)],
            }

        apply_censoring(
            input_path=synthetic_video,
            output_path=output,
            frame_actions=frame_actions,
            config=config,
            profile_settings=settings,
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_apply_custom_box_color(self, tmp_path, synthetic_video):
        """Test rendering with custom box color."""
        output = tmp_path / "custom_color.mp4"
        config = Config(
            input_path=synthetic_video,
            output_path=output,
            box_color=(0, 0, 255),  # Red
        )
        settings = get_settings(HardwareProfile.CPU)

        frame_actions = {
            0: {"action": Action.BLACK_BOX, "boxes": [(10, 10, 50, 50)]},
        }

        apply_censoring(
            input_path=synthetic_video,
            output_path=output,
            frame_actions=frame_actions,
            config=config,
            profile_settings=settings,
        )
        assert output.exists()
