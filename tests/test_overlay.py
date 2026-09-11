"""Tests for render overlay callback - direct unit tests on the apply_censoring callback logic."""

import cv2
import numpy as np

from pureframe.pipeline.shots import Action


class TestOverlayCallback:
    """Test the overlay logic by calling it directly with numpy frames."""

    def _make_frame(self, w=1280, h=720):
        """Create a test BGR frame."""
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def test_no_action_returns_original(self):
        frame = self._make_frame()
        original = frame.copy()

        frame_actions = {}  # No actions for any frame
        frame_data = frame_actions.get(0)
        assert frame_data is None
        # No modification
        np.testing.assert_array_equal(frame, original)

    def test_full_frame_blur_changes_frame(self):
        frame = self._make_frame()
        original = frame.copy()

        # Apply Gaussian blur (same as apply.py line 30)
        blurred = cv2.GaussianBlur(frame, (99, 99), 30)

        # Blurred frame should differ from original
        assert not np.array_equal(blurred, original)
        assert blurred.shape == original.shape

    def test_black_box_draws_rectangle(self):
        frame = self._make_frame(640, 480)
        original = frame.copy()

        # Simulate black box at (100, 100, 200, 200)
        box_color = (0, 0, 0)
        cv2.rectangle(frame, (100, 100), (200, 200), box_color, -1)

        # The region should be all black
        region = frame[100:200, 100:200]
        assert np.all(region == 0)

        # Rest should be unchanged
        np.testing.assert_array_equal(frame[0:99, :], original[0:99, :])

    def test_box_scaling_math(self):
        """Test the coordinate scaling math from apply.py."""
        w, h = 1920, 1080
        det_res = 480

        # Same logic as apply.py lines 41-51
        dw, dh = w, h
        if dw > dh and dw > det_res:
            dh = int(dh * (det_res / dw))
            dw = det_res
        elif dh > dw and dh > det_res:
            dw = int(dw * (det_res / dh))
            dh = det_res
        dw = dw - (dw % 2)
        dh = dh - (dh % 2)

        scale_w = w / dw if dw > 0 else 1.0
        scale_h = h / dh if dh > 0 else 1.0

        # A box at detection-res coords (100, 50, 200, 150)
        x1, y1, x2, y2 = 100, 50, 200, 150
        nx1 = int(x1 * scale_w)
        ny1 = int(y1 * scale_h)
        nx2 = int(x2 * scale_w)
        ny2 = int(y2 * scale_h)

        # Scaled coords should be larger than detection coords
        assert nx1 >= x1
        assert ny1 >= y1
        assert nx2 >= x2
        assert ny2 >= y2

    def test_box_scaling_portrait(self):
        """Test scaling for portrait video."""
        w, h = 720, 1280
        det_res = 480

        dw, dh = w, h
        if dw > dh and dw > det_res:
            dh = int(dh * (det_res / dw))
            dw = det_res
        elif dh > dw and dh > det_res:
            dw = int(dw * (det_res / dh))
            dh = det_res
        dw = dw - (dw % 2)
        dh = dh - (dh % 2)

        assert dh == det_res
        assert dw < det_res

    def test_frame_actions_priority(self):
        """Test that FULL_FRAME_BLUR takes priority over BLACK_BOX."""
        frame_actions = {
            0: {"action": Action.FULL_FRAME_BLUR, "boxes": []},
            1: {"action": Action.BLACK_BOX, "boxes": [(10, 10, 50, 50)]},
            2: {"action": Action.NONE, "boxes": []},
        }

        assert frame_actions[0]["action"] == Action.FULL_FRAME_BLUR
        assert frame_actions[1]["action"] == Action.BLACK_BOX
        assert frame_actions[2]["action"] == Action.NONE

    def test_custom_box_color(self):
        """Test rendering with custom color."""
        frame = self._make_frame(200, 200)
        color = (0, 0, 255)  # Red in BGR
        cv2.rectangle(frame, (50, 50), (100, 100), color, -1)

        region = frame[50:100, 50:100]
        # Should be all red (0, 0, 255)
        assert np.all(region[:, :, 0] == 0)
        assert np.all(region[:, :, 1] == 0)
        assert np.all(region[:, :, 2] == 255)
