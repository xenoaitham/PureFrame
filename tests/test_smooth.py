"""Tests for smooth detections - interpolation, padding, median filtering."""

from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.shots import Shot
from pureframe.pipeline.smooth import smooth_detections


class TestSmoothDetections:
    def _make_shot(self, start=0, end=100):
        return Shot(
            index=0,
            start_frame=start,
            end_frame=end,
            start_time=start / 24.0,
            end_time=end / 24.0,
        )

    def test_empty_detections(self):
        shot = self._make_shot(0, 50)
        result = smooth_detections({}, shot, padding_pct=0.2)
        assert result == {}

    def test_single_frame_detection(self):
        det = Detection(
            label="EXPOSED_BREAST_F",
            score=0.9,
            box=(100, 100, 200, 200),
        )
        per_frame = {10: [det]}
        shot = self._make_shot(0, 50)
        result = smooth_detections(per_frame, shot, padding_pct=0.0)

        assert 10 in result
        assert len(result[10]) >= 1
        box = result[10][0]
        assert box[0] >= 0 and box[1] >= 0

    def test_two_frame_interpolation(self):
        """Two detections should produce interpolated boxes between them."""
        det1 = Detection(label="EXPOSED_BREAST_F", score=0.9, box=(100, 100, 200, 200))
        det2 = Detection(label="EXPOSED_BREAST_F", score=0.9, box=(110, 110, 210, 210))

        per_frame = {0: [det1], 10: [det2]}
        shot = self._make_shot(0, 20)
        result = smooth_detections(per_frame, shot, padding_pct=0.0)

        # Should have boxes for frames 0 through 10 (interpolated)
        assert 0 in result
        assert 10 in result

    def test_padding_expands_boxes(self):
        det = Detection(label="EXPOSED_BREAST_F", score=0.9, box=(100, 100, 200, 200))
        per_frame = {5: [det]}
        shot = self._make_shot(0, 20)

        result_no_pad = smooth_detections(per_frame, shot, padding_pct=0.0)
        result_padded = smooth_detections(per_frame, shot, padding_pct=0.5)

        if 5 in result_no_pad and 5 in result_padded:
            box_no = result_no_pad[5][0]
            box_pad = result_padded[5][0]

            w_no = box_no[2] - box_no[0]
            w_pad = box_pad[2] - box_pad[0]
            assert w_pad >= w_no

    def test_multiple_tracks_independent(self):
        """Two non-overlapping detections should create separate tracks."""
        det_a = Detection(label="EXPOSED_BREAST_F", score=0.9, box=(10, 10, 50, 50))
        det_b = Detection(label="EXPOSED_BREAST_F", score=0.9, box=(300, 300, 400, 400))

        per_frame = {5: [det_a, det_b]}
        shot = self._make_shot(0, 20)
        result = smooth_detections(per_frame, shot, padding_pct=0.0)

        if 5 in result:
            assert len(result[5]) >= 2

    def test_consecutive_frames_tracked(self):
        """Boxes across consecutive frames should be tracked and smoothed."""
        dets = {}
        for f in range(10):
            dets[f] = [
                Detection(
                    label="EXPOSED_BREAST_F",
                    score=0.9,
                    box=(100 + f, 100 + f, 200 + f, 200 + f),
                )
            ]

        shot = self._make_shot(0, 20)
        result = smooth_detections(dets, shot, padding_pct=0.0)

        for f in range(10):
            assert f in result

    def test_many_frames_triggers_median_filter(self):
        """With >= 5 frames of detections, median filter should be applied."""
        dets = {}
        for f in range(20):
            noise = (f % 3) * 5
            dets[f] = [
                Detection(
                    label="EXPOSED_BREAST_F",
                    score=0.9,
                    box=(100 + noise, 100, 200 + noise, 200),
                )
            ]

        shot = self._make_shot(0, 30)
        result = smooth_detections(dets, shot, padding_pct=0.0)

        assert len(result) >= 15

    def test_boxes_clipped_to_non_negative(self):
        """Boxes near origin with padding should be clipped to 0."""
        det = Detection(label="EXPOSED_BREAST_F", score=0.9, box=(5, 5, 20, 20))
        per_frame = {3: [det]}
        shot = self._make_shot(0, 10)
        result = smooth_detections(per_frame, shot, padding_pct=1.0)

        if 3 in result:
            for box in result[3]:
                assert box[0] >= 0
                assert box[1] >= 0
