"""Pin the NudeNet 3.x box-format contract.

NudeDetector.detect() returns ``box`` as ``[x, y, w, h]`` (verified against
nudenet 3.4.2 source: nudenet.py appends ``[int(x), int(y), int(w), int(h)]``).
NudityDetector must convert to ``(x1, y1, x2, y2)`` - getting this wrong
either inflates boxes to cover half the frame or shifts them off-target.
"""

import numpy as np

from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.detect.nudity import NudityDetector


class _FakeNudeNet:
    """Stands in for nudenet.NudeDetector; returns fixed xywh predictions."""

    def detect(self, image):
        assert isinstance(image, np.ndarray)
        return [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.97, "box": [100, 50, 80, 60]},
            {"class": "FACE_FEMALE", "score": 0.99, "box": [10, 10, 20, 20]},
            {"class": "BUTTOCKS_EXPOSED", "score": 0.42, "box": [0, 0, 30, 30]},
        ]


def _detector_with_fake() -> NudityDetector:
    det = NudityDetector(get_settings(HardwareProfile.CPU))
    det.detector = _FakeNudeNet()
    return det


def test_xywh_converted_to_corners():
    det = _detector_with_fake()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    (dets,) = det.detect_batch([frame])

    # detect_batch passes every explicit-label hit through; score filtering
    # belongs to densify_shot, so both explicit predictions survive here.
    labels = sorted(d.label for d in dets)
    assert labels == ["BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED"]

    breast = next(d for d in dets if d.label == "FEMALE_BREAST_EXPOSED")
    # x,y,w,h = 100,50,80,60 -> corners (100,50)-(180,110)
    assert breast.box == (100, 50, 180, 110)


def test_explicit_label_filtering():
    det = _detector_with_fake()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    (dets,) = det.detect_batch([frame])
    assert all(
        d.label
        in {
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "BUTTOCKS_EXPOSED",
            "ANUS_EXPOSED",
        }
        for d in dets
    )


def test_detector_failure_degrades_to_empty_not_crash():
    class _Boom:
        def detect(self, image):
            raise RuntimeError("onnx exploded")

    det = NudityDetector(get_settings(HardwareProfile.CPU))
    det.detector = _Boom()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert det.detect_batch([frame]) == [[]]
