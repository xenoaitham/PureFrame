import numpy as np

from pureframe.pipeline.detect.face import FaceDetector


def test_kiss_mouth_region(monkeypatch):
    detector = FaceDetector()

    class MockNet:
        def setInput(self, blob):
            pass

        def forward(self):
            out = np.zeros((1, 1, 1, 7))
            out[0, 0, 0, 2] = 0.9
            out[0, 0, 0, 3] = 0.1
            out[0, 0, 0, 4] = 0.1
            out[0, 0, 0, 5] = 0.9
            out[0, 0, 0, 6] = 0.9
            return out

    detector.net = MockNet()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mouths = detector.detect_mouths(frame)

    assert len(mouths) == 1
    x1, y1, x2, y2 = mouths[0]

    assert x1 == 10
    assert y1 == 62
    assert x2 == 90
    assert y2 == 90
