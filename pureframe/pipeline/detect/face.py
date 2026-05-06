import cv2
import numpy as np
from pathlib import Path


class FaceDetector:
    def __init__(self):
        data_dir = Path(__file__).parent.parent.parent / "data"
        prototxt = data_dir / "deploy.prototxt"
        model = data_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))

    def detect_mouths(
        self, frame_bgr: np.ndarray, threshold: float = 0.5
    ) -> list[tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        mouths = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face_h = y2 - y1
                m_y1 = y2 - int(face_h * 0.35)
                m_y2 = y2

                # Make sure the box is non-negative
                x1 = max(0, x1)
                m_y1 = max(0, m_y1)
                x2 = min(w, x2)
                m_y2 = min(h, m_y2)

                if x2 > x1 and m_y2 > m_y1:
                    mouths.append((x1, m_y1, x2, m_y2))

        return mouths
