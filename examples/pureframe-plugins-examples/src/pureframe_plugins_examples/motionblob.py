"""The reference PureFrame detector plugin: a tiny motion-blob detector.

It censors whatever moves. There is no model, no weights and no network:
consecutive keyframes are differenced, the residual is thresholded and
connected components above a minimum area become ``motion_blob``
detections in the ``MOTION_VISIBLE`` box-provider category. That makes it
useful on its own (censor a person or object you cannot name, or anything
that moves in a region you want clean) and as the copy-paste starting
point for a real detector.

The class honors the PureFrame plugin contract - see ``docs/plugin-api.md``
for the developer reference and ``docs/plugins.md`` for the user page:

    label_categories / label_thresholds   declared on the class
    __init__(settings)                    light; heavy work stays lazy
    detect_batch(frames_bgr)              one Detection list per frame
    unload()                              drop cached state
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# PureFrame imports Detection from its own pipeline; the plugin keeps a
# local pydantic-free equivalent so the package has no pureframe dependency
# and installs standalone. The fields match the contract exactly
# (label, score, box as (x1, y1, x2, y2) in native frame pixels).
@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


class MotionBlobDetector:
    """Box-provider plugin flagging moving blobs between keyframes."""

    label_categories = {"motion_blob": "MOTION_VISIBLE"}
    label_thresholds = {"motion_blob": 0.45}

    #: Pixel difference (0-255) a pixel must exceed to count as changed.
    delta_threshold = 24
    #: Blob area as a fraction of the frame's pixels to be worth censoring.
    min_area_fraction = 0.0005
    #: Downscale factor for the differencing work.
    work_scale = 0.5

    def __init__(self, settings=None):
        # ``settings`` carries detection_resolution etc.; the differencing
        # works at native resolution, so nothing is needed at construction.
        self.settings = settings
        self._prev_small: np.ndarray | None = None

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        for frame in frames_bgr:
            results.append(self._detect_one(frame))
        return results

    def unload(self) -> None:
        self._prev_small = None

    def _detect_one(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            self._prev_small = None
            return []

        h, w = frame_bgr.shape[:2]
        small = cv2.resize(
            frame_bgr,
            (max(1, int(w * self.work_scale)), max(1, int(h * self.work_scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        prev = self._prev_small
        self._prev_small = gray

        if prev is None or prev.shape != gray.shape:
            # First frame of a sequence (or a resolution change): nothing
            # to difference against.
            return []

        delta = cv2.absdiff(gray, prev)
        _, mask = cv2.threshold(delta, self.delta_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        min_area = gray.size * self.min_area_fraction
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            score = min(1.0, float(delta[mask > 0].mean()) / 255.0 * 4.0)
            x, y, bw, bh = cv2.boundingRect(contour)
            # Report in native frame pixels: the work frame was downscaled
            # by work_scale (and gray matches small, which is frame-scaled).
            sx = w / small.shape[1]
            sy = h / small.shape[0]
            x1 = max(0, int(x * sx))
            y1 = max(0, int(y * sy))
            x2 = min(w, int((x + bw) * sx))
            y2 = min(h, int((y + bh) * sy))
            if x2 > x1 and y2 > y1:
                detections.append(
                    Detection(
                        label="motion_blob",
                        score=round(score, 4),
                        box=(x1, y1, x2, y2),
                    )
                )
        return detections
