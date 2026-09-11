"""The example plugin package must honor the plugin contract end to end.

These tests import ``pureframe_plugins_examples`` from ``examples/``
directly (sys.path, no install) and exercise the same machinery a real
installed plugin goes through: validate_plugin, EntryPoint resolution,
registration, and detect_batch on synthetic frames with a moving blob.
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np

EXAMPLES_SRC = (
    Path(__file__).parent.parent / "examples" / "pureframe-plugins-examples" / "src"
)


def _import_example_module():
    sys.path.insert(0, str(EXAMPLES_SRC))
    try:
        from pureframe_plugins_examples import motionblob
        from pureframe_plugins_examples.motionblob import MotionBlobDetector

        return motionblob, MotionBlobDetector
    finally:
        sys.path.remove(str(EXAMPLES_SRC))


def _static_frame(h=240, w=320):
    frame = np.full((h, w, 3), 90, dtype=np.uint8)
    # Texture so differencing has signal to work with.
    frame[::4, ::4] = 40
    return frame


def _frame_with_blob(h=240, w=320):
    frame = _static_frame(h, w)
    frame[80:160, 120:220] = 230
    return frame


class TestContract:
    def test_validate_plugin_passes(self):
        from pureframe.plugin_api import validate_plugin

        _, MotionBlobDetector = _import_example_module()
        assert validate_plugin(MotionBlobDetector) == []

    def test_resolves_through_a_real_entry_point(self):
        from pureframe.plugin_api import _resolve

        motionblob, MotionBlobDetector = _import_example_module()
        ep = EntryPoint(
            "motionblob",
            "pureframe_plugins_examples.motionblob:MotionBlobDetector",
            "pureframe.plugins",
        )
        registration = _resolve(ep)
        assert registration.cls is MotionBlobDetector
        assert registration.category_names() == {"MOTION_VISIBLE"}
        assert registration.category_threshold_bases() == {"MOTION_VISIBLE": 0.45}

    def test_declared_defaults_match_the_docs(self):
        _, MotionBlobDetector = _import_example_module()
        assert MotionBlobDetector.label_categories == {"motion_blob": "MOTION_VISIBLE"}
        assert MotionBlobDetector.label_thresholds == {"motion_blob": 0.45}


class TestDetection:
    def test_first_frame_yields_nothing(self):
        _, MotionBlobDetector = _import_example_module()
        detector = MotionBlobDetector(settings=None)
        assert detector.detect_batch([_static_frame()]) == [[]]

    def test_static_frames_yield_nothing(self):
        _, MotionBlobDetector = _import_example_module()
        detector = MotionBlobDetector(settings=None)
        results = detector.detect_batch([_static_frame() for _ in range(4)])
        assert all(dets == [] for dets in results)

    def test_moving_blob_is_detected_in_native_pixels(self):
        _, MotionBlobDetector = _import_example_module()
        detector = MotionBlobDetector(settings=None)
        results = detector.detect_batch(
            [_static_frame(), _frame_with_blob(), _frame_with_blob()]
        )
        # Frame 0 has no predecessor; the blob appears once it moves in.
        assert results[0] == []
        assert results[1], "blob arrival must be flagged"
        assert results[2] == [], "a steady blob is no longer new motion"

        for det in results[1]:
            assert det.label == "motion_blob"
            assert 0.0 <= det.score <= 1.0
            x1, y1, x2, y2 = det.box
            h, w = 240, 320
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            # The blob lives at [80:160, 120:220]; the box must cover its
            # center region in native pixels.
            assert x1 < 220 and x2 > 120
            assert y1 < 160 and y2 > 80

    def test_unload_clears_state(self):
        _, MotionBlobDetector = _import_example_module()
        detector = MotionBlobDetector(settings=None)
        detector.detect_batch([_static_frame(), _frame_with_blob()])
        detector.unload()
        assert detector._prev_small is None
        # After unload, the next frame is a first frame again.
        assert detector.detect_batch([_frame_with_blob()]) == [[]]
