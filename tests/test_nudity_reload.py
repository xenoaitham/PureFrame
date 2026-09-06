"""Regression test: the NudeNet ONNX session must load once per plan.

The CPU/LOW profiles used to unload the detector after every detect_batch
call. Since densify_shot calls detect_batch once per frame, every densified
frame paid a full ONNX session re-initialization - minutes per shot on a
low-end machine.
"""

from unittest.mock import MagicMock, patch

from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.detect.nudity import NudityDetector


class _FakeNudeNet:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        return []


def test_detector_loaded_once_across_calls():
    settings = get_settings(HardwareProfile.CPU)
    assert settings.keep_models_loaded is False

    with patch(
        "pureframe.pipeline.detect.nudity.NudeDetector", side_effect=_FakeNudeNet
    ) as ctor:
        det = NudityDetector(settings)
        assert ctor.call_count == 0, "lazy profile must not load at construction"

        frame = MagicMock()
        det.detect_batch([frame])
        det.detect_batch([frame])
        det.detect_batch([frame])

        assert ctor.call_count == 1, (
            "detect_batch must reuse the ONNX session instead of reloading "
            "the model after every call"
        )


def test_detect_batch_no_reload_after_explicit_unload_cycle():
    """After cli's end-of-plan unload(), the next plan's first call reloads
    exactly once - not once per frame."""
    settings = get_settings(HardwareProfile.CPU)

    with patch(
        "pureframe.pipeline.detect.nudity.NudeDetector", side_effect=_FakeNudeNet
    ) as ctor:
        det = NudityDetector(settings)
        det.detect_batch([MagicMock()])
        det.unload()

        det.detect_batch([MagicMock()])
        det.detect_batch([MagicMock()])

        assert ctor.call_count == 2  # one per plan generation, not per frame
