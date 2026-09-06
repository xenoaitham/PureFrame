"""Tests for the speed-offensive model features: int8 quantization wiring
and the PANNs analysis-window cap."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pureframe.config import Config
from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.detect.audio import analysis_window
from pureframe.pipeline.detect.nudity import NudityDetector


class _FakeNudeNet:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        return []


def _tmp_cfg(tmp_path, **kwargs) -> Config:
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    return Config.from_cli(input_path=src, output_path=tmp_path / "o.mp4", **kwargs)


def test_cpu_profile_uses_quantized_model(tmp_path):
    settings = get_settings(HardwareProfile.CPU)
    with (
        patch(
            "pureframe.pipeline.detect.nudity.NudeDetector",
            side_effect=_FakeNudeNet,
        ) as ctor,
        patch(
            "pureframe.pipeline.detect.quantize.quantized_model_path",
            return_value=Path("/cache/nudenet_int8.onnx"),
        ) as quant,
    ):
        det = NudityDetector(settings, quantize=True)
        det.detect_batch([MagicMock()])
        quant.assert_called_once()
        _, kwargs = ctor.call_args
        # Compare via Path so the assertion holds on Windows separators too.
        assert Path(kwargs.get("model_path")) == Path("/cache/nudenet_int8.onnx")


def test_quantize_disabled_uses_fp32(tmp_path):
    settings = get_settings(HardwareProfile.CPU)
    with (
        patch(
            "pureframe.pipeline.detect.nudity.NudeDetector",
            side_effect=_FakeNudeNet,
        ) as ctor,
        patch("pureframe.pipeline.detect.quantize.quantized_model_path") as quant,
    ):
        det = NudityDetector(settings, quantize=False)
        det.detect_batch([MagicMock()])
        quant.assert_not_called()
        _, kwargs = ctor.call_args
        assert "model_path" not in kwargs or kwargs.get("model_path") is None


def test_gpu_profile_skips_quantization(tmp_path):
    settings = get_settings(HardwareProfile.HIGH)
    assert settings.onnx_providers != ["CPUExecutionProvider"]
    with (
        patch(
            "pureframe.pipeline.detect.nudity.NudeDetector",
            side_effect=_FakeNudeNet,
        ) as ctor,
        patch("pureframe.pipeline.detect.quantize.quantized_model_path") as quant,
    ):
        det = NudityDetector(settings, quantize=True)
        det.detect_batch([MagicMock()])
        quant.assert_not_called()
        _, kwargs = ctor.call_args
        assert kwargs.get("model_path") is None


def test_quantization_failure_falls_back_to_fp32(tmp_path):
    settings = get_settings(HardwareProfile.CPU)
    with (
        patch(
            "pureframe.pipeline.detect.nudity.NudeDetector",
            side_effect=_FakeNudeNet,
        ) as ctor,
        patch(
            "pureframe.pipeline.detect.quantize.quantized_model_path",
            side_effect=RuntimeError("quantize exploded"),
        ),
    ):
        det = NudityDetector(settings, quantize=True)
        det.detect_batch([MagicMock()])
        _, kwargs = ctor.call_args
        assert kwargs.get("model_path") is None


def test_config_quantize_default_and_hash(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    assert cfg.quantize_cpu is True
    other = _tmp_cfg(tmp_path, quantize_cpu=False)
    assert cfg.config_hash != other.config_hash


def test_analysis_window_short_segment_unchanged():
    assert analysis_window(10.0, 15.0) == (10.0, 15.0)


def test_analysis_window_caps_long_segment():
    start, end = analysis_window(0.0, 120.0)
    assert abs((end - start) - 10.0) < 1e-9
    assert abs(((start + end) / 2) - 60.0) < 1e-9
