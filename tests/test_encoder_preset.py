"""Encoder preset gating: x264-scale presets are software-encoder-only."""

from pureframe.utils.ffmpeg import _encoder_preset_arg


def test_software_encoders_get_the_preset():
    assert _encoder_preset_arg("libx264", "veryfast") == "veryfast"
    assert _encoder_preset_arg("libx265", "faster") == "faster"


def test_hardware_encoders_reject_x264_presets():
    # nvenc errors with 'Unable to parse option value "veryfast"' - the
    # regression that failed the first full bench run on GPU profiles.
    assert _encoder_preset_arg("h264_nvenc", "veryfast") is None
    assert _encoder_preset_arg("hevc_nvenc", "medium") is None
    assert _encoder_preset_arg("h264_qsv", "faster") is None
    assert _encoder_preset_arg("h264_videotoolbox", "veryfast") is None
    assert _encoder_preset_arg("h264_amf", "veryfast") is None


def test_no_preset_passthrough():
    assert _encoder_preset_arg("libx264", None) is None
