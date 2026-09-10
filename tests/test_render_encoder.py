"""Source-matched render encoder selection and per-encoder quality flags.

Re-encoded (censored) chunks are concatenated with stream-copied chunks of
the original and written into the input's own container, so the encoder
has to follow the source codec: WebM carries only VP8/VP9/AV1, AVI wants
MPEG-4 or Annex-B H.264, and the concat demuxer needs one codec throughout.
"""

from pathlib import Path
from unittest.mock import patch

from pureframe.hardware import HardwareProfile
from pureframe.utils.ffmpeg import (
    _encoder_preset_arg,
    _quality_args,
    container_bsf_args,
    encoder_codec,
    probe_video_codec,
    select_render_encoder,
)


class TestSelectRenderEncoder:
    def test_webm_and_avi_codecs_follow_the_source(self):
        for profile in HardwareProfile:
            assert select_render_encoder(profile, "h264", "vp9") == "libvpx-vp9"
            assert select_render_encoder(profile, "h264", "vp8") == "libvpx"
            assert select_render_encoder(profile, "h264", "mpeg4") == "mpeg4"

    def test_hevc_source_stays_hevc(self):
        assert select_render_encoder(HardwareProfile.CPU, "h264", "hevc") == "libx265"
        assert select_render_encoder(HardwareProfile.CPU, "h264", "h265") == "libx265"

    def test_h264_source_honors_output_codec(self):
        assert select_render_encoder(HardwareProfile.CPU, "h264", "h264") == "libx264"
        assert select_render_encoder(HardwareProfile.CPU, "hevc", "h264") == "libx265"

    def test_unknown_or_missing_source_uses_configured_codec(self):
        assert select_render_encoder(HardwareProfile.CPU, "h264", None) == "libx264"
        assert select_render_encoder(HardwareProfile.CPU, "h264", "av1") == "libx264"
        assert select_render_encoder(HardwareProfile.CPU, "h264", "") == "libx264"

    def test_gpu_profiles_still_pick_hardware_encoders_for_h264_and_hevc(self):
        with patch("pureframe.utils.ffmpeg.subprocess.check_output") as co:
            co.return_value = " V..... h264_nvenc\n V..... hevc_nvenc\n"
            assert (
                select_render_encoder(HardwareProfile.HIGH, "h264", "h264")
                == "h264_nvenc"
            )
            assert (
                select_render_encoder(HardwareProfile.HIGH, "h264", "hevc")
                == "hevc_nvenc"
            )
            # No hardware VP9 path: software encoder even on GPU profiles.
            assert (
                select_render_encoder(HardwareProfile.HIGH, "h264", "vp9")
                == "libvpx-vp9"
            )

    def test_codec_name_is_case_insensitive(self):
        assert select_render_encoder(HardwareProfile.CPU, "h264", "VP9") == "libvpx-vp9"


class TestQualityArgs:
    def test_x264_family_uses_crf(self):
        assert _quality_args("libx264", 20) == {"crf": 20}
        assert _quality_args("libx265", 23) == {"crf": 23}
        assert _quality_args("h264_nvenc", 20) == {"crf": 20}

    def test_vp9_constant_quality_needs_zero_bitrate(self):
        args = _quality_args("libvpx-vp9", 20)
        assert args["crf"] == 20 and args["b:v"] == 0
        assert _quality_args("libvpx-vp9", 99)["crf"] == 63

    def test_vp8_crf_floor_and_bitrate_cap(self):
        args = _quality_args("libvpx", 2)
        assert args["crf"] == 4 and args["b:v"]

    def test_mpeg4_maps_crf_onto_qscale(self):
        assert _quality_args("mpeg4", 20) == {"qscale:v": 8}
        assert _quality_args("mpeg4", 0) == {"qscale:v": 2}
        assert _quality_args("mpeg4", 200) == {"qscale:v": 31}
        assert "crf" not in _quality_args("mpeg4", 20)


class TestPresetGating:
    def test_libvpx_and_mpeg4_do_not_get_x264_presets(self):
        assert _encoder_preset_arg("libvpx", "veryfast") is None
        assert _encoder_preset_arg("libvpx-vp9", "veryfast") is None
        assert _encoder_preset_arg("mpeg4", "veryfast") is None


class TestContainerBsf:
    def test_avi_h264_needs_annexb(self):
        assert container_bsf_args(Path("o.avi"), "h264") == [
            "-bsf:v",
            "h264_mp4toannexb",
        ]
        assert container_bsf_args(Path("o.AVI"), "h264") == [
            "-bsf:v",
            "h264_mp4toannexb",
        ]

    def test_other_combinations_need_nothing(self):
        assert container_bsf_args(Path("o.avi"), "mpeg4") == []
        assert container_bsf_args(Path("o.mkv"), "h264") == []
        assert container_bsf_args(Path("o.mp4"), "h264") == []
        assert container_bsf_args(Path("o.avi"), None) == []


def test_encoder_codec_families():
    assert encoder_codec("libx264") == "h264"
    assert encoder_codec("h264_nvenc") == "h264"
    assert encoder_codec("libx265") == "hevc"
    assert encoder_codec("hevc_videotoolbox") == "hevc"
    assert encoder_codec("libvpx") == "vp8"
    assert encoder_codec("libvpx-vp9") == "vp9"
    assert encoder_codec("mpeg4") == "mpeg4"


def test_probe_video_codec_is_tolerant(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 64)
    assert probe_video_codec(junk) is None
    assert probe_video_codec(tmp_path / "missing.mp4") is None
