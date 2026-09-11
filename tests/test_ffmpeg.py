"""Tests for FFmpeg utility functions - metadata extraction, encoder selection, probe."""

from fractions import Fraction

import pytest

from pureframe.hardware import HardwareProfile
from pureframe.utils.ffmpeg import (
    PureFrameError,
    VideoMetadata,
    extract_metadata,
    probe,
    select_hw_encoder,
)


class TestProbe:
    def test_probe_valid_video(self, synthetic_video):
        result = probe(synthetic_video)
        assert isinstance(result, dict)
        assert "format" in result
        assert "streams" in result

    def test_probe_nonexistent_raises(self, tmp_path):
        with pytest.raises(PureFrameError, match="Failed to probe|FFprobe error"):
            probe(tmp_path / "nonexistent.mp4")

    def test_probe_non_video_raises(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("not a video")
        with pytest.raises(PureFrameError):
            probe(txt)


class TestExtractMetadata:
    def test_extract_from_synthetic(self, synthetic_video):
        probe_result = probe(synthetic_video)
        meta = extract_metadata(probe_result)

        assert isinstance(meta, VideoMetadata)
        assert meta.width == 1280
        assert meta.height == 720
        assert float(meta.fps) > 0
        assert meta.duration_seconds > 0
        assert meta.total_frames > 0
        assert meta.has_audio is True
        assert meta.container != "unknown"
        assert meta.video_codec != "unknown"

    def test_extract_fps_fallback(self):
        """Test that invalid fps falls back to 24."""
        fake_probe = {
            "format": {"duration": "10.0", "format_name": "test"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "invalid_fps",
                    "nb_frames": "0",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "color_space": "unknown",
                    "color_transfer": "unknown",
                }
            ],
        }
        meta = extract_metadata(fake_probe)
        assert meta.fps == Fraction(24, 1)

    def test_extract_no_video_stream_raises(self):
        fake_probe = {
            "format": {"duration": "10.0"},
            "streams": [{"codec_type": "audio"}],
        }
        with pytest.raises(PureFrameError, match="No video stream"):
            extract_metadata(fake_probe)

    def test_hdr_detection(self):
        """Test HDR detection from color_transfer metadata."""
        fake_probe = {
            "format": {"duration": "10.0", "format_name": "test"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 3840,
                    "height": 2160,
                    "r_frame_rate": "30/1",
                    "nb_frames": "300",
                    "codec_name": "hevc",
                    "pix_fmt": "yuv420p10le",
                    "color_space": "bt2020nc",
                    "color_transfer": "smpte2084",
                }
            ],
        }
        meta = extract_metadata(fake_probe)
        assert meta.is_hdr is True

    def test_sdr_detection(self):
        fake_probe = {
            "format": {"duration": "10.0", "format_name": "test"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "24/1",
                    "nb_frames": "240",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "color_space": "bt709",
                    "color_transfer": "bt709",
                }
            ],
        }
        meta = extract_metadata(fake_probe)
        assert meta.is_hdr is False

    def test_metadata_serializable(self, synthetic_video):
        probe_result = probe(synthetic_video)
        meta = extract_metadata(probe_result)
        data = meta.model_dump()
        assert "width" in data
        assert "fps" in data
        assert "duration_seconds" in data


class TestSelectHwEncoder:
    def test_cpu_always_software(self):
        encoder = select_hw_encoder(HardwareProfile.CPU, "h264")
        assert encoder == "libx264"

    def test_cpu_hevc_software(self):
        encoder = select_hw_encoder(HardwareProfile.CPU, "hevc")
        assert encoder == "libx265"

    def test_high_returns_valid_encoder(self):
        """High profile should return some encoder - hw or sw fallback."""
        encoder = select_hw_encoder(HardwareProfile.HIGH, "h264")
        assert encoder in [
            "libx264",
            "h264_nvenc",
            "h264_videotoolbox",
            "h264_qsv",
            "h264_amf",
        ]

    def test_hevc_returns_valid_encoder(self):
        encoder = select_hw_encoder(HardwareProfile.HIGH, "hevc")
        assert encoder in [
            "libx265",
            "hevc_nvenc",
            "hevc_videotoolbox",
            "hevc_qsv",
            "hevc_amf",
        ]

    def test_h265_alias(self):
        encoder = select_hw_encoder(HardwareProfile.CPU, "h265")
        assert encoder == "libx265"
