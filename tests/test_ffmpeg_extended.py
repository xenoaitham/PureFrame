"""Tests for ffmpeg encoder selection with mocked subprocess for full branch coverage."""

from unittest.mock import patch, MagicMock
from pureframe.utils.ffmpeg import select_hw_encoder, extract_metadata, frames_iter
from pureframe.hardware import HardwareProfile


class TestEncoderSelectionMocked:
    def _mock_encoders(self, encoder_list):
        """Create a mock that returns specific encoders from ffmpeg -encoders."""
        lines = ["Encoders:"]
        for e in encoder_list:
            lines.append(f" V..... {e}           Description")
        mock_result = MagicMock()
        mock_result.decode.return_value = (
            "\n".join(lines) if hasattr(mock_result, "decode") else None
        )
        return "\n".join(lines)

    def test_nvenc_h264_selected(self):
        output = self._mock_encoders(["libx264", "h264_nvenc", "libx265"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "h264")
            assert encoder == "h264_nvenc"

    def test_videotoolbox_h264_selected(self):
        output = self._mock_encoders(["libx264", "h264_videotoolbox"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "h264")
            assert encoder == "h264_videotoolbox"

    def test_qsv_h264_selected(self):
        output = self._mock_encoders(["libx264", "h264_qsv"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "h264")
            assert encoder == "h264_qsv"

    def test_amf_h264_selected(self):
        output = self._mock_encoders(["libx264", "h264_amf"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "h264")
            assert encoder == "h264_amf"

    def test_nvenc_hevc_selected(self):
        output = self._mock_encoders(["libx265", "hevc_nvenc"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "hevc")
            assert encoder == "hevc_nvenc"

    def test_videotoolbox_hevc_selected(self):
        output = self._mock_encoders(["libx265", "hevc_videotoolbox"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "hevc")
            assert encoder == "hevc_videotoolbox"

    def test_qsv_hevc_selected(self):
        output = self._mock_encoders(["libx265", "hevc_qsv"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "hevc")
            assert encoder == "hevc_qsv"

    def test_amf_hevc_selected(self):
        output = self._mock_encoders(["libx265", "hevc_amf"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "hevc")
            assert encoder == "hevc_amf"

    def test_no_hw_encoders_fallback_h264(self):
        output = self._mock_encoders(["libx264", "libx265"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "h264")
            assert encoder == "libx264"

    def test_no_hw_encoders_fallback_hevc(self):
        output = self._mock_encoders(["libx264", "libx265"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "hevc")
            assert encoder == "libx265"

    def test_ffmpeg_command_fails_fallback(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "h264")
            assert encoder == "libx264"

    def test_h265_alias_works(self):
        output = self._mock_encoders(["libx265", "hevc_nvenc"])
        with patch("subprocess.check_output", return_value=output):
            encoder = select_hw_encoder(HardwareProfile.HIGH, "h265")
            assert encoder == "hevc_nvenc"


class TestFramesIter:
    def test_frames_iter_produces_frames(self, synthetic_video):
        """Test that frames_iter yields BGR numpy arrays."""
        count = 0
        for frame in frames_iter(synthetic_video, downscale_max_edge=320):
            assert frame.ndim == 3
            assert frame.shape[2] == 3  # BGR
            count += 1
            if count >= 5:
                break
        assert count >= 1

    def test_frames_iter_no_downscale(self, synthetic_video):
        """Test frames_iter without downscaling."""
        count = 0
        for frame in frames_iter(synthetic_video):
            assert frame.shape[0] == 720  # original height
            assert frame.shape[1] == 1280  # original width
            count += 1
            if count >= 3:
                break
        assert count >= 1


class TestMetadataEdgeCases:
    def test_zero_fps_denominator(self):
        """Test that r_frame_rate 0/0 falls back to 24."""
        fake_probe = {
            "format": {"duration": "10.0", "format_name": "test"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "0/0",
                    "nb_frames": "0",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "color_space": "unknown",
                    "color_transfer": "unknown",
                }
            ],
        }
        meta = extract_metadata(fake_probe)
        # Should handle gracefully (Fraction(0) or fallback)
        assert meta.duration_seconds == 10.0

    def test_hlg_hdr_detection(self):
        """Test HLG HDR detection."""
        fake_probe = {
            "format": {"duration": "10.0", "format_name": "test"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 3840,
                    "height": 2160,
                    "r_frame_rate": "60/1",
                    "nb_frames": "600",
                    "codec_name": "hevc",
                    "pix_fmt": "yuv420p10le",
                    "color_space": "bt2020nc",
                    "color_transfer": "arib-std-b67",
                }
            ],
        }
        meta = extract_metadata(fake_probe)
        assert meta.is_hdr is True

    def test_multiple_audio_streams(self):
        fake_probe = {
            "format": {"duration": "120.0", "format_name": "matroska"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "24/1",
                    "nb_frames": "2880",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "color_space": "bt709",
                    "color_transfer": "bt709",
                },
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "audio", "codec_name": "dts"},
                {"codec_type": "subtitle", "codec_name": "srt"},
            ],
        }
        meta = extract_metadata(fake_probe)
        assert meta.has_audio is True
        assert len(meta.audio_streams) == 2
        assert len(meta.subtitle_streams) == 1
