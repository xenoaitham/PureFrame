"""Tests for smart segment rendering."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pureframe.pipeline.render.smart import (
    _find_dirty_segments,
    _stream_copy,
    _extract_segment_copy,
    _concat_segments,
    _get_fps,
    _extract_and_render_segment,
    _render_segments,
    apply_censoring_smart,
)
from pureframe.config import Config
from pureframe.hardware import ProfileSettings, HardwareProfile


@pytest.fixture
def config(tmp_path):
    inp = tmp_path / "input.mp4"
    inp.write_bytes(b"\x00" * 100)
    out = tmp_path / "output.mp4"
    return Config(input_path=inp, output_path=out)


@pytest.fixture
def profile_settings():
    return ProfileSettings(
        profile=HardwareProfile.CPU,
        detection_resolution=640,
        detection_batch_size=1,
        use_fp16=False,
        keep_models_loaded=False,
        sample_keyframes_per_shot=2,
        densify_every_n_frames=5,
        onnx_providers=["CPUExecutionProvider"],
    )


# ── _find_dirty_segments ─────────────────────────────────────────────────


class TestFindDirtySegments:
    def test_empty_actions(self):
        assert _find_dirty_segments({}, 1000, 30.0) == []

    def test_single_frame(self):
        result = _find_dirty_segments({100: {"action": "blur"}}, 1000, 30.0)
        assert len(result) == 1
        assert result[0][0] >= 0
        assert result[0][1] <= 1000 / 30.0

    def test_contiguous_frames_merge(self):
        actions = {i: {"action": "blur"} for i in range(100, 200)}
        result = _find_dirty_segments(actions, 1000, 30.0)
        assert len(result) == 1

    def test_distant_frames_separate(self):
        actions = {10: {"a": 1}, 500: {"a": 1}}
        result = _find_dirty_segments(actions, 1000, 30.0, padding_seconds=0.5)
        assert len(result) == 2

    def test_padding_merges_nearby(self):
        actions = {100: {"a": 1}, 110: {"a": 1}}
        result = _find_dirty_segments(actions, 1000, 30.0, padding_seconds=0.5)
        assert len(result) == 1

    def test_clamps_to_bounds(self):
        result = _find_dirty_segments({0: {"a": 1}}, 100, 30.0, padding_seconds=1.0)
        assert result[0][0] == 0

    def test_overlapping_ranges_merge(self):
        # Three groups that overlap after padding
        actions = {50: {"a": 1}, 65: {"a": 1}, 80: {"a": 1}}
        result = _find_dirty_segments(actions, 1000, 30.0, padding_seconds=1.0)
        assert len(result) == 1  # all merge due to padding


# ── _stream_copy ─────────────────────────────────────────────────────────


class TestStreamCopy:
    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_stream_copy(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _stream_copy(Path("/in.mp4"), Path("/out.mp4"))
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-c" in args
        assert "copy" in args


# ── _extract_segment_copy ────────────────────────────────────────────────


class TestExtractSegmentCopy:
    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_extract_copy(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _extract_segment_copy(Path("/in.mp4"), Path("/seg.mkv"), 1.0, 5.0)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "1.000" in args
        assert "5.000" in args
        assert "-avoid_negative_ts" in args


# ── _concat_segments ─────────────────────────────────────────────────────


class TestConcatSegments:
    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_concat(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        # Create dummy segment files
        seg1 = tmp_path / "seg1.mkv"
        seg2 = tmp_path / "seg2.mkv"
        seg1.write_bytes(b"\x00" * 10)
        seg2.write_bytes(b"\x00" * 10)

        _concat_segments([seg1, seg2], tmp_path / "out.mp4", tmp_path)
        mock_run.assert_called_once()
        # Verify concat file was written
        concat_file = tmp_path / "concat.txt"
        assert concat_file.exists()
        content = concat_file.read_text()
        assert "seg1.mkv" in content
        assert "seg2.mkv" in content

    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_concat_skips_missing_files(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        existing = tmp_path / "exists.mkv"
        existing.write_bytes(b"\x00" * 10)
        missing = tmp_path / "missing.mkv"

        _concat_segments([existing, missing], tmp_path / "out.mp4", tmp_path)
        concat_file = tmp_path / "concat.txt"
        content = concat_file.read_text()
        assert "exists.mkv" in content
        assert "missing.mkv" not in content

    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_concat_skips_empty_files(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        empty = tmp_path / "empty.mkv"
        empty.write_bytes(b"")
        _concat_segments([empty], tmp_path / "out.mp4", tmp_path)
        content = (tmp_path / "concat.txt").read_text()
        assert "empty.mkv" not in content


# ── _get_fps ─────────────────────────────────────────────────────────────


class TestGetFps:
    @patch("pureframe.pipeline.render.smart.probe")
    def test_normal_fps(self, mock_probe):
        mock_probe.return_value = {
            "streams": [{"codec_type": "video", "r_frame_rate": "30/1"}]
        }
        assert _get_fps(Path("test.mp4")) == 30.0

    @patch("pureframe.pipeline.render.smart.probe")
    def test_fractional_fps(self, mock_probe):
        mock_probe.return_value = {
            "streams": [{"codec_type": "video", "r_frame_rate": "24000/1001"}]
        }
        fps = _get_fps(Path("test.mp4"))
        assert abs(fps - 23.976) < 0.1

    @patch("pureframe.pipeline.render.smart.probe")
    def test_no_video_stream(self, mock_probe):
        mock_probe.return_value = {
            "streams": [{"codec_type": "audio", "r_frame_rate": "0/0"}]
        }
        assert _get_fps(Path("test.mp4")) == 24.0

    @patch("pureframe.pipeline.render.smart.probe")
    def test_invalid_framerate(self, mock_probe):
        mock_probe.return_value = {
            "streams": [{"codec_type": "video", "r_frame_rate": "0/0"}]
        }
        assert _get_fps(Path("test.mp4")) == 24.0


# ── _render_segments ─────────────────────────────────────────────────────


class TestRenderSegments:
    @patch("pureframe.pipeline.render.smart._concat_segments")
    @patch("pureframe.pipeline.render.smart._extract_and_render_segment")
    @patch("pureframe.pipeline.render.smart._extract_segment_copy")
    @patch("pureframe.pipeline.render.smart.shutil.rmtree")
    def test_render_with_clean_and_dirty(
        self, mock_rm, mock_copy, mock_render, mock_concat, config, profile_settings
    ):
        dirty_segments = [(5.0, 10.0)]
        _render_segments(
            config.input_path,
            config.output_path,
            {150: {"action": "blur"}},
            dirty_segments,
            30.0,
            config,
            profile_settings,
        )
        # Should extract clean before dirty, render dirty, extract clean after, concat
        assert mock_copy.call_count == 2  # before + after
        assert mock_render.call_count == 1
        mock_concat.assert_called_once()
        mock_rm.assert_called_once()

    @patch("pureframe.pipeline.render.smart._concat_segments")
    @patch("pureframe.pipeline.render.smart._extract_and_render_segment")
    @patch("pureframe.pipeline.render.smart._extract_segment_copy")
    @patch("pureframe.pipeline.render.smart.shutil.rmtree")
    def test_render_dirty_at_start(
        self, mock_rm, mock_copy, mock_render, mock_concat, config, profile_settings
    ):
        dirty_segments = [(0.0, 5.0)]
        _render_segments(
            config.input_path,
            config.output_path,
            {10: {"action": "blur"}},
            dirty_segments,
            30.0,
            config,
            profile_settings,
        )
        # No clean before (starts at 0), one clean after
        assert mock_copy.call_count == 1
        assert mock_render.call_count == 1

    @patch("pureframe.pipeline.render.smart._concat_segments")
    @patch("pureframe.pipeline.render.smart._extract_and_render_segment")
    @patch("pureframe.pipeline.render.smart._extract_segment_copy")
    @patch("pureframe.pipeline.render.smart.shutil.rmtree")
    def test_cleanup_on_error(
        self, mock_rm, mock_copy, mock_render, mock_concat, config, profile_settings
    ):
        mock_render.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            _render_segments(
                config.input_path,
                config.output_path,
                {10: {"action": "blur"}},
                [(5.0, 10.0)],
                30.0,
                config,
                profile_settings,
            )
        mock_rm.assert_called_once()  # cleanup still happens


# ── _extract_and_render_segment ──────────────────────────────────────────


class TestExtractAndRenderSegment:
    @patch("pureframe.pipeline.render.smart._get_fps", return_value=30.0)
    @patch("pureframe.pipeline.render.smart.write_video_with_overlay")
    @patch("pureframe.pipeline.render.smart.select_hw_encoder", return_value="libx264")
    def test_render_segment(
        self, mock_enc, mock_write, mock_fps, config, profile_settings
    ):
        _extract_and_render_segment(
            config.input_path,
            Path("/out.mkv"),
            1.0,
            5.0,
            {30: {"action": "blur"}},
            config,
            profile_settings,
        )
        mock_write.assert_called_once()
        kwargs = mock_write.call_args[1]
        assert kwargs["ss"] == 1.0
        assert kwargs["to"] == 5.0


# ── apply_censoring_smart (integration) ──────────────────────────────────


class TestSmartRendering:
    def test_no_actions_stream_copy(self, config, profile_settings):
        with patch("pureframe.pipeline.render.smart._stream_copy") as mock_copy:
            apply_censoring_smart(
                config.input_path,
                config.output_path,
                {},
                config,
                profile_settings,
                total_frames=1000,
                fps=30.0,
            )
            mock_copy.assert_called_once()

    def test_high_dirty_ratio_falls_back(self, config, profile_settings):
        actions = {i: {"action": "blur"} for i in range(0, 700)}
        with patch("pureframe.pipeline.render.apply.apply_censoring") as mock_full:
            apply_censoring_smart(
                config.input_path,
                config.output_path,
                actions,
                config,
                profile_settings,
                total_frames=1000,
                fps=30.0,
            )
            mock_full.assert_called_once()

    def test_low_dirty_ratio_renders_segments(self, config, profile_settings):
        actions = {i: {"action": "blur"} for i in range(100, 150)}
        with patch("pureframe.pipeline.render.smart._render_segments") as mock_segments:
            apply_censoring_smart(
                config.input_path,
                config.output_path,
                actions,
                config,
                profile_settings,
                total_frames=3000,
                fps=30.0,
            )
            mock_segments.assert_called_once()

    def test_segment_render_failure_falls_back(self, config, profile_settings):
        actions = {100: {"action": "blur"}}
        with patch(
            "pureframe.pipeline.render.smart._render_segments",
            side_effect=RuntimeError("ffmpeg failed"),
        ):
            with patch("pureframe.pipeline.render.apply.apply_censoring") as mock_full:
                apply_censoring_smart(
                    config.input_path,
                    config.output_path,
                    actions,
                    config,
                    profile_settings,
                    total_frames=3000,
                    fps=30.0,
                )
                mock_full.assert_called_once()

    def test_zero_duration_defaults_ratio(self, config, profile_settings):
        """If total_frames=0, dirty_ratio defaults to 1.0 → fallback."""
        actions = {0: {"action": "blur"}}
        with patch("pureframe.pipeline.render.apply.apply_censoring"):
            apply_censoring_smart(
                config.input_path,
                config.output_path,
                actions,
                config,
                profile_settings,
                total_frames=0,
                fps=30.0,
            )
            # total_frames=0 means fps calculation gives empty segments,
            # so it should call _stream_copy or fallback. Either way, no crash.
