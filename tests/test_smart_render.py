"""Tests for smart segment rendering."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import pytest

from pureframe.config import Config
from pureframe.hardware import HardwareProfile, ProfileSettings
from pureframe.pipeline.render.smart import (
    _concat_segments,
    _extract_and_render_segment,
    _find_dirty_segments,
    _get_fps,
    _probe_keyframe_times,
    _render_segments,
    _segment_at_boundaries,
    _snap_to_keyframes,
    _stream_copy,
    apply_censoring_smart,
)


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


# ── _probe_keyframe_times ────────────────────────────────────────────────


class TestProbeKeyframeTimes:
    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_keeps_only_keyframe_packets(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="0.0,K__\n0.5,___\n1.0,K__\n1.5,__D\n2.5,K__\n", returncode=0
        )
        assert _probe_keyframe_times(Path("/in.mp4")) == [0.0, 1.0, 2.5]
        argv = mock_run.call_args[0][0]
        # Demux-only probe: packet flags, never a decode pass.
        assert "packet=pts_time,flags" in argv
        assert "-skip_frame" not in argv

    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_skips_packets_without_pts(self, mock_run):
        # AVI/H.264 carries no presentation timestamps — no usable map.
        mock_run.return_value = MagicMock(
            stdout="N/A,K__\nN/A,___\nN/A,K__\n", returncode=0
        )
        with pytest.raises(RuntimeError, match="no keyframes"):
            _probe_keyframe_times(Path("/in.avi"))

    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_raises_on_no_keyframes(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        with pytest.raises(RuntimeError, match="no keyframes"):
            _probe_keyframe_times(Path("/in.mp4"))

    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_raises_on_ffprobe_failure(self, mock_run):
        mock_run.side_effect = RuntimeError("ffprobe exploded")
        with pytest.raises(RuntimeError):
            _probe_keyframe_times(Path("/in.mp4"))


# ── _snap_to_keyframes ───────────────────────────────────────────────────


class TestSnapToKeyframes:
    def test_snaps_outward(self):
        result = _snap_to_keyframes([(2.8, 5.8)], [0.0, 1.0, 3.0, 6.0], 10.0)
        assert result == [(1.0, 6.0)]

    def test_tail_defaults_to_duration(self):
        result = _snap_to_keyframes([(8.0, 9.5)], [0.0, 5.0], 10.0)
        assert result == [(5.0, 10.0)]

    def test_head_defaults_to_zero(self):
        # Start before the first keyframe → 0.0; end snaps forward to the
        # next keyframe (2.0 lies inside the [1.0, 5.0] GOP).
        result = _snap_to_keyframes([(0.5, 2.0)], [1.0, 5.0], 10.0)
        assert result == [(0.0, 5.0)]

    def test_merges_ranges_that_snap_into_overlap(self):
        result = _snap_to_keyframes(
            [(2.0, 2.5), (2.8, 3.2)], [0.0, 1.0, 3.0, 4.0, 10.0], 10.0
        )
        # (2.0,2.5) → (1.0,3.0); (2.8,3.2) → (1.0,4.0); merged into one
        assert result == [(1.0, 4.0)]

    def test_degenerate_range_clamps(self):
        result = _snap_to_keyframes([(0.0, -1.0)], [0.0], 0.0)
        assert result == [(0.0, 0.0)]


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


# ── _segment_at_boundaries ───────────────────────────────────────────────


class TestSegmentAtBoundaries:
    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_filters_trivial_boundaries(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        for i in range(2):
            (tmp_path / f"chunk_{i:04d}.mkv").write_bytes(b"x")
        chunks = _segment_at_boundaries(
            Path("/in.mp4"), tmp_path, [0.0, 5.0, 30.0], 30.0
        )
        assert len(chunks) == 2
        argv = mock_run.call_args[0][0]
        times = argv[argv.index("-segment_times") + 1]
        assert times == "5.000"

    @patch("pureframe.pipeline.render.smart.subprocess.run")
    def test_raises_when_chunk_missing(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(RuntimeError, match="no chunk"):
            _segment_at_boundaries(Path("/in.mp4"), tmp_path, [5.0], 30.0)


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


def _make_chunks(tmp_path, n):
    chunks = []
    for i in range(n):
        c = tmp_path / f"chunk_{i:04d}.mkv"
        c.write_bytes(b"\x00" * 10)
        chunks.append(c)
    return chunks


class TestRenderSegments:
    @patch("pureframe.pipeline.render.smart._concat_segments")
    @patch("pureframe.pipeline.render.smart._extract_and_render_segment")
    @patch("pureframe.pipeline.render.smart._segment_at_boundaries")
    @patch("pureframe.pipeline.render.smart.shutil.rmtree")
    def test_render_with_clean_and_dirty(
        self,
        mock_rm,
        mock_seg,
        mock_render,
        mock_concat,
        config,
        profile_settings,
        tmp_path,
    ):
        chunks = _make_chunks(tmp_path, 3)
        mock_seg.return_value = chunks
        _render_segments(
            config.input_path,
            config.output_path,
            {150: {"action": "blur"}},
            [(5.0, 10.0)],
            30.0,
            config,
            profile_settings,
        )
        # Split at dirty boundaries; only the overlapping chunk re-renders
        mock_seg.assert_called_once()
        assert mock_render.call_count == 1
        merged = mock_concat.call_args[0][0]
        assert merged == [chunks[0], merged[1], chunks[2]]
        assert merged[1].name.startswith("dirty_")
        mock_rm.assert_called_once()

    @patch("pureframe.pipeline.render.smart._concat_segments")
    @patch("pureframe.pipeline.render.smart._extract_and_render_segment")
    @patch("pureframe.pipeline.render.smart._segment_at_boundaries")
    @patch("pureframe.pipeline.render.smart.shutil.rmtree")
    def test_render_dirty_at_start(
        self,
        mock_rm,
        mock_seg,
        mock_render,
        mock_concat,
        config,
        profile_settings,
        tmp_path,
    ):
        chunks = _make_chunks(tmp_path, 2)
        mock_seg.return_value = chunks
        _render_segments(
            config.input_path,
            config.output_path,
            {10: {"action": "blur"}},
            [(0.0, 5.0)],
            30.0,
            config,
            profile_settings,
        )
        # Chunk [0,5) is dirty, [5,30) passes through
        assert mock_render.call_count == 1
        merged = mock_concat.call_args[0][0]
        assert merged == [merged[0], chunks[1]]
        assert merged[0].name.startswith("dirty_")

    @patch("pureframe.pipeline.render.smart._concat_segments")
    @patch("pureframe.pipeline.render.smart._extract_and_render_segment")
    @patch("pureframe.pipeline.render.smart._segment_at_boundaries")
    @patch("pureframe.pipeline.render.smart.shutil.rmtree")
    def test_cleanup_on_error(
        self,
        mock_rm,
        mock_seg,
        mock_render,
        mock_concat,
        config,
        profile_settings,
        tmp_path,
    ):
        mock_seg.return_value = _make_chunks(tmp_path, 3)
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

    @patch(
        "pureframe.pipeline.render.smart._probe_keyframe_times",
        return_value=[0.0, 33.333],
    )
    def test_high_dirty_ratio_falls_back(self, _probe, config, profile_settings):
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

    @patch(
        "pureframe.pipeline.render.smart._probe_keyframe_times",
        return_value=[0.0, 2.8, 5.9, 100.0],
    )
    def test_low_dirty_ratio_renders_segments(self, _probe, config, profile_settings):
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

    @patch(
        "pureframe.pipeline.render.smart._probe_keyframe_times",
        return_value=[0.0, 2.8, 5.9, 100.0],
    )
    def test_accepts_fraction_fps(self, _probe, config, profile_settings):
        # execute_render passes plan.input_metadata.fps, a Fraction. The
        # "Smart render: …s" log line formats durations with :.1f, which
        # Fraction only supports from Python 3.12 — on 3.11 the smart path
        # crashed before reaching its fallback try-block.
        from fractions import Fraction

        actions = {i: {"action": "blur"} for i in range(100, 150)}
        with patch("pureframe.pipeline.render.smart._render_segments") as mock_segments:
            apply_censoring_smart(
                config.input_path,
                config.output_path,
                actions,
                config,
                profile_settings,
                total_frames=3000,
                fps=Fraction(30000, 1001),
            )
            mock_segments.assert_called_once()
            assert isinstance(mock_segments.call_args.kwargs["fps"], float)

    @patch(
        "pureframe.pipeline.render.smart._probe_keyframe_times",
        return_value=[0.0, 2.8, 3.9, 100.0],
    )
    def test_segment_render_failure_falls_back(self, _probe, config, profile_settings):
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

    @patch(
        "pureframe.pipeline.render.smart._probe_keyframe_times",
        return_value=[0.0],
    )
    def test_zero_duration_defaults_ratio(self, _probe, config, profile_settings):
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


# ── keyframe alignment (the stream-copy duplication regression) ─────────


class TestSmartRenderKeyframeAlignment:
    """End-to-end guard for the keyframe-snap contract.

    A stream-copy cut that is not on a keyframe snaps back to the previous
    sync point and duplicates everything from there — the output grows and
    replayed content shows up after the censored section (observed as a
    9.1s render of a 5s clip whose demo GIF then "played the original"
    after the blur). With ``-g 15`` there are keyframes every second, so
    the smart path stays a real segment render; the output must keep the
    input's exact frame count, with blur present only in the dirty window.
    """

    def test_output_preserves_frame_count_and_blur_window(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=6:size=320x240:rate=15",
                "-c:v",
                "libx264",
                "-g",
                "15",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ],
            check=True,
            capture_output=True,
            shell=False,
        )
        out = tmp_path / "out.mp4"
        config = Config(input_path=clip, output_path=out)
        profile_settings = ProfileSettings(
            profile=HardwareProfile.CPU,
            detection_resolution=480,
            detection_batch_size=1,
            use_fp16=False,
            keep_models_loaded=True,
            sample_keyframes_per_shot=2,
            densify_every_n_frames=5,
            onnx_providers=["CPUExecutionProvider"],
        )

        # Dirty window 2.0s–3.0s (frames 30–44), padded to 1.5–3.5s and
        # keyframe-snapped to 1.0–4.0s. Box in native coords (320px wide is
        # below the CPU profile's 480px detection resolution → no scaling).
        frame_actions = {
            f: {
                "action": "BLACK_BOX",
                "boxes": [(100, 80, 220, 160)],
            }
            for f in range(30, 45)
        }
        apply_censoring_smart(
            clip,
            out,
            frame_actions,
            config,
            profile_settings,
            total_frames=90,
            fps=15.0,
        )

        assert out.exists() and out.stat().st_size > 0

        def nb_frames(path: Path) -> int:
            res = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            )
            return int(res.stdout.strip())

        assert nb_frames(out) == nb_frames(clip) == 90

        def roi_lap_var(frame_idx: int) -> float:
            cap = cv2.VideoCapture(str(out))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            cap.release()
            assert ok
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[80:160, 100:220]
            return float(cv2.Laplacian(roi, cv2.CV_64F).var())

        clean_var = roi_lap_var(0)
        blurred_var = roi_lap_var(37)
        after_var = roi_lap_var(80)
        # Blurred ROI loses most of its detail; clean frames keep it.
        assert blurred_var < clean_var * 0.2
        assert after_var > clean_var * 0.5
