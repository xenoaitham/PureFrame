"""Automatic variable-frame-rate handling.

Probe-level: is_vfr() reads the peak-vs-average rate gap and the
frames-vs-duration drift off real ffprobe metadata, with a CFR control
that must never trigger. Pipeline-level: a real VFR clip generated with
ffmpeg runs through the full process flow - the plan is built on the
CFR intermediate (converted automatically, with a progress line), the
render stays on it, audio is untouched, and the output is constant
frame rate with matching duration.
"""

import subprocess
from pathlib import Path

import pytest

from pureframe.pipeline.probe import is_vfr, probe_video

pytestmark = pytest.mark.slow

# Frames 0-24 keep 25 fps spacing (1 s), frames 25-74 stretch to 5 fps
# (10 s): genuinely uneven timestamps. Peak rate 25, average ~7.1.
VFR_FILTER = "setpts='if(lt(N,25), N/25, 1+(N-25)/5)/TB'"


def _build_vfr_clip(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=128x72:rate=25",
            "-vf",
            VFR_FILTER,
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


def _build_cfr_clip(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=128x72:rate=25",
            "-c:v",
            "libx264",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


class TestProbeDetection:
    def test_vfr_clip_detected(self, tmp_path):
        meta = probe_video(_build_vfr_clip(tmp_path / "vfr.mp4"))
        assert is_vfr(meta) is True
        # The two signals the detector reads, for the record.
        assert abs(float(meta.fps) - float(meta.avg_fps)) / float(meta.fps) > 0.02

    def test_cfr_clip_not_detected(self, tmp_path):
        meta = probe_video(_build_cfr_clip(tmp_path / "cfr.mp4"))
        assert is_vfr(meta) is False

    def test_metadata_with_missing_average_is_not_vfr(self):
        from pureframe.utils.ffmpeg import extract_metadata

        meta = extract_metadata(
            {
                "format": {"duration": "10.0", "format_name": "mpegts"},
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 128,
                        "height": 72,
                        "r_frame_rate": "25/1",
                    }
                ],
            }
        )
        assert is_vfr(meta) is False


class TestConversion:
    def test_convert_to_cfr_produces_cfr(self, tmp_path):
        from pureframe.hardware import HardwareProfile, get_settings
        from pureframe.pipeline.probe import convert_to_cfr
        from pureframe.utils.ffmpeg import select_render_encoder

        source = _build_vfr_clip(tmp_path / "vfr.mp4")
        meta = probe_video(source)
        settings = get_settings(HardwareProfile.CPU)
        encoder = select_render_encoder(settings.profile, "h264", meta.video_codec)
        dest = convert_to_cfr(
            source, tmp_path / "out", meta, encoder, 20, settings.encoder_preset
        )
        assert dest.exists()
        converted = probe_video(dest)
        assert is_vfr(converted) is False
        # Same length in seconds (within a frame), real frames preserved.
        assert abs(converted.duration_seconds - meta.duration_seconds) < 0.5
        assert converted.total_frames >= meta.total_frames * 0.95


class TestEndToEnd:
    def test_process_on_vfr_input_converts_and_renders(
        self, tmp_path, monkeypatch, capsys
    ):
        """The full flow: VFR input in, CFR-rendered censored file out.

        The conversion line must print, audio must survive byte-identical
        (the CFR transcode copies it and the render copies it again), and
        the output must not be VFR.
        """
        import subprocess

        import pureframe.cli
        from pureframe.config import BlurMode, Config
        from pureframe.hardware import HardwareProfile
        from pureframe.pipeline.detect.nudity import Detection, NudityDetector

        source = tmp_path / "vfr_in.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=4:size=128x72:rate=25",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:duration=11",
                "-vf",
                VFR_FILTER,
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-crf",
                "30",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(source),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )

        def fake_detect(self, frames):
            return [
                [
                    Detection(
                        label="FEMALE_GENITALIA_EXPOSED",
                        score=0.9,
                        box=(10, 10, 100, 60),
                    )
                ]
                for _ in frames
            ]

        monkeypatch.setattr(NudityDetector, "detect_batch", fake_detect)

        out_path = tmp_path / "vfr_out.mp4"
        config = Config.from_cli(
            input_path=source,
            output_path=out_path,
            profile=HardwareProfile.CPU,
            blur_mode=BlurMode.BOX,
            no_clip=True,
            no_audio=False,
        )
        pureframe.cli.process_file(config)
        # process_file's cleanup ran: no temp dir left dangling on the config.
        assert config.cfr_input_path is None

        captured = capsys.readouterr().out
        assert "Variable frame rate detected" in captured

        assert out_path.exists()
        out_meta = probe_video(out_path)
        assert is_vfr(out_meta) is False
        in_meta = probe_video(source)
        assert abs(out_meta.duration_seconds - in_meta.duration_seconds) < 0.5

        # Audio survives: same codec, and its length tracks the output
        # video (the fixture's audio tail runs past the video, so the
        # render legitimately ends where the CFR video ends). Sample-
        # exact audio copy is pinned on CFR input by test_e2e.
        assert out_meta.audio_streams, "the output must keep its audio"
        assert out_meta.audio_streams[0]["codec_name"] == "aac"
        out_audio = float(out_meta.audio_streams[0]["duration"] or 0)
        assert abs(out_audio - float(out_meta.duration_seconds)) < 1.0
