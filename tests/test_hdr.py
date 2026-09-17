"""HDR10 metadata preservation on re-encode.

A synthetic HDR10 fixture (x265 with injected master-display and color
tags) runs through ``write_video_with_overlay``; the output must carry
the same mastering display, content light level and color tags, so
players keep the right transfer function instead of shifting colors.
The raw frame pipe stays 8-bit, so this pins the metadata and tags -
the full 10-bit signal is out of scope and documented as such.
"""

import subprocess
from pathlib import Path

import pytest

from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.probe import probe_video
from pureframe.utils.ffmpeg import write_video_with_overlay

pytestmark = pytest.mark.slow

# BT.2020 P3D65 limited range - the usual HDR10 mastering values.
MASTER_DISPLAY = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
MAX_CLL = "1000,400"


def _build_hdr_clip(path: Path) -> Path:
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
            "testsrc2=duration=1:size=160x120:rate=15",
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p",
            "-x265-params",
            f"master-display={MASTER_DISPLAY}:max-cll={MAX_CLL}",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


@pytest.fixture(scope="session")
def hdr_clip(tmp_path_factory):
    return _build_hdr_clip(tmp_path_factory.mktemp("hdr") / "hdr.mkv")


def _frame0_side_data(path: Path) -> list[dict]:
    import json

    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-read_intervals",
            "%+#1",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    frames = json.loads(res.stdout).get("frames", [])
    return frames[0].get("side_data_list", []) if frames else []


def _types(side_data: list[dict]) -> set[str]:
    return {entry.get("side_data_type", "") for entry in side_data}


class TestHdrPreservation:
    def test_fixture_carries_hdr10(self, hdr_clip):
        """Sanity: the fixture itself exposes the metadata the pipeline reads."""
        meta = probe_video(hdr_clip)
        assert meta.is_hdr is True
        assert meta.color_transfer == "smpte2084"
        assert meta.master_display == MASTER_DISPLAY
        assert meta.max_cll == MAX_CLL

    def test_reencode_preserves_metadata_and_tags(self, hdr_clip, tmp_path):
        out_path = tmp_path / "hdr_out.mkv"
        settings = get_settings(HardwareProfile.CPU)
        write_video_with_overlay(
            hdr_clip,
            out_path,
            lambda idx, frame: frame,
            settings,
            encoder="libx265",
            crf=20,
        )
        assert out_path.exists()

        out_meta = probe_video(out_path)
        assert out_meta.is_hdr is True
        assert out_meta.color_transfer == "smpte2084"
        assert out_meta.color_primaries == "bt2020"
        assert out_meta.color_space == "bt2020nc"
        # The mastering display survives the re-encode (frame-level SEI).
        assert out_meta.master_display == MASTER_DISPLAY
        assert out_meta.max_cll == MAX_CLL

        types = _types(_frame0_side_data(out_path))
        assert any("Mastering display" in t for t in types)
        assert any("Content light" in t for t in types)

    def test_sdr_input_stays_clean(self, tmp_path):
        """No HDR metadata in, no x265-params out - nothing to preserve."""
        import ffmpeg as ffmpeg_python

        from pureframe.pipeline.probe import probe_video as pv

        sdr = tmp_path / "sdr.mkv"
        (
            ffmpeg_python.input(
                "testsrc2=duration=0.5:size=160x120:rate=15",
                f="lavfi",
            )
            .output(str(sdr), **{"c:v": "libx265", "pix_fmt": "yuv420p"})
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
        meta = pv(sdr)
        assert meta.is_hdr is False
        assert meta.master_display == ""
        assert meta.max_cll == ""
