import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from pureframe.hardware import HardwareProfile

pytestmark = pytest.mark.slow


def get_audio_hash(path: Path) -> str:
    # Run ffmpeg to extract audio and hash it
    try:
        res = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Expected output format: "SHA256=..."
        for line in res.stdout.splitlines():
            if line.startswith("SHA256="):
                return line.strip()
    except subprocess.CalledProcessError:
        return ""
    return ""


def get_duration(path: Path) -> float:
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(res.stdout.strip())


def _run_pipeline(tmp_path, monkeypatch, synthetic_video, blur_mode, out_name):
    """Run process_file on the synthetic fixture with a mocked NudityDetector.

    Real NudeNet won't trigger on a grey video with a pink ellipse, so
    detect_batch is patched to return an explicit-label detection whenever the
    sampled frame contains the ellipse.
    """
    out_path = tmp_path / out_name

    import pureframe.cli
    from pureframe.pipeline.detect.nudity import Detection, NudityDetector

    # Patch settings so we sample enough frames to definitely hit 100-250
    original_get_settings = pureframe.cli.get_settings

    def mocked_get_settings(profile):
        s = original_get_settings(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", mocked_get_settings)

    def mocked_detect_batch(self, frames_bgr):
        batch_res = []
        for frame in frames_bgr:
            h, w = frame.shape[:2]
            px_x = int(300 * w / 1280)
            px_y = int(250 * h / 720)

            pixel = frame[px_y, px_x]
            # If it's not grey, it has the ellipse
            if not np.allclose(pixel, [128, 128, 128], atol=20):
                bx1 = int(200 * w / 1280)
                by1 = int(150 * h / 720)
                bx2 = int(400 * w / 1280)
                by2 = int(350 * h / 720)
                batch_res.append(
                    [
                        Detection(
                            label="FEMALE_BREAST_EXPOSED",
                            score=0.99,
                            box=(bx1, by1, bx2, by2),
                        )
                    ]
                )
            else:
                batch_res.append([])
        return batch_res

    monkeypatch.setattr(NudityDetector, "detect_batch", mocked_detect_batch)

    from pureframe.cli import process_file
    from pureframe.config import Config

    config = Config.from_cli(
        input_path=synthetic_video,
        output_path=out_path,
        profile=HardwareProfile.CPU,
        nudity_threshold=0.5,
        strict=False,
        no_clip=False,
        no_audio=False,
        blur_mode=blur_mode,
        log_level="DEBUG",
    )
    process_file(config)
    return out_path


def _assert_container_intact(synthetic_video, out_path):
    assert out_path.exists()

    out_dur = get_duration(out_path)
    in_dur = get_duration(synthetic_video)
    assert abs(out_dur - in_dur) < 0.1

    h1 = get_audio_hash(synthetic_video)
    h2 = get_audio_hash(out_path)
    assert h1 == h2
    assert h1 != ""


# Ellipse drawn at center (300, 250), axes (100, 100). Detector returns the
# box (200, 150)-(400, 350); padding widens it further. This ROI certainly
# contains the sharp grey→flesh ellipse boundary in the input.
CENSOR_ROI = np.s_[120:380, 170:430]


def _roi_lap_var(path: Path, frame_idx: int) -> float:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"could not read frame {frame_idx} from {path}"
    roi = frame[CENSOR_ROI]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def test_pipeline_e2e(synthetic_video, tmp_path, monkeypatch):
    """BOX mode: deterministic solid-colour censoring lands in the render."""
    from pureframe.config import BlurMode

    out_path = _run_pipeline(
        tmp_path, monkeypatch, synthetic_video, BlurMode.BOX, "output.mp4"
    )
    _assert_container_intact(synthetic_video, out_path)

    # Check that frames 110-240 contain a black rectangle where the ellipse was
    cap = cv2.VideoCapture(str(out_path))
    frame_idx = 0
    covered_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 110 <= frame_idx <= 240:
            # Look at center of the box, should be black (0,0,0) or close due to compression
            pixel = frame[250, 300]
            if np.all(pixel < [10, 10, 10]):
                covered_frames += 1
        frame_idx += 1
    cap.release()

    # It might not cover exactly 110-240 if sampling density caused edges to miss,
    # but the bulk should be covered.
    assert covered_frames > 100


def test_pipeline_e2e_blur_default(synthetic_video, tmp_path, monkeypatch):
    """Default BLUR mode: the flagged region loses its sharp edges while the
    rest of the video passes through untouched."""
    from pureframe.config import BlurMode

    out_path = _run_pipeline(
        tmp_path, monkeypatch, synthetic_video, BlurMode.BLUR, "output_blur.mp4"
    )
    _assert_container_intact(synthetic_video, out_path)

    # Censored mid-shot frame: edge energy inside the flagged ROI must collapse
    in_var = _roi_lap_var(synthetic_video, 150)
    out_var = _roi_lap_var(out_path, 150)
    assert in_var > 0
    assert out_var < in_var * 0.25, (
        f"expected blur to flatten the flagged ROI (in={in_var:.1f}, out={out_var:.1f})"
    )

    # Uncensored frame (before the ellipse appears): stream-copied/re-encoded
    # but structurally unchanged
    cap_in = cv2.VideoCapture(str(synthetic_video))
    cap_in.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret_in, f_in = cap_in.read()
    cap_in.release()

    cap_out = cv2.VideoCapture(str(out_path))
    cap_out.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret_out, f_out = cap_out.read()
    cap_out.release()
    assert ret_in and ret_out

    diff = float(np.mean(np.abs(f_in.astype(int) - f_out.astype(int))))
    assert diff < 5.0, f"clean frame altered too much (mean abs diff {diff:.2f})"
