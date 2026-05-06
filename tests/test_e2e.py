import pytest
from pathlib import Path
import subprocess
import hashlib
from pureframe.cli import process_cmd
import cv2
import numpy as np
from pureframe.hardware import HardwareProfile

def get_audio_hash(path: Path) -> str:
    # Run ffmpeg to extract audio and hash it
    try:
        res = subprocess.run([
            'ffmpeg', '-i', str(path), '-map', '0:a:0', '-f', 'hash', '-hash', 'sha256', '-'
        ], capture_output=True, text=True, check=True)
        # Expected output format: "SHA256=..."
        for line in res.stdout.splitlines():
            if line.startswith("SHA256="):
                return line.strip()
    except subprocess.CalledProcessError:
        return ""
    return ""

@pytest.mark.slow
def test_pipeline_e2e(synthetic_video, tmp_path, monkeypatch):
    out_path = tmp_path / "output.mp4"
    
    # We must mock NudityDetector to actually detect our synthetic ellipse 
    # since real NudeNet won't trigger on a grey video with a pink ellipse.
    from pureframe.pipeline.detect.nudity import NudityDetector, Detection
    import pureframe.cli
    from pureframe.hardware import ProfileSettings, HardwareProfile
    
    # Patch settings so we sample enough frames to definitely hit 100-250
    original_get_settings = pureframe.cli.get_settings
    def mocked_get_settings(profile):
        s = original_get_settings(profile)
        s.sample_keyframes_per_shot = 10
        return s
    monkeypatch.setattr(pureframe.cli, "get_settings", mocked_get_settings)
    
    original_detect_batch = NudityDetector.detect_batch
    
    def mocked_detect_batch(self, frames_bgr):
        batch_res = []
        for frame in frames_bgr:
            # Reconstruct dimensions to check if it has the shape.
            h, w = frame.shape[:2]
            # Since the frame might be downscaled, we just check the center pixel relative to its size
            # Original was 1280x720, ellipse at 300, 250
            px_x = int(300 * w / 1280)
            px_y = int(250 * h / 720)
            
            pixel = frame[px_y, px_x]
            # If it's not grey, it has the ellipse
            if not np.allclose(pixel, [128, 128, 128], atol=20):
                # We want a box that covers roughly (200, 150) to (400, 350) in original 1280x720 space
                bx1 = int(200 * w / 1280)
                by1 = int(150 * h / 720)
                bx2 = int(400 * w / 1280)
                by2 = int(350 * h / 720)
                batch_res.append([Detection(label="FEMALE_BREAST_EXPOSED", score=0.99, box=(bx1, by1, bx2, by2))])
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
        log_level="DEBUG"
    )
    process_file(config)
    
    assert out_path.exists()
    
    # Check duration via ffprobe
    res = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', str(out_path)
    ], capture_output=True, text=True, check=True)
    out_dur = float(res.stdout.strip())
    
    res = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', str(synthetic_video)
    ], capture_output=True, text=True, check=True)
    in_dur = float(res.stdout.strip())
    
    assert abs(out_dur - in_dur) < 0.1
    
    # Check audio hash
    h1 = get_audio_hash(synthetic_video)
    h2 = get_audio_hash(out_path)
    assert h1 == h2
    assert h1 != ""
    
    # Check that frames 100-250 contain a black rectangle where the ellipse was
    # The original ellipse is at 200,150 to 400,350
    # Padding is 12% by default
    cap = cv2.VideoCapture(str(out_path))
    frame_idx = 0
    black_covered_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 110 <= frame_idx <= 240:
            # Look at center of the box, should be black (0,0,0) or close due to compression
            pixel = frame[250, 300]
            if np.all(pixel < [10, 10, 10]):
                black_covered_frames += 1
        frame_idx += 1
    cap.release()
    
    # It might not cover exactly 110-240 if sampling density caused edges to miss, 
    # but the bulk should be covered.
    assert black_covered_frames > 100
