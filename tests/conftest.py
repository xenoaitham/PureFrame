import pytest
from pathlib import Path
import subprocess
import os

@pytest.fixture(autouse=True)
def mock_store(monkeypatch, tmp_path):
    from pureframe.checkpoint import CheckpointStore
    import pureframe.cli
    db_path = tmp_path / "test_checkpoints.db"
    store = CheckpointStore(db_path)
    monkeypatch.setattr(pureframe.cli, "get_store", lambda: store)
    return store

@pytest.fixture(scope="session")
def fixtures_dir():
    d = Path(__file__).parent / "fixtures"
    d.mkdir(exist_ok=True, parents=True)
    return d

@pytest.fixture(scope="session")
def synthetic_video(fixtures_dir):
    out_path = fixtures_dir / "synthetic_explicit.mp4"
    if out_path.exists():
        return out_path
        
    # Generate a 15-second 24fps 720p video
    # Frames 100-250 will have an ellipse that represents explicit content.
    # A 1 kHz sine wave audio track.
    
    import cv2
    import numpy as np
    
    temp_vid = fixtures_dir / "temp_synth.mp4"
    w, h = 1280, 720
    fps = 24
    total_frames = 15 * fps
    
    out = cv2.VideoWriter(str(temp_vid), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i in range(total_frames):
        frame = np.full((h, w, 3), 128, dtype=np.uint8) # solid grey
        
        if 100 <= i <= 250:
            # Flesh-toned ellipse
            # Let's say flesh tone is BGR: 140, 180, 210
            # NudeNet expects realistic things, so a synthetic shape might not be detected!
            # Wait, NudeNet is very specific. A simple ellipse might NOT be detected by NudeNet.
            # But the prompt says: "Generate synthetic test clips programmatically - solid color frames with shapes that look like skin tones in specific bounding boxes - and verify the pipeline detects and censors them correctly."
            # Actually, NudeNet will likely NOT detect a random ellipse. 
            # So for test_e2e, we'll need to mock NudityDetector OR force it. 
            # I will draw something skin-colored but if it fails to detect, I might need to mock NudityDetector in the test.
            # Let's draw an ellipse.
            cv2.ellipse(frame, (300, 250), (100, 100), 0, 0, 360, (140, 180, 210), -1)
            
        # Draw frame number for debug
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    
    # Add audio and re-encode to proper format
    subprocess.run([
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=15',
        '-i', str(temp_vid),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        str(out_path)
    ], check=True, stderr=subprocess.DEVNULL)
    
    os.remove(temp_vid)
    return out_path
