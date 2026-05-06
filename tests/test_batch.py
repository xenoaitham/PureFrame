import pytest
from pathlib import Path
import shutil
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.batch import process_folder
from pureframe.checkpoint import CheckpointStore, Job
from pureframe.pipeline.shots import ShotVerdict, Category, Action

def global_mocked_process_file(cfg):
    shutil.copy(cfg.input_path, cfg.output_path)

def create_dummy_video(path: Path):
    # Create a tiny 1-second video
    import cv2
    import numpy as np
    import subprocess
    import os
    
    temp_vid = path.with_suffix('.temp.mp4')
    out = cv2.VideoWriter(str(temp_vid), cv2.VideoWriter_fourcc(*'mp4v'), 24, (320, 240))
    for i in range(24):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        out.write(frame)
    out.release()
    
    subprocess.run([
        'ffmpeg', '-y', '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=1',
        '-i', str(temp_vid), '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(path)
    ], check=True, stderr=subprocess.DEVNULL)
    os.remove(temp_vid)

def test_batch_folder_execution(tmp_path, monkeypatch, mock_store):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    
    # Create nested structure
    (in_dir / "sub").mkdir()
    create_dummy_video(in_dir / "1.mp4")
    create_dummy_video(in_dir / "sub" / "2.mp4")
    
    config = Config(
        input_path=in_dir,
        output_path=out_dir,
        profile=HardwareProfile.CPU
    )
    
    # Mock ProcessPoolExecutor to run synchronously
    class DummyExecutor:
        def __init__(self, max_workers=None, mp_context=None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def submit(self, fn, *args, **kwargs):
            class DummyFuture:
                def __init__(self, res):
                    self._res = res
                def result(self):
                    return self._res
            try:
                res = fn(*args, **kwargs)
                return DummyFuture(res)
            except Exception as e:
                err_msg = str(e)  # capture before 'e' goes out of scope (Python 3)

                class FailedFuture:
                    def result(self):
                        return "err", "FAILED", err_msg
                return FailedFuture()

    monkeypatch.setattr("pureframe.batch.ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr("pureframe.batch.as_completed", lambda futures: futures)
    
    # Mock process_file so we don't actually run models
    monkeypatch.setattr("pureframe.cli.process_file", global_mocked_process_file)
    
    process_folder(in_dir, True, 1, config)
    
    
    assert (in_dir / "1.pureframe.mp4").exists()
    assert (in_dir / "sub" / "2.pureframe.mp4").exists()

def test_checkpoint_crash_and_resume(tmp_path, mock_store):
    in_file = tmp_path / "video.mp4"
    out_file = tmp_path / "out.mp4"
    create_dummy_video(in_file)
    
    config = Config(
        input_path=in_file,
        output_path=out_file,
        profile=HardwareProfile.CPU
    )
    
    # Simulate a crash after 1 shot
    job = mock_store.find_or_create_job(in_file, out_file, config)
    assert job.status == "PENDING"
    
    verdict = ShotVerdict(shot_index=0, category=Category.SAFE, action=Action.NONE, confidence=0.9, reasoning="Safe")
    mock_store.save_verdict(job.id, verdict)
    
    mock_store.update_status(job.id, "FAILED", error="Crash simulated")
    
    # Now simulate resume
    # The config hash is the same
    job_resumed = mock_store.find_or_create_job(in_file, out_file, config)
    assert job_resumed.id == job.id
    
    # Verify we can load the verdict
    verdicts = mock_store.load_verdicts(job_resumed.id)
    assert len(verdicts) == 1
    assert verdicts[0].shot_index == 0

def test_config_hash_change_new_job(tmp_path, mock_store):
    in_file = tmp_path / "video.mp4"
    out_file = tmp_path / "out.mp4"
    
    cfg1 = Config(input_path=in_file, output_path=out_file, profile=HardwareProfile.CPU, strict=False)
    job1 = mock_store.find_or_create_job(in_file, out_file, cfg1)
    
    cfg2 = Config(input_path=in_file, output_path=out_file, profile=HardwareProfile.CPU, strict=True)
    job2 = mock_store.find_or_create_job(in_file, out_file, cfg2)
    
    assert job1.id != job2.id
    assert job1.config_hash != job2.config_hash
