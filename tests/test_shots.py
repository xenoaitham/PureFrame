from pureframe.pipeline.shots import detect_shots
import cv2
import numpy as np
import pytest


@pytest.fixture
def abrupt_cut_video(tmp_path):
    out_path = tmp_path / "cut.mp4"
    fps = 24
    w, h = 640, 480
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 2 seconds black
    for _ in range(48):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))

    # 2 seconds white
    for _ in range(48):
        out.write(np.full((h, w, 3), 255, dtype=np.uint8))

    out.release()
    return out_path


def test_detect_shots(abrupt_cut_video):
    shots = detect_shots(abrupt_cut_video, threshold=27.0)
    assert len(shots) == 2
    assert shots[0].start_frame == 0
    # scene cut should be around frame 48
    assert abs(shots[0].end_frame - 48) <= 2
    assert shots[1].start_frame == shots[0].end_frame
    assert shots[1].end_frame == 96
