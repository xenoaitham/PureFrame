from pureframe.pipeline.smooth import smooth_detections
from pureframe.pipeline.shots import Shot
from pureframe.pipeline.detect.nudity import Detection

def test_smoothing_interpolation():
    shot = Shot(index=0, start_frame=100, end_frame=110, start_time=0.0, end_time=1.0)
    
    per_frame_detections = {
        100: [Detection(label="FEMALE_BREAST_EXPOSED", score=0.9, box=(100, 100, 200, 200))],
        # missing 101, 102
        103: [Detection(label="FEMALE_BREAST_EXPOSED", score=0.9, box=(130, 100, 230, 200))]
    }
    
    smoothed = smooth_detections(per_frame_detections, shot, padding_pct=0.0)
    
    assert 100 in smoothed
    assert 101 in smoothed
    assert 102 in smoothed
    assert 103 in smoothed
    
    # Check linear interpolation
    # Frame 100: 100
    # Frame 103: 130
    # Frame 101 should be 110
    b101 = smoothed[101][0]
    assert b101[0] == 110
