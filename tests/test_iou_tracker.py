from pureframe.tracking.iou_tracker import IoUTracker, compute_iou
from pureframe.pipeline.detect.nudity import Detection

def test_iou_compute():
    b1 = (0, 0, 10, 10)
    b2 = (5, 5, 15, 15)
    # Area1 = 100
    # Area2 = 100
    # Inter = 5*5 = 25
    # Union = 200 - 25 = 175
    iou = compute_iou(b1, b2)
    assert abs(iou - 25/175) < 1e-5
    
def test_iou_tracker_match():
    tracker = IoUTracker(iou_threshold=0.3, max_age=5)
    
    # Frame 1
    dets = [Detection(label="FEMALE_BREAST_EXPOSED", score=0.9, box=(0, 0, 100, 100))]
    tracks = tracker.update(dets)
    assert len(tracks) == 1
    tid = tracks[0].track_id
    
    # Frame 2 - moved slightly
    dets = [Detection(label="FEMALE_BREAST_EXPOSED", score=0.95, box=(10, 10, 110, 110))]
    tracks = tracker.update(dets)
    assert len(tracks) == 1
    assert tracks[0].track_id == tid # matched
    
    # Frame 3 - missed
    tracks = tracker.update([])
    assert len(tracks) == 0 # no active
    assert len(tracker.tracks) == 1 # still in memory, aging
    assert tracker.tracks[0].age == 1
