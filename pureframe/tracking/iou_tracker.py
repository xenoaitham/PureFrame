from pydantic import BaseModel
from pureframe.pipeline.detect.nudity import Detection

class Track(BaseModel):
    track_id: int
    label: str
    box: tuple[int, int, int, int]
    age: int
    score: float

def compute_iou(box1: tuple[int,int,int,int], box2: tuple[int,int,int,int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.next_id = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        for t in self.tracks:
            t.age += 1
            
        unmatched_dets = list(detections)
        unmatched_tracks = list(self.tracks)
        
        matched = []
        
        # Greedy matching
        for d in list(unmatched_dets):
            best_iou = 0.0
            best_track = None
            for t in unmatched_tracks:
                if t.label == d.label:
                    iou = compute_iou(d.box, t.box)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_track = t
            if best_track is not None:
                best_track.box = d.box
                best_track.score = d.score
                best_track.age = 0
                matched.append(best_track)
                unmatched_dets.remove(d)
                unmatched_tracks.remove(best_track)
                
        # Drop old
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        
        # Add new
        for d in unmatched_dets:
            new_track = Track(
                track_id=self.next_id,
                label=d.label,
                box=d.box,
                age=0,
                score=d.score
            )
            self.tracks.append(new_track)
            self.next_id += 1
            
        return [t for t in self.tracks if t.age == 0]
