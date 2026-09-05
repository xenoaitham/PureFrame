from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from scenedetect import ContentDetector, SceneManager, open_video

from pureframe.pipeline.detect.nudity import Detection


class FrameResult(BaseModel):
    frame_idx: int
    detections: list[Detection]


class Shot(BaseModel):
    index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frames: dict[int, FrameResult] = {}


class Category(str, Enum):
    NUDITY_EXPLICIT = "NUDITY_EXPLICIT"
    SEXUAL_ACT_VISIBLE = "SEXUAL_ACT_VISIBLE"
    KISS_INTENSE = "KISS_INTENSE"
    KISS_LIGHT = "KISS_LIGHT"
    SEXUAL_CONTEXT_NO_NUDITY = "SEXUAL_CONTEXT_NO_NUDITY"
    VIOLENCE_GORE = "VIOLENCE_GORE"
    SAFE = "SAFE"


class Action(str, Enum):
    BLACK_BOX = "BLACK_BOX"
    FULL_FRAME_BLUR = "FULL_FRAME_BLUR"
    NONE = "NONE"


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    frame_idx: int | None = None


class ShotVerdict(BaseModel):
    shot_index: int
    category: Category
    action: Action
    confidence: float
    boxes: list[Box] | None = None
    reasoning: str


def detect_shots(path: Path, threshold: float = 27.0) -> list[Shot]:
    video = open_video(str(path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    shots = []
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()

        shots.append(
            Shot(
                index=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
            )
        )

    merged_shots = []
    i = 0
    while i < len(shots):
        current = shots[i]
        while current.end_frame - current.start_frame < 4 and i + 1 < len(shots):
            next_shot = shots[i + 1]
            current.end_frame = next_shot.end_frame
            current.end_time = next_shot.end_time
            i += 1
        merged_shots.append(current)
        i += 1

    for j, s in enumerate(merged_shots):
        s.index = j

    if not merged_shots:
        video = open_video(str(path))
        duration = video.duration.get_seconds()
        total_frames = video.duration.get_frames()
        merged_shots.append(
            Shot(
                index=0,
                start_frame=0,
                end_frame=total_frames,
                start_time=0.0,
                end_time=duration,
            )
        )

    return merged_shots
