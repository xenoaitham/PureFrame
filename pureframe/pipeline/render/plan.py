from pydantic import BaseModel
from pathlib import Path
from datetime import datetime

from pureframe.pipeline.probe import VideoMetadata
from pureframe.pipeline.shots import Shot, ShotVerdict, Action


class CensorPlan(BaseModel):
    pureframe_version: str
    plan_version: int = 1
    input_metadata: VideoMetadata
    config_snapshot: dict  # the config that produced this plan
    shots: list[Shot]
    verdicts: list[ShotVerdict]
    total_censored_frames: int
    total_blur_frames: int
    generated_at: datetime

    def serialize(self, path: Path):
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "CensorPlan":
        content = path.read_text()
        return cls.model_validate_json(content)

    def build_frame_actions(self) -> dict:
        frame_actions = {}
        # We index shots by their index to allow zip-independent lookup
        # In case the JSON was edited and elements are out of order
        shot_map = {s.index: s for s in self.shots}

        for verdict in self.verdicts:
            if verdict.action == Action.NONE:
                continue

            shot = shot_map.get(verdict.shot_index)
            if not shot:
                continue

            boxes_by_frame = {}
            if verdict.boxes:
                for b in verdict.boxes:
                    if b.frame_idx not in boxes_by_frame:
                        boxes_by_frame[b.frame_idx] = []
                    boxes_by_frame[b.frame_idx].append((b.x1, b.y1, b.x2, b.y2))

            for f in range(shot.start_frame, shot.end_frame):
                if f not in frame_actions:
                    frame_actions[f] = {"action": Action.NONE, "boxes": []}
                curr = frame_actions[f]
                if verdict.action == Action.FULL_FRAME_BLUR:
                    curr["action"] = Action.FULL_FRAME_BLUR
                elif (
                    verdict.action == Action.BLACK_BOX
                    and curr["action"] != Action.FULL_FRAME_BLUR
                ):
                    curr["action"] = Action.BLACK_BOX
                    if f in boxes_by_frame:
                        curr["boxes"].extend(boxes_by_frame[f])
        return frame_actions
