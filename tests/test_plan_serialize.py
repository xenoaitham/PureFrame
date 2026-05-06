import pytest
import json
from pathlib import Path
from datetime import datetime, timezone

from pureframe.pipeline.render.plan import CensorPlan
from pureframe.pipeline.probe import VideoMetadata
from pureframe.pipeline.shots import Shot, ShotVerdict, Action, Category, Box, FrameResult
from pureframe.pipeline.detect.nudity import Detection

def test_plan_serialization(tmp_path: Path):
    meta = VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        duration_seconds=10.0,
        video_codec="h264",
        color_space="bt709",
        is_hdr=False,
        total_frames=300,
        has_audio=True,
        subtitle_streams=[],
        container="mp4",
        pixel_format="yuv420p",
        audio_streams=[{"index": 1, "codec_name": "aac"}]
    )
    
    config_snapshot = {
        "input_path": "/fake/in.mp4",
        "output_path": "/fake/out.mp4",
        "nudity_threshold": 0.55
    }
    
    shot1 = Shot(
        index=0,
        start_frame=0,
        end_frame=30,
        start_time=0.0,
        end_time=1.0,
        frames={
            0: FrameResult(frame_idx=0, detections=[Detection(label="EXPOSED_BREAST_F", score=0.9, box=(10, 10, 50, 50))])
        }
    )
    
    verdict1 = ShotVerdict(
        shot_index=0,
        action=Action.BLACK_BOX,
        category=Category.NUDITY_EXPLICIT,
        confidence=0.9,
        reasoning="Found explicit content",
        boxes=[Box(x1=10, y1=10, x2=50, y2=50, frame_idx=0)]
    )
    
    plan = CensorPlan(
        pureframe_version="0.2.0",
        plan_version=1,
        input_metadata=meta,
        config_snapshot=config_snapshot,
        shots=[shot1],
        verdicts=[verdict1],
        total_censored_frames=30,
        total_blur_frames=0,
        generated_at=datetime.now(timezone.utc)
    )
    
    plan_file = tmp_path / "plan.json"
    plan.serialize(plan_file)
    
    # Reload
    loaded = CensorPlan.load(plan_file)
    
    assert loaded.pureframe_version == "0.2.0"
    assert loaded.plan_version == 1
    assert loaded.input_metadata.fps == 30.0
    assert loaded.total_censored_frames == 30
    assert loaded.config_snapshot["nudity_threshold"] == 0.55
    
    assert len(loaded.shots) == 1
    assert loaded.shots[0].end_frame == 30
    assert 0 in loaded.shots[0].frames
    assert loaded.shots[0].frames[0].detections[0].label == "EXPOSED_BREAST_F"
    
    assert len(loaded.verdicts) == 1
    assert loaded.verdicts[0].action == Action.BLACK_BOX
    assert loaded.verdicts[0].category == Category.NUDITY_EXPLICIT
    assert len(loaded.verdicts[0].boxes) == 1
    assert loaded.verdicts[0].boxes[0].x2 == 50

def test_build_frame_actions(tmp_path: Path):
    meta = VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        duration_seconds=10.0,
        video_codec="h264",
        color_space="bt709",
        is_hdr=False,
        total_frames=300,
        has_audio=True,
        subtitle_streams=[],
        container="mp4",
        pixel_format="yuv420p",
        audio_streams=[]
    )
    
    shot1 = Shot(index=0, start_frame=0, end_frame=5, start_time=0.0, end_time=1.0)
    verdict1 = ShotVerdict(
        shot_index=0,
        action=Action.BLACK_BOX,
        category=Category.NUDITY_EXPLICIT,
        confidence=0.9,
        reasoning="",
        boxes=[
            Box(x1=0, y1=0, x2=10, y2=10, frame_idx=0),
            Box(x1=0, y1=0, x2=10, y2=10, frame_idx=2),
        ]
    )
    
    shot2 = Shot(index=1, start_frame=5, end_frame=10, start_time=1.0, end_time=2.0)
    verdict2 = ShotVerdict(
        shot_index=1,
        action=Action.FULL_FRAME_BLUR,
        category=Category.NUDITY_EXPLICIT,
        confidence=0.9,
        reasoning="",
        boxes=[]
    )
    
    plan = CensorPlan(
        pureframe_version="0.2.0",
        plan_version=1,
        input_metadata=meta,
        config_snapshot={},
        shots=[shot1, shot2],
        verdicts=[verdict1, verdict2],
        total_censored_frames=10,
        total_blur_frames=5,
        generated_at=datetime.now(timezone.utc)
    )
    
    actions = plan.build_frame_actions()
    
    # frames 0-4 have BLACK_BOX, but boxes are mapped explicitly per frame if they exist in verdict.boxes
    assert actions[0]["action"] == Action.BLACK_BOX
    assert len(actions[0]["boxes"]) == 1
    
    assert actions[1]["action"] == Action.BLACK_BOX
    assert len(actions[1]["boxes"]) == 0
    
    assert actions[2]["action"] == Action.BLACK_BOX
    assert len(actions[2]["boxes"]) == 1
    
    # frames 5-9 have FULL_FRAME_BLUR
    assert actions[5]["action"] == Action.FULL_FRAME_BLUR
    assert actions[9]["action"] == Action.FULL_FRAME_BLUR
