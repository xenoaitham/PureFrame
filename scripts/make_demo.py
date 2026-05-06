#!/usr/bin/env python3
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

SYNTHETIC_MP4 = ASSETS_DIR / "demo_input.mp4"
PLAN_JSON = ASSETS_DIR / "demo_plan.json"
OUTPUT_MP4 = ASSETS_DIR / "demo_output.mp4"
FINAL_GIF = ASSETS_DIR / "demo.gif"

def create_synthetic_video():
    print("Generating synthetic 5-second video...")
    # 5 seconds, 30fps
    # Scene 1: Green background (Safe) - 0 to 1.5s
    # Scene 2: Red box moving (Explicit shape) - 1.5s to 3.5s
    # Scene 3: Blue background (Soft implied) - 3.5s to 5.0s
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=5:size=854x480:rate=30",
        "-vf", "drawbox=x='iw/2-50+t*20':y='ih/2-50':w=100:h=100:color=red@0.8:t=fill",
        "-c:v", "libx264", "-preset", "ultrafast",
        str(SYNTHETIC_MP4)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from pureframe.pipeline.probe import probe_video

def create_plan():
    print("Generating synthetic censor plan...")
    metadata = probe_video(SYNTHETIC_MP4)
    plan = {
        "pureframe_version": "0.1.0",
        "plan_version": 1,
        "input_metadata": metadata.model_dump(),
        "config_snapshot": {"no_audio": True, "profile": "CPU"},
        "shots": [
            {"index": 0, "start_frame": 0, "end_frame": 45, "start_time": 0.0, "end_time": 1.5},
            {"index": 1, "start_frame": 45, "end_frame": 105, "start_time": 1.5, "end_time": 3.5},
            {"index": 2, "start_frame": 105, "end_frame": 150, "start_time": 3.5, "end_time": 5.0}
        ],
        "verdicts": [
            {
                "shot_index": 1,
                "action": "BLACK_BOX",
                "category": "NUDITY_EXPLICIT",
                "confidence": 0.99,
                "reasoning": "Detected red explicit shape",
                "boxes": []
            }
        ],
        "total_censored_frames": 60,
        "total_blur_frames": 0,
        "generated_at": datetime.now().isoformat()
    }

    # Add frame-by-frame bounding boxes tracking the moving red box
    for f in range(45, 105):
        t = f / 30.0
        x1 = int(854/2 - 50 + t*20)
        y1 = int(480/2 - 50)
        plan["verdicts"][0]["boxes"].append({
            "x1": x1,
            "y1": y1,
            "x2": x1 + 100,
            "y2": y1 + 100,
            "frame_idx": f
        })

    with open(PLAN_JSON, "w") as f:
        json.dump(plan, f, indent=2)

def run_apply():
    print("Applying censor plan via PureFrame...")
    cmd = [
        "pureframe", "apply",
        str(SYNTHETIC_MP4),
        str(PLAN_JSON),
        "--output", str(OUTPUT_MP4),
    ]
    subprocess.run(cmd, check=True)

def create_side_by_side_gif():
    print("Encoding side-by-side GIF...")
    # hstack the two videos and output to gif
    cmd = [
        "ffmpeg", "-y",
        "-i", str(SYNTHETIC_MP4),
        "-i", str(OUTPUT_MP4),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v];"
        "[v]split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(FINAL_GIF)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Done! GIF saved to {FINAL_GIF}")

if __name__ == "__main__":
    create_synthetic_video()
    create_plan()
    run_apply()
    create_side_by_side_gif()
