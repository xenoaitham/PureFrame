#!/usr/bin/env python3
"""Regenerate the README demo assets (assets/demo.gif).

The demo must be shippable without licensing concerns, so it is fully
synthetic - but it should look like a film frame, not a test pattern.
This script renders a cinematic dusk scene with numpy (animated gradient,
drifting bokeh, film grain, vignette) and a soft skin-tone figure crossing
the frame, fabricates a censor plan whose per-frame boxes track the
figure's torso, renders the censored output with ``pureframe apply``
(default Gaussian-blur mode), and encodes a labelled side-by-side GIF.

Plan boxes are authored in the CPU profile's detection space (480 px) -
the same space real NudeNet detections occupy - so the renderer's
detection→native rescale lands the blur exactly on the figure. (The
previous demo authored boxes in native coordinates and the blur drifted
into the bottom-right corner as a result.)

Usage:
    uv run python scripts/make_demo.py
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pureframe.pipeline.probe import probe_video

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"

SYNTHETIC_MP4 = ASSETS_DIR / "demo_input.mp4"
PLAN_JSON = ASSETS_DIR / "demo_plan.json"
OUTPUT_MP4 = ASSETS_DIR / "demo_output.mp4"
FINAL_GIF = ASSETS_DIR / "demo.gif"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

WIDTH, HEIGHT, FPS, DURATION_S = 854, 480, 30, 5.0
N_FRAMES = int(FPS * DURATION_S)  # 150

# The figure crosses the frame during this window; the flagged middle shot
# (1.5 s – 3.5 s) covers it almost exactly, so the blur appears as the
# figure enters and clears as it leaves - shot-level flagging, honestly.
ENTER_S, EXIT_S = 1.35, 3.65

# CPU profile detection resolution; the plan snapshot pins this profile.
DETECTION_RES = 480

_TORSO_RX, _TORSO_RY = 52.0, 74.0
_HEAD_R = 27.0
_HEAD_LIFT = 104.0
_BOX_PAD = 12  # native-px padding around the torso before detection scaling


def figure_pose(t: float) -> tuple[float, float] | None:
    """Torso centre (x, y) at time *t*, or ``None`` while off-frame."""
    if t < ENTER_S or t > EXIT_S:
        return None
    u = (t - ENTER_S) / (EXIT_S - ENTER_S)
    x = -60.0 + u * (WIDTH + 120.0)
    y = 252.0 + 20.0 * np.sin(2.0 * np.pi * u * 1.5 + 0.4)
    return x, float(y)


def _smoothstep(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return v * v * (3.0 - 2.0 * v)


class SceneRenderer:
    """Deterministic per-frame renderer for the synthetic dusk scene."""

    def __init__(self) -> None:
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)
        gx = (self.xx - WIDTH / 2) / (WIDTH / 2)
        gy = (self.yy - HEIGHT / 2) / (HEIGHT / 2)
        self._vignette = (
            1.0 - 0.38 * np.clip(gx * gx + gy * gy, 0.0, 1.0) ** 1.2
        ).astype(np.float32)
        # (speed px/s, radius, edge softness, colour, alpha, y, x0, phase)
        self._bokeh = [
            (14.0, 78.0, 10.0, (255, 196, 130), 0.16, 96.0, -40.0, 0.0),
            (9.0, 46.0, 7.0, (255, 224, 180), 0.13, 388.0, 210.0, 1.1),
            (20.0, 30.0, 6.0, (150, 214, 224), 0.14, 150.0, 480.0, 2.2),
            (11.0, 58.0, 9.0, (255, 170, 110), 0.18, 300.0, 640.0, 0.6),
            (16.0, 24.0, 5.0, (210, 235, 255), 0.10, 230.0, 90.0, 2.8),
            (7.0, 88.0, 12.0, (255, 210, 150), 0.10, 420.0, 760.0, 1.7),
        ]
        # The backdrop is deliberately static: an animated gradient/breathe
        # changes every pixel every frame, which defeats the GIF encoder's
        # diff_mode (a fully animated build measured 6.6 MiB). Bokeh and
        # the figure carry all the motion; the static regions then
        # compress to almost nothing across frames.
        self._backdrop = self._render_backdrop()

    def _render_backdrop(self) -> np.ndarray:
        frame = self._gradient()

        # Moonlit window, top right: soft bleed first, then the hard pane
        # with dark cross bars. The hard edges give the blur rectangle
        # something to smear - without structure the whole soft scene
        # hides where the censor blur actually is.
        wx1, wy1, wx2, wy2 = 560.0, 40.0, 810.0, 300.0
        d2 = (self.xx - (wx1 + wx2) / 2) ** 2 + (self.yy - (wy1 + wy2) / 2) ** 2
        frame += (0.28 * np.exp(-d2 / (2.0 * 95.0**2)))[..., None] * np.array(
            (110, 140, 200), np.float32
        )

        pane_a = _smoothstep(
            np.minimum(
                np.minimum(self.xx - wx1, wx2 - self.xx),
                np.minimum(self.yy - wy1, wy2 - self.yy),
            )
            / 3.0
        )
        pane_v = np.clip((self.yy - wy1) / (wy2 - wy1), 0.0, 1.0)[..., None]
        pane = (
            np.array((150, 175, 220), np.float32) * (1.0 - pane_v)
            + np.array((85, 105, 155), np.float32) * pane_v
        )
        bars = ((np.abs(self.xx - 677.0) < 9.0) | (np.abs(self.yy - 161.0) < 9.0))[
            ..., None
        ]
        pane = np.where(bars, np.array((20, 24, 36), np.float32), pane)
        frame = (
            frame * (1.0 - (0.92 * pane_a)[..., None])
            + pane * (0.92 * pane_a)[..., None]
        )

        # Warm lamp glow, bottom left; dark floor below a hard horizon.
        d2 = (self.xx - 140.0) ** 2 + (self.yy - 420.0) ** 2
        frame += (0.45 * np.exp(-d2 / (2.0 * 130.0**2)))[..., None] * np.array(
            (255, 178, 102), np.float32
        )
        floor_a = _smoothstep((self.yy - 384.0) / 10.0)[..., None]
        frame = (
            frame * (1.0 - floor_a)
            + np.array((18, 18, 26), np.float32)[None, None, :] * floor_a
        )
        frame *= self._vignette[..., None]
        return frame

    def _gradient(self) -> np.ndarray:
        stops = np.array([(22, 24, 36), (52, 52, 72), (78, 58, 56)], np.float32)
        pos = np.array([0.0, 0.6, 1.0])
        rows = np.stack(
            [
                np.interp(np.linspace(0.0, 1.0, HEIGHT), pos, stops[:, c])
                for c in range(3)
            ],
            axis=-1,
        ).astype(np.float32)
        return np.broadcast_to(rows[:, None, :], (HEIGHT, WIDTH, 3)).copy()

    def render(self, t: float) -> np.ndarray:
        frame = self._backdrop.copy()

        # Drifting soft bokeh discs with wrap-around.
        span = WIDTH + 240.0
        for speed, radius, soft, color, alpha, y0, x0, phase in self._bokeh:
            x = ((x0 + speed * t) % span) - 120.0
            y = y0 + 8.0 * np.sin(2.0 * np.pi * (t / DURATION_S) + phase)
            d = np.sqrt((self.xx - x) ** 2 + (self.yy - y) ** 2)
            a = alpha * _smoothstep((radius + soft - d) / soft)
            frame += a[..., None] * np.array(color, np.float32)

        # The skin-tone figure: shaded head + torso, softly edged.
        pose = figure_pose(t)
        if pose is not None:
            x, y = pose
            head_y = y - _HEAD_LIFT
            top = head_y - _HEAD_R
            bottom = y + _TORSO_RY
            shade = np.clip((self.yy - top) / max(bottom - top, 1.0), 0.0, 1.0)
            hi = np.array((233, 190, 152), np.float32)
            lo = np.array((156, 100, 64), np.float32)
            color = (
                hi[None, None, :] * (1.0 - shade[..., None])
                + lo[None, None, :] * shade[..., None]
            )
            dt = ((self.xx - x) / _TORSO_RX) ** 2 + ((self.yy - y) / _TORSO_RY) ** 2
            dh = ((self.xx - x) / _HEAD_R) ** 2 + ((self.yy - head_y) / _HEAD_R) ** 2
            alpha = np.maximum(
                _smoothstep((1.0 - dt) / 0.16),
                _smoothstep((1.0 - dh) / 0.20),
            )[..., None]
            frame = frame * (1.0 - alpha) + color * alpha

        # No film grain: per-pixel noise defeats GIF palette compression
        # (a grainy build measured 15 MiB). The bayer dither in the GIF
        # encode supplies the anti-banding texture instead.
        return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def write_scene_video() -> None:
    print(f"Rendering {N_FRAMES}-frame synthetic scene ...")
    renderer = SceneRenderer()
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(SYNTHETIC_MP4),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    assert proc.stdin is not None and proc.stderr is not None
    for i in range(N_FRAMES):
        proc.stdin.write(renderer.render(i / FPS).tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg scene encode failed:\n{stderr[-2000:]}")


def detection_scale() -> tuple[float, float]:
    """Scale factors ``overlay._scale_to_native`` will apply for the CPU
    profile, so fabricated plan boxes land back on the figure."""
    dw, dh = WIDTH, HEIGHT
    if dw > dh and dw > DETECTION_RES:
        dh = int(dh * (DETECTION_RES / dw))
        dw = DETECTION_RES
    elif dh > dw and dh > DETECTION_RES:
        dw = int(dw * (DETECTION_RES / dh))
        dh = DETECTION_RES
    dw -= dw % 2
    dh -= dh % 2
    return WIDTH / dw, HEIGHT / dh


def torso_box_native(t: float) -> tuple[int, int, int, int] | None:
    pose = figure_pose(t)
    if pose is None:
        return None
    x, y = pose
    return (
        int(x - _TORSO_RX - _BOX_PAD),
        int(y - _TORSO_RY - _BOX_PAD),
        int(x + _TORSO_RX + _BOX_PAD),
        int(y + _TORSO_RY + _BOX_PAD),
    )


def create_plan() -> None:
    print("Generating censor plan (boxes in CPU detection space) ...")
    metadata = probe_video(SYNTHETIC_MP4)
    try:
        pf_version = importlib.metadata.version("pureframe")
    except importlib.metadata.PackageNotFoundError:
        pf_version = "unknown"

    sw, sh = detection_scale()
    boxes = []
    for f in range(45, 105):  # the flagged shot: 1.5 s – 3.5 s
        native = torso_box_native(f / FPS)
        if native is None:
            continue
        x1, y1, x2, y2 = native
        boxes.append(
            {
                "x1": int(x1 / sw),
                "y1": int(y1 / sh),
                "x2": int(x2 / sw),
                "y2": int(y2 / sh),
                "frame_idx": f,
            }
        )

    plan = {
        "pureframe_version": pf_version,
        "plan_version": 1,
        "input_metadata": metadata.model_dump(),
        "config_snapshot": {"no_audio": True, "profile": "CPU"},
        "shots": [
            {
                "index": 0,
                "start_frame": 0,
                "end_frame": 45,
                "start_time": 0.0,
                "end_time": 1.5,
            },
            {
                "index": 1,
                "start_frame": 45,
                "end_frame": 105,
                "start_time": 1.5,
                "end_time": 3.5,
            },
            {
                "index": 2,
                "start_frame": 105,
                "end_frame": 150,
                "start_time": 3.5,
                "end_time": 5.0,
            },
        ],
        "verdicts": [
            {
                "shot_index": 1,
                "action": "BLACK_BOX",
                "category": "NUDITY_EXPLICIT",
                "confidence": 0.99,
                "reasoning": "Detected skin-tone figure crossing the frame",
                "boxes": boxes,
            }
        ],
        "total_censored_frames": len(boxes),
        "total_blur_frames": 0,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    PLAN_JSON.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def run_apply() -> None:
    print("Applying censor plan via PureFrame ...")
    subprocess.run(
        [
            "pureframe",
            "apply",
            str(SYNTHETIC_MP4),
            str(PLAN_JSON),
            "--output",
            str(OUTPUT_MP4),
        ],
        check=True,
        shell=False,
    )


def create_side_by_side_gif() -> None:
    print("Encoding labelled side-by-side GIF ...")
    label = (
        f"drawtext=fontfile={FONT}:text={{}}:fontsize=24:fontcolor=white:"
        "x=20:y=16:box=1:boxcolor=black@0.45:boxborderw=10"
    )
    filter_complex = (
        f"[0:v]{label.format(chr(39) + 'Original' + chr(39))}[l];"
        f"[1:v]{label.format(chr(39) + 'PureFrame' + chr(39))}[r];"
        "[l][r]hstack=inputs=2[s];"
        "[s]drawbox=x=852:y=0:w=4:h=ih:color=black:t=fill,"
        "fps=10,scale=1080:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff:max_colors=224[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(SYNTHETIC_MP4),
            "-i",
            str(OUTPUT_MP4),
            "-filter_complex",
            filter_complex,
            str(FINAL_GIF),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        shell=False,
    )
    size_mb = FINAL_GIF.stat().st_size / 1_048_576
    print(f"Done! GIF saved to {FINAL_GIF} ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(exist_ok=True)
    write_scene_video()
    create_plan()
    run_apply()
    create_side_by_side_gif()
