#!/usr/bin/env bash
set -e
mkdir -p assets
INPUT=/tmp/pureframe_smoke/smoke_test.mp4

if [ ! -f "$INPUT" ]; then
    bash scripts/make_smoke_test_clip.sh
fi

python3 <<'PY'
import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    os.system("pip install Pillow -q")
    from PIL import Image, ImageDraw, ImageFont

BG = (15, 18, 28)
GREEN = (80, 220, 120)
CYAN = (100, 200, 255)
GRAY = (130, 140, 160)
WHITE = (230, 235, 245)

lines = [
    ("$ pureframe process movie.mkv --output movie.clean.mkv", CYAN),
    ("", WHITE),
    ("[*] Hardware detected: RTX 3060  →  MEDIUM profile", GREEN),
    ("[*] Probing: 1920×1080 @ 24fps, 1h 32m 14s", GRAY),
    ("[*] Shot detection: 1 842 shots found", GRAY),
    ("[*] Running visual + audio detection:  100%  ████████", GREEN),
    ("[*] Smoothing bounding-box tracks ...", GRAY),
    ("[*] Rendering filtered video + muxing audio ...", GRAY),
    ("", WHITE),
    ("[✓] Done!  movie.clean.mkv  |  8 shots filtered  |  41m 22s", GREEN),
]

W, LINE_H, PAD = 860, 32, 20
H = LINE_H * len(lines) + PAD * 2

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
except:
    font = ImageFont.load_default()

frames = []
for i in range(len(lines)):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    for j, (text, color) in enumerate(lines[:i+1]):
        draw.text((PAD, PAD + j * LINE_H), text, fill=color, font=font)
    frames.append(img)

# Hold last frame longer
frames += [frames[-1]] * 6

frames[0].save(
    "assets/demo.gif",
    save_all=True,
    append_images=frames[1:],
    duration=500,
    loop=0,
    optimize=True,
)
size = Path("assets/demo.gif").stat().st_size // 1024
print(f"Wrote assets/demo.gif  ({size} KB)")
PY
