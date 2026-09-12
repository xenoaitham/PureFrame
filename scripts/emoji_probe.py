"""Render the emoji overlay's marker characters onto plain frames and
save them as PNGs, so a rendered frame from each CI platform can be
pulled from the run artifacts and eyeballed. A missing font or an empty
glyph shows up as either an untouched gray region or the solid-box
fallback instead of the character.

Usage:
    python scripts/emoji_probe.py [output-dir]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from pureframe.config import DEFAULT_EMOJI, EMOJI_BY_CATEGORY, BlurMode, Config
from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.render.overlay import (
    _apply_emoji,
    _load_emoji_font,
    _render_emoji_tile,
    build_overlay_callback,
)
from pureframe.pipeline.shots import Action

# Large box: the character at a comfortable size, drawn straight through
# the overlay's own apply path. Small boxes: two sizes through the real
# overlay callback, so the size-from-box scaling is visible too.
_LARGE_BOX = (40, 30, 280, 210)
_SMALL_BOXES = [(20, 40, 100, 90), (150, 100, 230, 150)]


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("emoji-probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    font = _load_emoji_font()
    print(f"emoji font resolved: {font is not None}")

    frame = np.full((240, 320, 3), 96, dtype=np.uint8)
    cfg = Config(input_path="d", output_path="d", blur_mode=BlurMode.EMOJI)
    settings = get_settings(HardwareProfile.CPU)
    callback = build_overlay_callback(
        {
            0: {
                "action": Action.BLACK_BOX,
                "boxes": _SMALL_BOXES,
                "category": "NUDITY_EXPLICIT",
            }
        },
        cfg,
        settings,
    )

    chars = sorted(set(EMOJI_BY_CATEGORY.values()) | {DEFAULT_EMOJI})
    for i, char in enumerate(chars):
        tile = _render_emoji_tile(char)
        shape = None if tile is None else (tile.shape[0], tile.shape[1])
        print(f"  {char!r}: tile={'ok' if tile is not None else 'EMPTY'} shape={shape}")

        large = frame.copy()
        _apply_emoji(large, _LARGE_BOX, char)
        cv2.imwrite(str(out_dir / f"{i}_large_{ord(char):x}.png"), large)

        small = callback(0, frame.copy())
        cv2.imwrite(str(out_dir / f"{i}_small_{ord(char):x}.png"), small)

    print(f"probe frames written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
