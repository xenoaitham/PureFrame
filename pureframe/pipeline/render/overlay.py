"""Shared overlay rendering helpers.

Provides a frame-level callback factory used by both the full re-encode
renderer (`apply.py`) and the smart segment renderer (`smart.py`).

The previous implementation used `cv2.rectangle(..., -1)` for ``BLACK_BOX``,
which painted a solid colour rectangle. The README advertised "smooth,
localized blur", so this module implements actual localized Gaussian blur
and pixelation modes, with the solid-colour rectangle kept as an optional
fallback (``BlurMode.BOX``) plus an emoji overlay (``BlurMode.EMOJI``):
one character, sized from the box, drawn over its center.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache

import cv2
import numpy as np

from pureframe.config import (
    DEFAULT_EMOJI,
    EMOJI_BY_CATEGORY,
    BlurMode,
    Config,
)
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.shots import Action

# Color-emoji fonts ship as fixed-size bitmap strikes: NotoColorEmoji only
# accepts pixel size 109, Apple Color Emoji its own strikes. We render the
# glyph at a strike size onto a transparent tile and resize that to the box.
_EMOJI_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto-color-emoji/NotoColorEmoji.ttf",
    "/usr/local/share/fonts/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/System/Library/Fonts/Supplemental/Apple Color Emoji.ttc",
    "C:\\Windows\\Fonts\\seguiemj.ttf",
)
_EMOJI_STRIKE_SIZES = (109, 160, 128)
_EMOJI_TILE = 256


def _scale_to_native(
    boxes: list[tuple[int, int, int, int]],
    frame_shape: tuple[int, int],
    det_res: int,
) -> list[tuple[int, int, int, int]]:
    """Scale detection-space boxes back to native frame coordinates."""
    h, w = frame_shape[:2]
    dw, dh = w, h
    if dw > dh and dw > det_res:
        dh = int(dh * (det_res / dw))
        dw = det_res
    elif dh > dw and dh > det_res:
        dw = int(dw * (det_res / dh))
        dh = det_res
    dw = dw - (dw % 2)
    dh = dh - (dh % 2)

    scale_w = w / dw if dw > 0 else 1.0
    scale_h = h / dh if dh > 0 else 1.0

    scaled: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        nx1 = max(0, min(w, int(x1 * scale_w)))
        ny1 = max(0, min(h, int(y1 * scale_h)))
        nx2 = max(0, min(w, int(x2 * scale_w)))
        ny2 = max(0, min(h, int(y2 * scale_h)))
        if nx2 > nx1 and ny2 > ny1:
            scaled.append((nx1, ny1, nx2, ny2))
    return scaled


def _apply_localized_blur(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    kernel: int,
    sigma: float,
) -> None:
    """In-place Gaussian blur on the ROI defined by *box*."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    # Kernel must be odd and >= 3.
    k = max(3, kernel | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), sigma)


def _apply_pixelate(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    blocks: int,
) -> None:
    """In-place mosaic/pixelate on the ROI."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    long_edge = max(w, h)
    n_blocks = max(2, min(blocks, long_edge))
    small_w = max(1, w * n_blocks // long_edge)
    small_h = max(1, h * n_blocks // long_edge)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_solid(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)


@lru_cache(maxsize=4)
def _load_emoji_font():
    """Return a PIL font able to draw color emoji, or None.

    Only real emoji fonts are probed - a symbol fallback would draw tofu
    boxes, which is worse than the solid-box fallback the caller applies.
    """
    from PIL import ImageFont

    for path in _EMOJI_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        for size in _EMOJI_STRIKE_SIZES:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return None


@lru_cache(maxsize=32)
def _render_emoji_tile(emoji_char: str) -> np.ndarray | None:
    """Render *emoji_char* centered on a transparent RGBA tile (BGR order).

    Returns None when no emoji font exists or the glyph came out empty;
    callers must fall back to an opaque censoring in that case so a box
    never renders as untouched pixels.
    """
    from PIL import Image, ImageDraw

    font = _load_emoji_font()
    if font is None:
        return None
    img = Image.new("RGBA", (_EMOJI_TILE, _EMOJI_TILE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (_EMOJI_TILE / 2, _EMOJI_TILE / 2),
        emoji_char,
        font=font,
        embedded_color=True,
        anchor="mm",
    )
    bbox = img.getbbox()
    if bbox is None:
        return None
    tile = img.crop(bbox)
    arr = np.array(tile)  # RGBA
    # RGBA -> BGRA so the alpha survives the cv2 resize and blending.
    bgra = arr[:, :, [2, 1, 0, 3]]
    return bgra


def _apply_emoji(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    emoji_char: str,
) -> None:
    """Draw *emoji_char* sized to and centered on *box*.

    The glyph is scaled so its larger dimension covers the box's larger
    dimension, keeping a boxy region visibly claimed even when the emoji
    itself is round. No font (or an empty glyph) falls back to a solid
    box: the region must never stay visible because a font was missing.
    """
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    tile = _render_emoji_tile(emoji_char)
    if tile is None:
        _apply_solid(frame, box, (0, 0, 0))
        return

    box_h, box_w = roi.shape[:2]
    side = max(box_w, box_h)
    tile = cv2.resize(tile, (side, side), interpolation=cv2.INTER_AREA)
    tile_h, tile_w = tile.shape[:2]

    # Center the tile on the ROI, cropping whatever overflows.
    x_off = (box_w - tile_w) // 2
    y_off = (box_h - tile_h) // 2
    src_x1, src_y1 = max(0, -x_off), max(0, -y_off)
    dst_x1, dst_y1 = max(0, x_off), max(0, y_off)
    overlap_w = min(tile_w - src_x1, box_w - dst_x1)
    overlap_h = min(tile_h - src_y1, box_h - dst_y1)
    if overlap_w <= 0 or overlap_h <= 0:
        return

    src = tile[src_y1 : src_y1 + overlap_h, src_x1 : src_x1 + overlap_w].astype(
        np.float32
    )
    alpha = src[:, :, 3:4] / 255.0
    dst = roi[dst_y1 : dst_y1 + overlap_h, dst_x1 : dst_x1 + overlap_w].astype(
        np.float32
    )
    blended = src[:, :, :3] * alpha + dst * (1.0 - alpha)
    roi[dst_y1 : dst_y1 + overlap_h, dst_x1 : dst_x1 + overlap_w] = blended.astype(
        roi.dtype
    )


def resolve_emoji(config: Config, category: str | None) -> str:
    """The emoji for a box: the explicit override, else the category
    default, else the generic marker."""
    if config.emoji_char:
        return config.emoji_char
    if category:
        return EMOJI_BY_CATEGORY.get(category, DEFAULT_EMOJI)
    return DEFAULT_EMOJI


def build_overlay_callback(
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
    frame_offset: int = 0,
) -> Callable[[int, np.ndarray], np.ndarray]:
    """Return an overlay callback compatible with ``write_video_with_overlay``.

    Parameters
    ----------
    frame_offset:
        Added to each ``frame_idx`` before lookup. Used by the smart segment
        renderer where ``frame_idx`` is local to the extracted segment but
        ``frame_actions`` is keyed by absolute frame index.
    """
    blur_mode = config.blur_mode
    blur_kernel = config.blur_kernel
    blur_sigma = config.blur_sigma
    pixelate_blocks = config.pixelate_blocks
    box_color = config.box_color
    det_res = profile_settings.detection_resolution

    def callback(frame_idx: int, frame_bgr: np.ndarray) -> np.ndarray:
        data = frame_actions.get(frame_idx + frame_offset)
        if not data:
            return frame_bgr

        action = data.get("action", Action.NONE)
        if action == Action.NONE:
            return frame_bgr

        if action == Action.FULL_FRAME_BLUR:
            k = max(3, blur_kernel | 1) * 2 + 1
            return cv2.GaussianBlur(frame_bgr, (k, k), blur_sigma * 1.5)

        if action == Action.BLACK_BOX:
            raw_boxes = data.get("boxes") or []
            if not raw_boxes:
                return frame_bgr
            boxes = _scale_to_native(raw_boxes, frame_bgr.shape, det_res)
            category = data.get("category")
            for box in boxes:
                if blur_mode == BlurMode.BLUR:
                    _apply_localized_blur(frame_bgr, box, blur_kernel, blur_sigma)
                elif blur_mode == BlurMode.PIXELATE:
                    _apply_pixelate(frame_bgr, box, pixelate_blocks)
                elif blur_mode == BlurMode.EMOJI:
                    _apply_emoji(frame_bgr, box, resolve_emoji(config, category))
                else:
                    _apply_solid(frame_bgr, box, box_color)

        return frame_bgr

    return callback
