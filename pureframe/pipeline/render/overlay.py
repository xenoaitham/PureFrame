"""Shared overlay rendering helpers.

Provides a frame-level callback factory used by both the full re-encode
renderer (`apply.py`) and the smart segment renderer (`smart.py`).

The previous implementation used `cv2.rectangle(..., -1)` for ``BLACK_BOX``,
which painted a solid colour rectangle. The README advertised "smooth,
localized blur", so this module implements actual localized Gaussian blur
and pixelation modes, with the solid-colour rectangle kept as an optional
fallback (``BlurMode.BOX``).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from pureframe.config import BlurMode, Config
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.shots import Action


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
            for box in boxes:
                if blur_mode == BlurMode.BLUR:
                    _apply_localized_blur(frame_bgr, box, blur_kernel, blur_sigma)
                elif blur_mode == BlurMode.PIXELATE:
                    _apply_pixelate(frame_bgr, box, pixelate_blocks)
                else:
                    _apply_solid(frame_bgr, box, box_color)

        return frame_bgr

    return callback
