"""Tiled inference: the standard small-object fix.

Small or distant explicit regions occupy too few pixels at full-frame
scale for the detector to score them confidently; zooming in raises their
effective resolution. These helpers split a frame into an overlapping
tile grid, run detection per tile upscaled back to the frame's detection
resolution, then map the boxes to native frame coordinates and merge
them with class-aware NMS so an object crossing a tile boundary is not
reported twice.

The geometry helpers are pure (no model) and unit-tested directly; the
detect entry points take any detector exposing ``detect_batch``.
"""

from __future__ import annotations

import cv2
import numpy as np

from .nudity import Detection

# A tile in native frame pixels: (x, y, w, h).
Tile = tuple[int, int, int, int]


def tile_grid(
    width: int, height: int, grid: tuple[int, int] = (2, 2), overlap: float = 0.15
) -> list[Tile]:
    """Split ``width x height`` into ``grid`` tiles sharing ``overlap``.

    Tile side is ``(dimension / n) * (1 + overlap)``, centered per row and
    column and clipped to the frame: the first tile starts at 0, the last
    ends at the edge, and neighbors overlap by exactly ``overlap`` of the
    base cell so an object cut by one boundary is fully inside the next
    tile. A single-cell grid (or a frame no larger than one cell) yields
    one whole-frame tile.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame must be non-empty, got {width}x{height}")
    cols = max(1, int(grid[0]))
    rows = max(1, int(grid[1]))
    overlap = min(max(overlap, 0.0), 0.9)

    def _axis(total: int, n: int) -> list[tuple[int, int]]:
        cell = total / n
        size = min(total, int(np.ceil(cell * (1 + overlap))))
        if n == 1 or size >= total:
            return [(0, total)]
        # Evenly spread the (total - size) slack across the n tiles.
        starts = [round(i * (total - size) / (n - 1)) for i in range(n)]
        return [(s, size) for s in starts]

    tiles: list[Tile] = []
    for y, th in _axis(height, rows):
        for x, tw in _axis(width, cols):
            tiles.append((x, y, tw, th))
    return tiles


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_detections(
    dets: list[Detection], iou_threshold: float = 0.5
) -> list[Detection]:
    """Class-aware greedy NMS: highest score wins, same-label overlaps die.

    Different labels never suppress each other - a boundary-crossing body
    can legitimately carry, say, a torso label and a breast label over
    nearly the same pixels.
    """
    order = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for det in order:
        if any(
            d.label == det.label and _iou(d.box, det.box) > iou_threshold for d in kept
        ):
            continue
        kept.append(det)
    return kept


def map_tile_detections(
    dets: list[Detection],
    tile: Tile,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> list[Detection]:
    """Map detection boxes from an upscaled tile back to frame coordinates.

    ``scale`` is the factor the tile crop was resized by before inference
    (1.0 when the tile was used as-is). Boxes are divided by it, shifted
    by the tile offset, and clipped to the frame.
    """
    tx, ty, _, _ = tile
    mapped: list[Detection] = []
    for det in dets:
        x1, y1, x2, y2 = det.box
        nx1 = min(max(int(round(x1 / scale)) + tx, 0), frame_width)
        ny1 = min(max(int(round(y1 / scale)) + ty, 0), frame_height)
        nx2 = min(max(int(round(x2 / scale)) + tx, 0), frame_width)
        ny2 = min(max(int(round(y2 / scale)) + ty, 0), frame_height)
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        mapped.append(
            Detection(label=det.label, score=det.score, box=(nx1, ny1, nx2, ny2))
        )
    return mapped


def tiled_detect_frame(
    frame: np.ndarray,
    detector,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.15,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Detect over an overlapping tile grid, merged back with NMS.

    Each tile is upscaled so its long edge matches the frame's long edge;
    a 2x2 grid with the default 15 percent overlap zooms each region by
    roughly 1.74x. Returns native-frame-space detections.
    """
    h, w = frame.shape[:2]
    target = max(w, h)
    merged: list[Detection] = []
    for tile in tile_grid(w, h, grid=grid, overlap=overlap):
        tx, ty, tw, th = tile
        crop = frame[ty : ty + th, tx : tx + tw]
        scale = target / max(tw, th)
        if scale > 1.01:
            resized = cv2.resize(
                crop,
                (max(1, int(round(tw * scale))), max(1, int(round(th * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            scale = 1.0
            resized = crop
        dets = detector.detect_batch([resized])[0]
        merged.extend(map_tile_detections(dets, tile, scale, w, h))
    return nms_detections(merged, iou_threshold=iou_threshold)


def tiled_detect_frames(
    frames: dict[int, np.ndarray],
    detector,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.15,
    iou_threshold: float = 0.5,
) -> dict[int, list[Detection]]:
    """Tile-detect a batch of frames keyed by frame index."""
    return {
        idx: tiled_detect_frame(
            frame, detector, grid=grid, overlap=overlap, iou_threshold=iou_threshold
        )
        for idx, frame in frames.items()
    }
