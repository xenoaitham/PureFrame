"""Detection-only preprocessing for dark, low-contrast and B&W frames.

Very dark scenes and artistic black-and-white content compress the
dynamic range the detector was trained on, so the same explicit region
scores far below its well-lit equivalent. When a frame measures dark, or
near-grayscale with a compressed histogram, a percentile stretch (the
"simple gain" option - more predictable than CLAHE on the synthetic
frames the parity gate pins) restores the range before inference.

The renderer never sees these frames: normalization feeds the detector
only, and a linear map applied equally to all three channels does not
move box geometry, so the overlay lands on the original pixels.
Grayscale content stays a 3-channel equalized view (the map is shared),
which is exactly what the detector needs for B&W material.
"""

from __future__ import annotations

import cv2
import numpy as np

# Mean luma below this reads as a dark scene (bedroom, candlelight,
# night).
DARK_MEAN_THRESHOLD = 70.0
# Luma standard deviation below this reads as a compressed histogram -
# flat, hazy or crushed content.
LOW_CONTRAST_STD = 30.0
# Mean per-pixel channel spread at or below this reads as grayscale
# (artistic B&W, desaturated footage). Colored scenes never take the
# low-contrast branch - they keep their color information.
GRAYSCALE_CHANNEL_DEVIATION = 12.0
# Percentile stretch: robust against outliers where min/max is not.
STRETCH_PERCENTILES = (2.0, 98.0)
# A narrower span than this is codec noise on a flat frame; stretching
# it would only amplify grain, so the frame passes through unchanged.
MIN_STRETCH_SPAN = 8.0


def frame_mean_luma(frame: np.ndarray) -> float:
    """Mean of the luma plane (BGR input)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def channel_deviation(frame: np.ndarray) -> float:
    """Mean per-pixel spread between the strongest and weakest channel."""
    return float(
        np.mean(frame.max(axis=2).astype(np.int16) - frame.min(axis=2).astype(np.int16))
    )


def is_dark_frame(frame: np.ndarray) -> bool:
    return frame_mean_luma(frame) < DARK_MEAN_THRESHOLD


def is_low_contrast_grayscale(frame: np.ndarray) -> bool:
    """Near-grayscale content whose luma histogram is compressed."""
    if channel_deviation(frame) > GRAYSCALE_CHANNEL_DEVIATION:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < LOW_CONTRAST_STD


def needs_normalization(frame: np.ndarray) -> bool:
    """True when the frame should be stretched before inference."""
    return is_dark_frame(frame) or is_low_contrast_grayscale(frame)


def normalize_for_detection(frame: np.ndarray) -> np.ndarray:
    """Percentile stretch of the luma range, shared across all channels.

    A single linear map keeps grayscale content grayscale and preserves
    channel relationships in color content. Frames whose 2nd-98th
    percentile span is tiny pass through unchanged - there is no signal
    to recover, only grain to amplify.
    """
    lo, hi = np.percentile(frame, STRETCH_PERCENTILES)
    span = float(hi) - float(lo)
    if span < MIN_STRETCH_SPAN:
        return frame
    stretched = (frame.astype(np.float32) - float(lo)) * (255.0 / span)
    return np.clip(stretched, 0, 255).astype(np.uint8)
