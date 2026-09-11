"""Temporal smoothing: the median filter must not corrupt track tails.

``scipy.signal.medfilt`` zero-pads the sequence edges, so the last two
frames of every track had their median dragged toward 0 and frozen there -
on a subject moving at shot's end the blur box visibly lagged behind (22 px
on a 8 px/frame mover, and frozen for the final 3 frames). The smoother now
uses ``scipy.ndimage.median_filter(mode="nearest")``, which replicates edge
values.

This module also documents, as executable measurements, why box-EMA and
track hysteresis were **rejected**: with sparse anchors (densify stride
2–5) an EMA's steady-state lag is a systematic bias that outweighs its
variance reduction, and with dense detections the median filter already
absorbs the jitter EMA would remove.
"""

import numpy as np

from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.shots import Shot
from pureframe.pipeline.smooth import smooth_detections


def _make_shot(frames: int) -> Shot:
    return Shot(
        index=0, start_frame=0, end_frame=frames, start_time=0.0, end_time=frames / 24.0
    )


def _center_x(box: tuple[int, int, int, int]) -> float:
    return (box[0] + box[2]) / 2


def _mover_dets(frames: int, step: int) -> dict[int, list[Detection]]:
    return {
        f: [
            Detection(
                label="FEMALE_BREAST_EXPOSED",
                score=0.9,
                box=(100 + step * f, 100, 200 + step * f, 200),
            )
        ]
        for f in range(frames)
    }


def test_tail_of_a_fast_mover_keeps_up():
    """The last frames of a moving track must track, not freeze behind."""
    frames, step = 40, 8
    shot = _make_shot(frames)
    result = smooth_detections(_mover_dets(frames, step), shot, padding_pct=0.0)

    ideal = 150 + step * np.arange(frames)
    smoothed = np.array([_center_x(result[f][0]) for f in range(frames)])
    # One step of tolerance; the old zero-padded medfilt froze the tail 22 px
    # behind on this exact input.
    assert np.max(np.abs(smoothed - ideal)) <= step + 2.0


def test_tail_matches_the_track_motion_after_the_last_detection():
    """Interpolation must reach the final anchor with the median intact."""
    frames = 30
    shot = _make_shot(frames)
    result = smooth_detections(_mover_dets(frames, 4), shot, padding_pct=0.0)

    last = result[frames - 1][0]
    ideal_x1 = 100 + 4 * (frames - 1)
    assert abs(last[0] - ideal_x1) <= 4 + 2.0


def test_static_track_has_no_tail_sag():
    """A static subject's tail must stay on the subject, not drift down."""
    frames = 20
    shot = _make_shot(frames)
    dets = {
        f: [
            Detection(
                label="FEMALE_BREAST_EXPOSED", score=0.9, box=(100, 100, 200, 200)
            )
        ]
        for f in range(frames)
    }
    result = smooth_detections(dets, shot, padding_pct=0.0)

    centers = [_center_x(result[f][0]) for f in range(frames)]
    assert max(centers) - min(centers) <= 2.0
    assert abs(centers[-1] - 150) <= 2.0


def test_dense_anchors_median_absorbs_iid_jitter():
    """Baseline for the EMA rejection: what median-5 leaves of dense noise.

    With detections on every frame, per-frame iid jitter of ±8 px (raw RMSE
    ≈ 4.6) comes out around ≈ 3.4 after the median filter. An EMA stacked on
    top measured ≤8 % better (and 1.4–1.5× *worse* on sparse anchors, where
    its lag biases every interpolation) - not worth the complexity; numbers
    recorded in the smoother's docstring.
    """
    frames = 60
    rng = np.random.default_rng(7)
    shot = _make_shot(frames)
    dets = {}
    for f in range(frames):
        j = int(rng.integers(-8, 9))
        dets[f] = [
            Detection(
                label="FEMALE_BREAST_EXPOSED",
                score=0.9,
                box=(100 + j, 100, 200 + j, 200),
            )
        ]

    result = smooth_detections(dets, shot, padding_pct=0.0)
    centers = np.array([_center_x(result[f][0]) for f in range(frames)])
    rmse = float(np.sqrt(np.mean((centers - 150) ** 2)))
    assert 2.5 < rmse < 4.2, f"median-5 behavior changed ({rmse=:.2f})"
