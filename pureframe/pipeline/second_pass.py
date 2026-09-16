"""Second-pass rescan: dense stride plus tiled zoom for candidate shots.

After the first pass, a shot is a candidate when it is not silent: it
either produced a detection at or above :data:`SECOND_PASS_FLOOR` (the
signal was there but under the bar) or sits inside a parental-guide
window (the guide marked it on external evidence). Candidates get one
bounded re-look - frames re-sample at the profile's densify stride, and
every frame the full-frame pass still reads below the shot's bar goes
through the overlapping tile grid (the small-object fix for distant or
screen-in-screen content).

The scan is per-shot, never whole-video, so the cost stays proportional
to the few shots that earned a second look.
"""

from __future__ import annotations

from pathlib import Path

from pureframe.hardware import ProfileSettings
from pureframe.pipeline.detect.nudity import Detection, NudityDetector
from pureframe.pipeline.detect.tiling import tiled_detect_frame
from pureframe.pipeline.sample import extract_frames
from pureframe.pipeline.shots import Shot

# Any first-pass detection at or above this floor makes the shot a
# rescan candidate: the content registered on the detector, just under
# the censoring bar. Deliberately not configurable - it is a cost guard,
# not a quality knob.
SECOND_PASS_FLOOR = 0.25


def shot_has_weak_signal(
    batch_dets: list[list[Detection]], floor: float = SECOND_PASS_FLOOR
) -> bool:
    """True when any first-pass detection reached the rescan floor."""
    return any(d.score >= floor for dets in batch_dets for d in dets)


def rescan_shot(
    shot: Shot,
    video_path: Path,
    detector: NudityDetector,
    settings: ProfileSettings,
    meta,
    threshold: float,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.15,
) -> dict[int, list[Detection]]:
    """Rescan one candidate shot; return per-frame detections at/above *threshold*.

    Dense-stride frames are detected full-frame first; a frame that still
    reads below *threshold* is re-detected through the tile grid before
    the next frame is considered, so tiles only ever run on frames the
    cheap pass could not resolve. *threshold* must be the exact bar
    fuse() used for this shot (effective nudity threshold times the
    strict-mode and guide factors) so the rescan can never surface a
    detection fuse would have rejected.
    """
    stride = max(1, settings.densify_every_n_frames)
    frame_indices = list(range(shot.start_frame, shot.end_frame, stride))
    if frame_indices and frame_indices[-1] != shot.end_frame - 1:
        frame_indices.append(shot.end_frame - 1)

    frames_bgr = extract_frames(
        video_path, frame_indices, settings.detection_resolution, meta=meta
    )

    results: dict[int, list[Detection]] = {}
    for idx in frame_indices:
        frame = frames_bgr.get(idx)
        if frame is None:
            continue
        dets = detector.detect_batch([frame])[0]
        passing = [d for d in dets if d.score >= threshold]
        if passing:
            results[idx] = passing
            continue
        tiled = tiled_detect_frame(frame, detector, grid=grid, overlap=overlap)
        tiled_pass = [d for d in tiled if d.score >= threshold]
        if tiled_pass:
            results[idx] = tiled_pass
    return results


def merge_rescan_detections(
    base: dict[int, list[Detection]],
    extra: dict[int, list[Detection]],
) -> dict[int, list[Detection]]:
    """Merge rescan detections into a per-frame dict (densify output)."""
    merged = dict(base)
    for idx, dets in extra.items():
        merged[idx] = list(merged.get(idx, [])) + list(dets)
    return merged
