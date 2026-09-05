from pathlib import Path

from pureframe.hardware import ProfileSettings
from pureframe.pipeline.detect.nudity import Detection, NudityDetector
from pureframe.pipeline.sample import extract_frames
from pureframe.pipeline.shots import Shot


def densify_shot(
    shot: Shot,
    video_path: Path,
    detector: NudityDetector,
    settings: ProfileSettings,
    threshold: float,
) -> dict[int, list[Detection]]:
    n = settings.densify_every_n_frames

    frame_indices = list(range(shot.start_frame, shot.end_frame, n))
    if shot.end_frame - 1 not in frame_indices:
        frame_indices.append(shot.end_frame - 1)

    frames_bgr = extract_frames(
        video_path, frame_indices, settings.detection_resolution
    )

    results = {}

    for idx in frame_indices:
        frame = frames_bgr.get(idx)
        if frame is not None:
            dets = detector.detect_batch([frame])[0]
            # filter by threshold
            valid_dets = [d for d in dets if d.score >= threshold]
            results[idx] = valid_dets
        else:
            results[idx] = []

    # Interpolation for frames between detection runs happens in
    # ``smooth_detections``, which runs an IoU tracker over the sparse
    # per-frame results produced here. We therefore return the known frames
    # as-is; the tracker bridges the gaps.
    return results
