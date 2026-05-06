from pathlib import Path
from pureframe.pipeline.shots import ShotVerdict
from pureframe.pipeline.detect.nudity import NudityDetector, Detection
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.sample import extract_frames


def densify_shot(
    shot: ShotVerdict,
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

    # Interpolate for frames between indices
    # We will do this properly during smoothing, but here we can just return the sparse detections.
    # The requirement: "For frames between detection runs, interpolate boxes linearly between the two surrounding detections."

    # Let's do simple nearest-neighbor or just pass to smooth.py to handle temporal tracking.
    # Actually, the tracker can handle missing frames if we pass empty lists, but we need boxes for every frame.

    densified = {}
    for i in range(len(frame_indices) - 1):
        idx1 = frame_indices[i]

        dets1 = results[idx1]

        densified[idx1] = dets1

        # We need to match detections to interpolate
        # Simple heuristic: if same label and high IoU, interpolate
        # For Phase 1, the prompt says "interpolate boxes linearly between the two surrounding detections."
        # The best place for this is actually `smooth_detections` which uses an IoU tracker over the whole shot.
        # So we'll just populate the known frames and let smooth handle interpolation.

    densified[frame_indices[-1]] = results[frame_indices[-1]]

    return densified
