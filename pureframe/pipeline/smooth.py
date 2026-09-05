from collections import defaultdict

import numpy as np

from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.shots import Shot
from pureframe.tracking.iou_tracker import IoUTracker


def smooth_detections(
    per_frame_detections: dict[int, list[Detection]],
    shot: Shot,
    padding_pct: float,
) -> dict[int, list[tuple[int, int, int, int]]]:
    tracker = IoUTracker(iou_threshold=0.2, max_age=10)  # higher max age to bridge gaps

    track_history = defaultdict(list)

    # 1. Run IoU tracker across all frames in the shot
    for frame_idx in range(shot.start_frame, shot.end_frame):
        dets = per_frame_detections.get(frame_idx, [])
        active_tracks = tracker.update(dets)

        # record positions
        for t in active_tracks:
            track_history[t.track_id].append((frame_idx, t.box))

    # 2. For each track, apply median filter + interpolate + pad
    final_boxes = defaultdict(list)

    for tid, history in track_history.items():
        if not history:
            continue

        history.sort(key=lambda x: x[0])
        frames = [x[0] for x in history]
        boxes = [x[1] for x in history]

        # interpolate missing frames
        full_frames = list(range(frames[0], frames[-1] + 1))
        interp_boxes = []

        if len(frames) == 1:
            full_frames = frames
            interp_boxes = boxes
        else:
            boxes_np = np.array(boxes)
            x1 = np.interp(full_frames, frames, boxes_np[:, 0])
            y1 = np.interp(full_frames, frames, boxes_np[:, 1])
            x2 = np.interp(full_frames, frames, boxes_np[:, 2])
            y2 = np.interp(full_frames, frames, boxes_np[:, 3])

            for i in range(len(full_frames)):
                interp_boxes.append((x1[i], y1[i], x2[i], y2[i]))

        # apply median filter of size 5
        if len(interp_boxes) >= 5:
            ib_np = np.array(interp_boxes)
            from scipy.signal import medfilt

            m_x1 = medfilt(ib_np[:, 0], 5)
            m_y1 = medfilt(ib_np[:, 1], 5)
            m_x2 = medfilt(ib_np[:, 2], 5)
            m_y2 = medfilt(ib_np[:, 3], 5)

            for i in range(len(full_frames)):
                interp_boxes[i] = (m_x1[i], m_y1[i], m_x2[i], m_y2[i])

        # pad and assign
        for f, b in zip(full_frames, interp_boxes):
            x1, y1, x2, y2 = b
            w = x2 - x1
            h = y2 - y1

            px = w * padding_pct
            py = h * padding_pct

            nx1 = int(x1 - px)
            ny1 = int(y1 - py)
            nx2 = int(x2 + px)
            ny2 = int(y2 + py)

            # clip to non-negative
            nx1 = max(0, nx1)
            ny1 = max(0, ny1)
            nx2 = max(0, nx2)
            ny2 = max(0, ny2)

            # Ensure not zero area
            if nx2 > nx1 and ny2 > ny1:
                final_boxes[f].append((nx1, ny1, nx2, ny2))

    return dict(final_boxes)
