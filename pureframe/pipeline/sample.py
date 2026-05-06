from pathlib import Path
import numpy as np
import ffmpeg
from .shots import ShotVerdict
from pureframe.utils.ffmpeg import extract_metadata, probe


def sample_keyframes(shot: ShotVerdict, n: int) -> list[int]:
    length = shot.end_frame - shot.start_frame
    if length <= n:
        return list(range(shot.start_frame, shot.end_frame))

    # Evenly spaced
    # If n=3: start, mid, end-1
    indices = np.linspace(shot.start_frame, shot.end_frame - 1, n, dtype=int)
    return sorted(list(set(indices)))


def extract_frames(
    path: Path, frame_indices: list[int], downscale_max_edge: int
) -> dict[int, np.ndarray]:
    if not frame_indices:
        return {}

    meta = extract_metadata(probe(path))

    # Calculate scale
    w, h = meta.width, meta.height
    if w > h and w > downscale_max_edge:
        h = int(h * (downscale_max_edge / w))
        w = downscale_max_edge
    elif h > w and h > downscale_max_edge:
        w = int(w * (downscale_max_edge / h))
        h = downscale_max_edge
    w = w - (w % 2)
    h = h - (h % 2)

    # Build select filter string
    # select='eq(n,0)+eq(n,10)+...'
    select_expr = "+".join([f"eq(n,{i})" for i in frame_indices])

    process = (
        ffmpeg.input(str(path))
        .filter("select", select_expr)
        .filter("scale", w, h)
        .output("pipe:", format="rawvideo", pix_fmt="bgr24", vsync=0)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    frame_size = w * h * 3
    results = {}

    try:
        for idx in frame_indices:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) != frame_size:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([h, w, 3])
            results[idx] = frame
    finally:
        process.stdout.close()
        process.wait()

    return results
