import threading
from pathlib import Path

import ffmpeg
import numpy as np

from pureframe.utils.ffmpeg import PureFrameError, extract_metadata, probe

from .shots import Shot


def sample_keyframes(shot: Shot, n: int) -> list[int]:
    length = shot.end_frame - shot.start_frame
    if length <= n:
        return list(range(shot.start_frame, shot.end_frame))

    # Evenly spaced
    # If n=3: start, mid, end-1
    indices = np.linspace(shot.start_frame, shot.end_frame - 1, n, dtype=int)
    # Plain ints: these land in Shot.frames keys and Box.frame_idx, and the
    # numpy scalars break pydantic's strict JSON serialization of the plan.
    return sorted({int(i) for i in indices})


def _spawn_decode(
    path: Path,
    select_expr: str,
    w: int,
    h: int,
    use_fps_mode: bool,
    seek_time: float | None = None,
):
    """Spawn the ffmpeg frame-selection decoder.

    ``-vsync 0`` was removed in ffmpeg 7; its replacement is
    ``-fps_mode passthrough`` (ffmpeg >= 5.1). Callers start with the modern
    option and retry with the legacy one when the local ffmpeg rejects it.

    ``seek_time`` (when set) becomes an input seek so ffmpeg only decodes
    from the keyframe before the requested window instead of the start of
    the file - without it, every per-shot extraction re-decoded the whole
    video from frame 0, a near-quadratic cost across a movie's shots.
    """
    out_kwargs = {"format": "rawvideo", "pix_fmt": "bgr24"}
    if use_fps_mode:
        out_kwargs["fps_mode"] = "passthrough"
    else:
        out_kwargs["vsync"] = 0

    input_stream = (
        ffmpeg.input(str(path), ss=seek_time) if seek_time else ffmpeg.input(str(path))
    )
    return (
        input_stream.filter("select", select_expr)
        .filter("scale", w, h)
        .output("pipe:", **out_kwargs)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )


def _drain(pipe, tail: list) -> None:
    """Consume ffmpeg's stderr in a background thread.

    Long `select` expressions make ffmpeg emit per-frame warnings; if nobody
    reads stderr, the OS pipe buffer fills up, ffmpeg blocks, and the stdout
    read deadlocks. The last chunks are kept so failures can be reported.
    """
    try:
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            tail.append(chunk)
            while len(tail) > 4 and sum(len(c) for c in tail) > 262144:
                tail.pop(0)
    except Exception:
        pass


def extract_frames(
    path: Path,
    frame_indices: list[int],
    downscale_max_edge: int,
    meta=None,
) -> dict[int, np.ndarray]:
    """Extract the requested frames as BGR arrays.

    Uses an input seek to the requested window (constant-framerate videos).
    With accurate seek, decoded frame ``n=0`` is exactly frame
    ``first_index - 1``, so the select filter matches relative indices and
    never re-decodes the video from its start. Pass ``meta`` (from
    ``extract_metadata(probe(path))``) to reuse an already-probed result
    instead of spawning an ffprobe per call.
    """
    if not frame_indices:
        return {}

    if meta is None:
        meta = extract_metadata(probe(path))

    indices = sorted(frame_indices)
    first = indices[0]

    fps = float(getattr(meta, "fps", 0.0) or 0.0)
    # Seek so decoded n=0 is exactly frame ``first - 1``: land the seek point
    # strictly between frames first-2 and first-1 (frame k has pts k/fps).
    seek_frame = first - 1
    seek_time = (seek_frame - 0.5) / fps if (fps > 0 and seek_frame > 0) else None
    if seek_time is not None and seek_time < 0:
        seek_time = None

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

    # Select RELATIVE frame numbers when seeking (n counts decoded frames
    # after the seek point), absolute otherwise.
    rel_base = seek_frame if seek_time is not None else 0
    select_expr = "+".join([f"eq(n,{i - rel_base})" for i in indices])

    # First attempt uses -fps_mode (ffmpeg >= 5.1, required on 7+); if the
    # local ffmpeg predates it, retry once with the legacy -vsync form.
    for use_fps_mode in (True, False):
        process = _spawn_decode(path, select_expr, w, h, use_fps_mode, seek_time)
        tail: list = []
        drain_t = threading.Thread(
            target=_drain, args=(process.stderr, tail), daemon=True
        )
        drain_t.start()

        frame_size = w * h * 3
        results = {}

        try:
            for idx in indices:
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes or len(in_bytes) != frame_size:
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape([h, w, 3])
                results[idx] = frame
        finally:
            process.stdout.close()
            process.wait()
            drain_t.join(timeout=2.0)

        if process.returncode == 0:
            return results

        stderr_text = b"".join(tail).decode(errors="replace")
        if use_fps_mode and "fps_mode" in stderr_text:
            continue  # ancient ffmpeg: fall back to -vsync and decode again

        raise PureFrameError(
            f"Frame extraction failed (exit {process.returncode}). "
            f"stderr (tail): {stderr_text[-3000:]}"
        )

    return {}
