import logging
import subprocess
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import ffmpeg
import numpy as np
from pydantic import BaseModel

from pureframe.hardware import HardwareProfile, ProfileSettings

logger = logging.getLogger(__name__)


class PureFrameError(Exception):
    pass


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: Fraction
    duration_seconds: float
    total_frames: int
    has_audio: bool
    audio_streams: list[dict]
    subtitle_streams: list[dict]
    container: str
    video_codec: str
    pixel_format: str
    color_space: str
    is_hdr: bool


def probe(path: Path) -> dict:
    try:
        probe_result = ffmpeg.probe(str(path))
        return probe_result
    except ffmpeg.Error as e:
        msg = f"FFprobe error on {path}"
        if e.stderr:
            msg += f": {e.stderr.decode()}"
        raise PureFrameError(msg)
    except Exception as e:
        raise PureFrameError(f"Failed to probe {path}: {e}")


def extract_metadata(probe_result: dict) -> VideoMetadata:
    format_info = probe_result.get("format", {})
    streams = probe_result.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise PureFrameError("No video stream found.")

    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

    # Parse FPS
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, r_frame_rate.split("/"))
        fps = Fraction(num, den) if den != 0 else Fraction(0)
    except Exception:
        fps = Fraction(24, 1)  # Fallback

    duration = float(format_info.get("duration", 0.0))
    if "duration" in video_stream:
        duration = float(video_stream["duration"])

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    total_frames = int(video_stream.get("nb_frames", 0))
    if total_frames == 0 and fps > 0:
        total_frames = int(duration * float(fps))

    color_space = video_stream.get("color_space", "unknown")
    color_transfer = video_stream.get("color_transfer", "unknown")
    is_hdr = "smpte2084" in color_transfer or "arib-std-b67" in color_transfer

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
        total_frames=total_frames,
        has_audio=len(audio_streams) > 0,
        audio_streams=audio_streams,
        subtitle_streams=subtitle_streams,
        container=format_info.get("format_name", "unknown"),
        video_codec=video_stream.get("codec_name", "unknown"),
        pixel_format=video_stream.get("pix_fmt", "unknown"),
        color_space=color_space,
        is_hdr=is_hdr,
    )


def select_hw_encoder(profile: HardwareProfile, codec: str) -> str:
    # We map 'h264' or 'hevc'
    base = "libx264" if codec == "h264" else "libx265"
    if profile == HardwareProfile.CPU:
        return base

    # Attempt to read encoders
    try:
        output = subprocess.check_output(
            ["ffmpeg", "-encoders"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return base

    encoders: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("V"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            encoders.append(parts[1])

    if codec == "h264":
        if "h264_nvenc" in encoders:
            return "h264_nvenc"
        if "h264_videotoolbox" in encoders:
            return "h264_videotoolbox"
        if "h264_qsv" in encoders:
            return "h264_qsv"
        if "h264_amf" in encoders:
            return "h264_amf"
    elif codec == "hevc" or codec == "h265":
        if "hevc_nvenc" in encoders:
            return "hevc_nvenc"
        if "hevc_videotoolbox" in encoders:
            return "hevc_videotoolbox"
        if "hevc_qsv" in encoders:
            return "hevc_qsv"
        if "hevc_amf" in encoders:
            return "hevc_amf"

    return base


# Source codecs whose container forbids a codec change: WebM carries only
# VP8/VP9/AV1 and AVI expects MPEG-4 ASP (or Annex-B H.264). The concat
# demuxer also needs one codec across stream-copied and re-encoded chunks,
# so the re-encode must match the source. Software encoders only - these
# codecs have no reliable hardware paths on consumer machines.
_SOURCE_MATCHED_ENCODERS = {
    "vp8": "libvpx",
    "vp9": "libvpx-vp9",
    "mpeg4": "mpeg4",
}


def select_render_encoder(
    profile: HardwareProfile, output_codec: str, input_codec: str | None
) -> str:
    """Encoder for the re-encoded (censored) parts of a render.

    Re-encoded chunks are joined to stream-copied chunks of the original and
    written into the input's own container, so the codec has to follow the
    source: VP8/VP9 for WebM, MPEG-4 for AVI, and HEVC stays HEVC (mixing it
    with H.264 broke concat and silently forced a full re-encode). H.264
    sources - and anything we cannot match - keep honoring ``output_codec``
    exactly as before.
    """
    codec = (input_codec or "").lower()
    if codec in _SOURCE_MATCHED_ENCODERS:
        return _SOURCE_MATCHED_ENCODERS[codec]
    if codec in ("hevc", "h265"):
        return select_hw_encoder(profile, "hevc")
    return select_hw_encoder(profile, output_codec)


def probe_video_codec(path: Path) -> str | None:
    """``codec_name`` of the first video stream, or None if it cannot be read.

    Tolerant on purpose: the codec only steers encoder choice. If the probe
    fails, the render falls back to the configured codec and ffmpeg reports
    the real problem with the file.
    """
    try:
        return extract_metadata(probe(path)).video_codec
    except Exception as e:
        logger.debug(f"video codec probe failed for {path}: {e}")
        return None


def encoder_codec(encoder: str) -> str:
    """Codec family produced by *encoder* (``libx264`` → ``h264`` …)."""
    if encoder == "libvpx":
        return "vp8"
    if encoder == "libvpx-vp9":
        return "vp9"
    if encoder == "mpeg4":
        return "mpeg4"
    if encoder == "libx265" or encoder.startswith("hevc_"):
        return "hevc"
    return "h264"


def container_bsf_args(output_path: Path, video_codec: str | None) -> list[str]:
    """Bitstream-filter flags a stream copy into *output_path* needs.

    AVI wants Annex-B H.264; ffmpeg refuses the MP4-style stream our MKV
    intermediates carry ("no startcode found") unless told to convert it.
    """
    if output_path.suffix.lower() == ".avi" and (video_codec or "") == "h264":
        return ["-bsf:v", "h264_mp4toannexb"]
    return []


def _quality_args(encoder: str, crf: int) -> dict:
    """Constant-quality flags in each encoder's own vocabulary.

    x264/x265 and the hardware encoders take ``crf`` directly; libvpx needs
    ``b:v`` alongside it (0 = pure constant quality for VP9, a cap for VP8);
    the MPEG-4 encoder has no ``crf`` at all and uses ``qscale`` (2–31).
    """
    if encoder == "libvpx-vp9":
        return {"crf": min(max(crf, 0), 63), "b:v": 0, "cpu-used": 4, "row-mt": 1}
    if encoder == "libvpx":
        return {"crf": min(max(crf, 4), 63), "b:v": "4M", "cpu-used": 4}
    if encoder == "mpeg4":
        return {"qscale:v": min(max(round(crf / 2.5), 2), 31)}
    return {"crf": crf}


def frames_iter(
    path: Path, downscale_max_edge: int | None = None
) -> Iterator[np.ndarray]:
    probe_res = probe(path)
    meta = extract_metadata(probe_res)

    stream = ffmpeg.input(str(path))
    if downscale_max_edge:
        # scale down
        w, h = meta.width, meta.height
        if w > h and w > downscale_max_edge:
            h = int(h * (downscale_max_edge / w))
            w = downscale_max_edge
        elif h > w and h > downscale_max_edge:
            w = int(w * (downscale_max_edge / h))
            h = downscale_max_edge
        # Ensure dimensions are even
        w = w - (w % 2)
        h = h - (h % 2)
        stream = ffmpeg.filter(stream, "scale", w, h)
        out_w, out_h = w, h
    else:
        out_w, out_h = meta.width, meta.height

    process = stream.output("pipe:", format="rawvideo", pix_fmt="bgr24").run_async(
        pipe_stdout=True, pipe_stderr=True
    )

    # Drain stderr in background - Windows pipe buffers are ~4KB and ffmpeg
    # blocks writing stderr if we never read it, deadlocking process.wait().
    import threading

    def _drain(pipe):
        try:
            while pipe.read(65536):
                pass
        except Exception:
            pass

    drain_t = threading.Thread(target=_drain, args=(process.stderr,), daemon=True)
    drain_t.start()

    frame_size = out_w * out_h * 3
    try:
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) != frame_size:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([out_h, out_w, 3])
            yield frame
    finally:
        process.stdout.close()
        process.wait()
        drain_t.join(timeout=2.0)


def _encoder_preset_arg(encoder: str, preset: str | None) -> str | None:
    """Map the profile's speed preset onto the encoder, or drop it.

    The profile presets use the x264/x265 scale ("veryfast" etc.), which
    hardware encoders (nvenc/qsv/videotoolbox/amf) reject outright -
    nvenc: ``Unable to parse option value "veryfast"`` - and libvpx/mpeg4
    do not know at all. Those encoders keep their own defaults (libvpx gets
    its speed from ``cpu-used`` in :func:`_quality_args`).
    """
    return preset if preset and encoder in ("libx264", "libx265") else None


def write_video_with_overlay(
    input_path: Path,
    output_path: Path,
    overlay_callback: Callable[[int, np.ndarray], np.ndarray],
    settings: ProfileSettings,
    encoder: str,
    crf: int,
    ss: float | None = None,
    to: float | None = None,
    preset: str | None = None,
) -> None:
    """Re-encode *input_path* to *output_path*, applying *overlay_callback* per frame.

    Optional ``ss``/``to`` (seconds) restrict decoding to a sub-range, used by
    the smart segment renderer to re-encode only dirty segments. ``preset``
    sets the encoder speed/quality preset - ffmpeg's default ``medium`` costs
    2-3x encode time on software encoders, so profiles ship a faster one.
    """
    import os
    import tempfile

    meta = extract_metadata(probe(input_path))

    # temp file for video
    temp_fd, temp_video = tempfile.mkstemp(suffix=".mkv")
    os.close(temp_fd)

    logger.info(f"Writing temp video to {temp_video}")

    # 1. decode frames to raw BGR24 (optionally restricted to [ss, to])
    in_kwargs: dict = {}
    if ss is not None:
        in_kwargs["ss"] = f"{ss:.3f}"
    if to is not None:
        in_kwargs["to"] = f"{to:.3f}"
    process_in = (
        ffmpeg.input(str(input_path), **in_kwargs)
        .output("pipe:", format="rawvideo", pix_fmt="bgr24")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    # Drain stderr in a background thread. On Windows the OS pipe buffer is
    # ~4KB; if we never read it, ffmpeg blocks writing stderr and process.wait()
    # deadlocks. We keep the bytes so encoder failures can still be reported.
    import threading

    def _drain_pipe(pipe, sink):
        try:
            while True:
                chunk = pipe.read(65536)
                if not chunk:
                    break
                sink.append(chunk)
        except Exception:
            pass

    stderr_in_buf: list[bytes] = []
    t_in = threading.Thread(
        target=_drain_pipe, args=(process_in.stderr, stderr_in_buf), daemon=True
    )
    t_in.start()

    # We don't know exact count from the pipe, we just read
    out_kwargs = {
        "vcodec": encoder,
        "pix_fmt": "yuv420p",  # safe default
        **_quality_args(encoder, crf),
    }
    # Speed presets from the profile use the x264/x265 scale ("veryfast"
    # etc.), which hardware encoders reject outright (nvenc: "Unable to
    # parse option value"). Software encoders get the preset; hardware
    # encoders keep their defaults - they don't need it.
    preset_arg = _encoder_preset_arg(encoder, preset)
    if preset_arg:
        out_kwargs["preset"] = preset_arg

    # Colorspace is not passed through to avoid encoder compat issues;
    # ffmpeg will autodetect from input.

    # Ensure even dimensions (ffmpeg encoders require this for yuv420p)
    enc_width = meta.width - (meta.width % 2)
    enc_height = meta.height - (meta.height % 2)

    process_out = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{enc_width}x{enc_height}",
            r=float(meta.fps) if float(meta.fps) > 0 else 24.0,
        )
        .output(temp_video, **out_kwargs)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    stderr_out_buf: list[bytes] = []
    t_out = threading.Thread(
        target=_drain_pipe, args=(process_out.stderr, stderr_out_buf), daemon=True
    )
    t_out.start()

    frame_size = meta.width * meta.height * 3
    frame_idx = 0

    try:
        while True:
            in_bytes = process_in.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) != frame_size:
                break
            frame = (
                np.frombuffer(in_bytes, np.uint8)
                .reshape([meta.height, meta.width, 3])
                .copy()
            )

            # modify
            frame = overlay_callback(frame_idx, frame)

            # Crop to even dimensions if needed
            if enc_width != meta.width or enc_height != meta.height:
                frame = frame[:enc_height, :enc_width]

            try:
                # memoryview avoids a second full-frame copy; the frame
                # buffer is exclusively ours after the overlay callback.
                process_out.stdin.write(memoryview(frame))
            except BrokenPipeError:
                # Wait briefly for drain thread to flush remaining stderr.
                t_out.join(timeout=2.0)
                stderr_out = b"".join(stderr_out_buf).decode(errors="replace")
                # Show the tail of stderr (after the version banner)
                raise PureFrameError(
                    f"FFmpeg encoder crashed after {frame_idx} frames. "
                    f"stderr (tail): {stderr_out[-3000:]}"
                )
            frame_idx += 1
            if frame_idx % 1000 == 0:
                logger.debug(f"Rendered {frame_idx} frames...")
    finally:
        process_in.stdout.close()
        process_in.wait()
        if process_out.stdin and not process_out.stdin.closed:
            process_out.stdin.close()
        process_out.wait()
        # Allow drain threads to finish reading any trailing stderr.
        t_in.join(timeout=2.0)
        t_out.join(timeout=2.0)

    # Mux back. If a sub-range was decoded, slice audio/subs to match.
    logger.info("Muxing audio and subtitles...")
    mux_cmd: list[str] = ["ffmpeg", "-y", "-i", temp_video]
    if ss is not None:
        mux_cmd += ["-ss", f"{ss:.3f}"]
    if to is not None:
        mux_cmd += ["-to", f"{to:.3f}"]
    mux_cmd += [
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-c",
        "copy",
        *container_bsf_args(Path(output_path), encoder_codec(encoder)),
        str(output_path),
    ]
    try:
        subprocess.run(mux_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as sub_e:
        stderr_tail = b""
        if sub_e.stderr:
            stderr_tail = sub_e.stderr
        raise PureFrameError(
            f"Muxing failed (exit {sub_e.returncode}). stderr (tail): "
            f"{stderr_tail.decode(errors='replace')[-3000:]}"
        )
    finally:
        os.remove(temp_video)
