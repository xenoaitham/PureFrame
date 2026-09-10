"""Smart segment renderer — re-encodes only censored regions, copies the rest via stream copy.

Strategy:
1. Build a list of "dirty segments" from frame_actions (contiguous ranges needing overlay).
2. For each dirty segment, re-encode just that range with the overlay callback.
3. For clean segments between dirty ones, use ffmpeg stream copy (no re-encode).
4. Concatenate all segments using the concat demuxer.

This gives 2-5x speedup on typical content where only 5-15% of frames are censored.

Segment cuts must land on stream keyframes: a stream-copy cut can only begin
at a sync point, so bounds are snapped outward to the probed keyframe map
(see ``_snap_to_keyframes``).
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from pureframe.config import Config
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.render.overlay import build_overlay_callback
from pureframe.utils.ffmpeg import (
    container_bsf_args,
    probe,
    probe_video_codec,
    select_hw_encoder,
    select_render_encoder,
    write_video_with_overlay,
)

logger = logging.getLogger(__name__)


def _find_dirty_segments(
    frame_actions: dict[int, dict],
    total_frames: int,
    fps: float,
    padding_seconds: float = 0.5,
) -> list[tuple[float, float]]:
    """Find time ranges that need re-encoding (dirty segments).

    Returns list of (start_seconds, end_seconds) tuples.
    Merges nearby segments within `padding_seconds` to avoid excessive splits.
    """
    if not frame_actions:
        return []

    dirty_frames = sorted(frame_actions.keys())
    if not dirty_frames:
        return []

    # Build contiguous dirty ranges
    ranges = []
    start = dirty_frames[0]
    end = dirty_frames[0]

    for f in dirty_frames[1:]:
        if f <= end + 1:
            end = f
        else:
            ranges.append((start, end))
            start = f
            end = f
    ranges.append((start, end))

    # Convert to seconds with padding
    pad_frames = int(padding_seconds * fps)
    time_ranges = []
    for s, e in ranges:
        t_start = max(0, (s - pad_frames)) / fps
        t_end = min(total_frames - 1, (e + pad_frames)) / fps
        time_ranges.append((t_start, t_end))

    # Merge overlapping ranges
    merged = [time_ranges[0]]
    for s, e in time_ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def _probe_keyframe_times(path: Path) -> list[float]:
    """Timestamps of all video keyframes in *path*, ascending.

    Stream copy can only begin a cut at a keyframe: an input-side ``-ss``
    snaps back to the previous sync point and silently duplicates everything
    from there (a single-keyframe clip duplicates the whole video). Segment
    planning therefore needs the real keyframe map.

    The map is read from packet flags — demux only, no decoding — which is
    both faster on long inputs and codec-independent. The earlier
    decode-based probe (``-skip_frame nokey``) relied on decoder support:
    the VP8/VP9 decoders ignore that flag and reported *every* frame as a
    keyframe, which would have put copy cuts on non-keyframes for WebM.
    Packets without a presentation timestamp (AVI/H.264) are skipped; such
    inputs take the full re-encode fallback.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,flags",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
        shell=False,
    )
    times: list[float] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        pts, flags = parts[0], parts[-1]
        if "K" in flags and pts and pts != "N/A":
            times.append(float(pts))
    if not times:
        raise RuntimeError(f"no keyframes found in {path}")
    return sorted(times)


def _snap_to_keyframes(
    ranges: list[tuple[float, float]],
    keyframes: list[float],
    duration: float,
) -> list[tuple[float, float]]:
    """Expand segment bounds outward so every cut lands on a keyframe.

    Clean segments are stream-copied, and copy cuts are only exact at
    keyframes; dirty segments are re-encoded whole-GOP so their bounds line
    up with the copies around them. Overlaps created by the expansion are
    merged. ``duration`` is the fallback end for ranges extending past the
    last keyframe (the video tail).
    """
    snapped: list[tuple[float, float]] = []
    for s, e in ranges:
        k_start = 0.0
        for k in keyframes:  # ascending: last keyframe <= s
            if k <= s + 1e-6:
                k_start = k
            else:
                break
        k_end = duration
        for k in reversed(keyframes):  # descending: first keyframe >= e
            if k >= e - 1e-6:
                k_end = k
            else:
                break
        if k_end < k_start:  # degenerate (zero-duration inputs) — clamp
            k_end = k_start
        snapped.append((k_start, k_end))

    merged = [snapped[0]]
    for s, e in snapped[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def apply_censoring_smart(
    input_path: Path,
    output_path: Path,
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
    total_frames: int,
    fps: float,
    input_codec: str | None = None,
) -> None:
    """Smart renderer that only re-encodes dirty segments.

    ``input_codec`` (the source's video codec, probed when omitted) steers
    the re-encode so it matches the stream-copied chunks and the input's
    container — WebM must stay VP8/VP9, AVI stays MPEG-4/Annex-B H.264.

    Falls back to full re-encode if:
    - More than 60% of video is dirty (not worth the concat overhead)
    - Only a single segment covers the whole video
    - The keyframe map cannot be probed (copy cuts would be inexact)
    - ffmpeg concat fails for any reason
    """
    padded_ranges = _find_dirty_segments(frame_actions, total_frames, fps)

    if not padded_ranges:
        # No censoring needed — just stream copy
        logger.info("No dirty segments — stream copying entire video")
        _stream_copy(input_path, output_path)
        return

    if input_codec is None:
        input_codec = probe_video_codec(input_path)

    try:
        keyframes = _probe_keyframe_times(input_path)
    except Exception as e:
        logger.warning(f"Keyframe probe failed ({e}) — falling back to full re-encode")
        from pureframe.pipeline.render.apply import apply_censoring

        apply_censoring(
            input_path,
            output_path,
            frame_actions,
            config,
            profile_settings,
            input_codec=input_codec,
        )
        return

    # Cuts must sit on keyframes, or stream copy duplicates whole GOPs.
    total_duration = total_frames / fps
    dirty_segments = _snap_to_keyframes(padded_ranges, keyframes, total_duration)

    # Calculate dirty ratio
    dirty_duration = sum(e - s for s, e in dirty_segments)
    dirty_ratio = dirty_duration / total_duration if total_duration > 0 else 1.0

    if dirty_ratio > 0.6:
        logger.info(
            f"Dirty ratio {dirty_ratio:.0%} > 60% — falling back to full re-encode"
        )
        from pureframe.pipeline.render.apply import apply_censoring

        apply_censoring(
            input_path,
            output_path,
            frame_actions,
            config,
            profile_settings,
            input_codec=input_codec,
        )
        return

    logger.info(
        f"Smart render: {len(dirty_segments)} dirty segments "
        f"({dirty_duration:.1f}s / {total_duration:.1f}s = {dirty_ratio:.0%})"
    )

    try:
        # Encoder selection and fps probing are per-RENDER facts, not
        # per-segment facts — resolve once and thread through the segments
        # instead of re-running `ffmpeg -encoders` + ffprobe N times.
        encoder = select_render_encoder(
            profile_settings.profile, config.output_codec, input_codec
        )
        _render_segments(
            input_path,
            output_path,
            frame_actions,
            dirty_segments,
            total_duration,
            config,
            profile_settings,
            encoder=encoder,
            fps=fps,
            video_codec=input_codec,
        )
    except Exception as e:
        logger.warning(f"Smart render failed ({e}), falling back to full re-encode")
        from pureframe.pipeline.render.apply import apply_censoring

        apply_censoring(
            input_path,
            output_path,
            frame_actions,
            config,
            profile_settings,
            input_codec=input_codec,
        )


def _stream_copy(input_path: Path, output_path: Path) -> None:
    """Copy video without re-encoding."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c",
            "copy",
            "-map",
            "0",
            str(output_path),
        ],
        capture_output=True,
        check=True,
        shell=False,
    )


def _render_segments(
    input_path: Path,
    output_path: Path,
    frame_actions: dict[int, dict],
    dirty_segments: list[tuple[float, float]],
    total_duration: float,
    config: Config,
    profile_settings: ProfileSettings,
    encoder: str | None = None,
    fps: float | None = None,
    video_codec: str | None = None,
) -> None:
    """Re-encode dirty segments with overlay; stream-copy the rest; concatenate.

    Copy cuts cannot be made with ``-ss``/``-to`` on a plain ``-c copy``:
    the demuxer cuts in decode order, so B-frame tails truncate or shift the
    boundary (observed as a 9.1s render of a 5s clip). Instead the input is
    split at the keyframe-aligned dirty boundaries with the segment muxer —
    which cuts losslessly on keyframes — and only chunks inside dirty
    ranges are replaced by re-encoded overlay renders before concat.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="pureframe_smart_"))

    try:
        boundaries = sorted({t for s, e in dirty_segments for t in (s, e)})
        # Chunk intervals must align with the bounds the muxer actually
        # cuts at, so filter here (0.0 and end-of-video are implicit).
        bounds = [t for t in boundaries if 1e-6 < t < total_duration - 1e-6]
        chunks = _segment_at_boundaries(input_path, tmpdir, bounds, total_duration)

        segment_files: list[Path] = []
        start = 0.0
        for i, chunk in enumerate(chunks):
            end = bounds[i] if i < len(bounds) else total_duration
            if _overlaps_any(start, end, dirty_segments):
                rendered = tmpdir / f"dirty_{i:04d}.mkv"
                _extract_and_render_segment(
                    input_path,
                    rendered,
                    start,
                    end,
                    frame_actions,
                    config,
                    profile_settings,
                    encoder=encoder,
                    fps=fps,
                )
                segment_files.append(rendered)
            else:
                segment_files.append(chunk)
            start = end

        # Concatenate all segments
        _concat_segments(segment_files, output_path, tmpdir, video_codec=video_codec)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _overlaps_any(start: float, end: float, ranges: list[tuple[float, float]]) -> bool:
    eps = 1e-6
    return any(s < end - eps and e > start + eps for s, e in ranges)


def _segment_at_boundaries(
    input_path: Path,
    tmpdir: Path,
    boundaries: list[float],
    total_duration: float,
) -> list[Path]:
    """Split *input_path* at *boundaries* via the segment muxer (lossless).

    Returns one chunk file per interval, timestamps reset per chunk. Cuts
    at 0.0 (implied) and at/after the video end are skipped; a cut that is
    not exactly on a keyframe snaps forward within ``segment_time_delta``.
    """
    bounds = [t for t in boundaries if 1e-6 < t < total_duration - 1e-6]
    pattern = str(tmpdir / "chunk_%04d.mkv")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-f",
            "segment",
            "-segment_times",
            ",".join(f"{t:.3f}" for t in bounds),
            "-segment_time_delta",
            "0.05",
            "-c",
            "copy",
            "-map",
            "0",
            "-avoid_negative_ts",
            "make_zero",
            "-reset_timestamps",
            "1",
            pattern,
        ],
        capture_output=True,
        check=True,
        shell=False,
    )
    chunks: list[Path] = []
    for i in range(len(bounds) + 1):
        chunk = tmpdir / f"chunk_{i:04d}.mkv"
        if not chunk.exists() or chunk.stat().st_size == 0:
            raise RuntimeError(f"segment muxer produced no chunk {i}")
        chunks.append(chunk)
    return chunks


def _extract_and_render_segment(
    input_path: Path,
    output: Path,
    start: float,
    end: float,
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
    encoder: str | None = None,
    fps: float | None = None,
) -> None:
    """Extract a segment and re-encode it with the shared overlay callback.

    ``frame_actions`` is keyed by absolute frame index, but the underlying
    callback receives segment-local indices. We compute the segment's frame
    offset once and pass it via ``frame_offset`` so the shared overlay knows
    how to translate. ``encoder``/``fps`` are resolved once per render by the
    caller; both fall back to per-call resolution when omitted.
    """
    if encoder is None:
        encoder = select_hw_encoder(profile_settings.profile, config.output_codec)
    if fps is None:
        fps = _get_fps(input_path)
    frame_offset = int(round(start * fps))

    overlay_callback = build_overlay_callback(
        frame_actions, config, profile_settings, frame_offset=frame_offset
    )

    write_video_with_overlay(
        input_path=input_path,
        output_path=output,
        overlay_callback=overlay_callback,
        settings=profile_settings,
        encoder=encoder,
        crf=config.output_crf,
        ss=start,
        to=end,
        preset=profile_settings.encoder_preset,
    )


def _ensure_within(base: Path, candidate: Path) -> Path:
    """Return ``candidate`` resolved, refusing paths that escape ``base``.

    Every path written into the concat file is one ffmpeg then reads back —
    a resolved path outside the working dir would let a corrupted segment
    name point ffmpeg at arbitrary files.
    """
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path {candidate} escapes allowed directory {base}")
    return resolved


def _concat_segments(
    segment_files: list[Path],
    output: Path,
    tmpdir: Path,
    video_codec: str | None = None,
) -> None:
    """Concatenate segments using ffmpeg concat demuxer.

    ``video_codec`` (the stream's codec) lets the copy add any bitstream
    filter the output container demands — AVI needs Annex-B H.264.
    """
    tmpdir = tmpdir.resolve()
    concat_file = _ensure_within(tmpdir, tmpdir / "concat.txt")
    lines = []
    for sf in segment_files:
        if sf.exists() and sf.stat().st_size > 0:
            sf = _ensure_within(tmpdir, sf)
            # Escape single quotes for the concat demuxer: 'it's.mp4'
            # becomes 'it'\''s.mp4'. Without this, a quote in the path
            # terminates the quoted token and the remainder is parsed as
            # concat directives.
            safe = str(sf).replace("'", "'\\''")
            lines.append(f"file '{safe}'\n")
    concat_file.write_text("".join(lines), encoding="utf-8")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-map",
            "0",
            *container_bsf_args(output, video_codec),
            str(output),
        ],
        capture_output=True,
        check=True,
        shell=False,
    )


def _get_fps(path: Path) -> float:
    """Get FPS from video file."""
    from fractions import Fraction

    probe_data = probe(path)
    for stream in probe_data["streams"]:
        if stream["codec_type"] == "video":
            try:
                return float(Fraction(stream["r_frame_rate"]))
            except (ValueError, ZeroDivisionError):
                return 24.0
    return 24.0
