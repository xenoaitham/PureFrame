"""Smart segment renderer - re-encodes only censored regions, copies the rest via stream copy.

Strategy:
1. Build a list of "dirty segments" from frame_actions (contiguous ranges needing overlay).
2. For each dirty segment, re-encode just that range with the overlay callback.
3. For clean segments between dirty ones, use ffmpeg stream copy (no re-encode).
4. Concatenate all segments using the concat demuxer.

This gives 2-5x speedup on typical content where only 5-15% of frames are censored.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from pureframe.config import Config
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.render.overlay import build_overlay_callback
from pureframe.utils.ffmpeg import probe, select_hw_encoder, write_video_with_overlay

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


def apply_censoring_smart(
    input_path: Path,
    output_path: Path,
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
    total_frames: int,
    fps: float,
) -> None:
    """Smart renderer that only re-encodes dirty segments.

    Falls back to full re-encode if:
    - More than 60% of video is dirty (not worth the concat overhead)
    - Only a single segment covers the whole video
    - ffmpeg concat fails for any reason
    """
    dirty_segments = _find_dirty_segments(frame_actions, total_frames, fps)

    if not dirty_segments:
        # No censoring needed - just stream copy
        logger.info("No dirty segments - stream copying entire video")
        _stream_copy(input_path, output_path)
        return

    # Calculate dirty ratio
    total_duration = total_frames / fps
    dirty_duration = sum(e - s for s, e in dirty_segments)
    dirty_ratio = dirty_duration / total_duration if total_duration > 0 else 1.0

    if dirty_ratio > 0.6:
        logger.info(
            f"Dirty ratio {dirty_ratio:.0%} > 60% - falling back to full re-encode"
        )
        from pureframe.pipeline.render.apply import apply_censoring

        apply_censoring(
            input_path, output_path, frame_actions, config, profile_settings
        )
        return

    logger.info(
        f"Smart render: {len(dirty_segments)} dirty segments "
        f"({dirty_duration:.1f}s / {total_duration:.1f}s = {dirty_ratio:.0%})"
    )

    try:
        _render_segments(
            input_path,
            output_path,
            frame_actions,
            dirty_segments,
            total_duration,
            config,
            profile_settings,
        )
    except Exception as e:
        logger.warning(f"Smart render failed ({e}), falling back to full re-encode")
        from pureframe.pipeline.render.apply import apply_censoring

        apply_censoring(
            input_path, output_path, frame_actions, config, profile_settings
        )


def _stream_copy(input_path: Path, output_path: Path) -> None:
    """Copy video without re-encoding."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c",
        "copy",
        "-map",
        "0",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _render_segments(
    input_path: Path,
    output_path: Path,
    frame_actions: dict[int, dict],
    dirty_segments: list[tuple[float, float]],
    total_duration: float,
    config: Config,
    profile_settings: ProfileSettings,
) -> None:
    """Render dirty segments with overlay, copy clean segments, then concatenate."""
    tmpdir = Path(tempfile.mkdtemp(prefix="pureframe_smart_"))
    segment_files = []

    try:
        # Build complete timeline: interleave clean copies and dirty re-encodes
        prev_end = 0.0

        for i, (dirty_start, dirty_end) in enumerate(dirty_segments):
            # Clean segment before this dirty one
            if dirty_start > prev_end + 0.1:
                clean_file = tmpdir / f"clean_{i:04d}.mkv"
                _extract_segment_copy(input_path, clean_file, prev_end, dirty_start)
                segment_files.append(clean_file)

            # Dirty segment - re-encode with overlay
            dirty_file = tmpdir / f"dirty_{i:04d}.mkv"
            _extract_and_render_segment(
                input_path,
                dirty_file,
                dirty_start,
                dirty_end,
                frame_actions,
                config,
                profile_settings,
            )
            segment_files.append(dirty_file)
            prev_end = dirty_end

        # Final clean segment after last dirty one
        if prev_end < total_duration - 0.1:
            final_clean = tmpdir / "clean_final.mkv"
            _extract_segment_copy(input_path, final_clean, prev_end, total_duration)
            segment_files.append(final_clean)

        # Concatenate all segments
        _concat_segments(segment_files, output_path, tmpdir)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_segment_copy(
    input_path: Path, output: Path, start: float, end: float
) -> None:
    """Extract a segment via stream copy (no re-encode)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(input_path),
        "-c",
        "copy",
        "-map",
        "0",
        "-avoid_negative_ts",
        "make_zero",
        str(output),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _extract_and_render_segment(
    input_path: Path,
    output: Path,
    start: float,
    end: float,
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
) -> None:
    """Extract a segment and re-encode it with the shared overlay callback.

    ``frame_actions`` is keyed by absolute frame index, but the underlying
    callback receives segment-local indices. We compute the segment's frame
    offset once and pass it via ``frame_offset`` so the shared overlay knows
    how to translate.
    """
    encoder = select_hw_encoder(profile_settings.profile, config.output_codec)
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
    )


def _ensure_within(base: Path, candidate: Path) -> Path:
    """Return ``candidate`` resolved, refusing paths that escape ``base``.

    Every path written into the concat file is one ffmpeg then reads back -
    a resolved path outside the working dir would let a corrupted segment
    name point ffmpeg at arbitrary files.
    """
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path {candidate} escapes allowed directory {base}")
    return resolved


def _concat_segments(segment_files: list[Path], output: Path, tmpdir: Path) -> None:
    """Concatenate segments using ffmpeg concat demuxer."""
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

    cmd = [
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
        str(output),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


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
