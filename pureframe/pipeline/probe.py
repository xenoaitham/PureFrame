"""Video probing: metadata plus container quirks the metadata lies about.

The plan pipeline trusts ``total_frames`` for shot boundaries and
keyframe sampling. Some containers overcount it: AVI muxes MPEG-4 with
a stream duration one tick longer than the frames it actually holds,
so ``nb_frames`` (and the ``duration * fps`` fallback) land one past
the real packet count. A sampled keyframe at that phantom index
extracts as nothing - and when a profile samples only two keyframes per
shot, a pairwise detector (the motion-blob plugin differencing
consecutive keyframes) receives a single-frame batch, finds nothing to
difference against, and the shot silently renders SAFE. Seen on real
AVI input during the adversarial critic pass; WebM/MP4/MKV of the same
content were unaffected.

Variable frame rate is the other lie: screen recordings and phone
timelapses carry uneven presentation timestamps, so "frame N is at
N / fps seconds" is wrong and blur windows land off-target. When the
probe detects VFR, :func:`convert_to_cfr` transcodes to a constant
frame rate intermediate the normal pipeline can trust - no more asking
users to run ffmpeg by hand.
"""

import subprocess
from pathlib import Path

from pureframe.utils.ffmpeg import VideoMetadata, extract_metadata, probe

# Containers whose frame-count metadata is known to overcount. Extend
# when a format proves untrustworthy; the recount is a demux-only pass,
# so the cost is seconds even on long files.
_OVERCOUNTING_CONTAINERS = {"avi"}

# VFR signals, as relative drifts. Peak vs average rate diverging past
# 2% is uneven timestamps; frames vs duration*avg drifting past 5%
# catches files whose rate metadata agrees but whose packet count does
# not match its own timeline.
VFR_RATE_DRIFT = 0.02
VFR_FRAME_DRIFT = 0.05


def _count_video_packets(path: Path) -> int:
    """Exact video packet count (demux only, no decoding)."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
        shell=False,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def is_vfr(meta: VideoMetadata) -> bool:
    """True when the metadata shows variable-frame-rate timing.

    Both checks need positive denominators to mean anything; a file with
    no average rate or no duration simply cannot prove itself VFR here.
    """
    peak = float(meta.fps)
    avg = float(meta.avg_fps)
    if peak > 0 and avg > 0 and abs(peak - avg) / peak > VFR_RATE_DRIFT:
        return True
    if meta.total_frames > 0 and meta.duration_seconds > 0 and avg > 0:
        expected = meta.duration_seconds * avg
        if abs(meta.total_frames - expected) / expected > VFR_FRAME_DRIFT:
            return True
    return False


def convert_to_cfr(
    source: Path,
    dest_dir: Path,
    meta: VideoMetadata,
    encoder: str,
    crf: int,
    preset: str | None = None,
) -> Path:
    """Transcode *source* to constant frame rate; return the new file.

    The intermediate lands in *dest_dir* as MKV - any codec, audio and
    subtitle streams go through untouched - and the video keeps its
    codec family via *encoder* so the render can still mux the result
    into the original container. *preset* follows the profile's encoder
    preset where the encoder understands one. Built on ffmpeg-python
    like the rest of the media plumbing; ``fps_mode`` needs ffmpeg
    >= 5.1 and falls back to the legacy ``vsync`` spelling once.
    """
    import ffmpeg as ffmpeg_python

    from pureframe.utils.ffmpeg import (
        PureFrameError,
        _encoder_preset_arg,
        _quality_args,
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{source.stem}.cfr.mkv"

    out_kwargs = {
        "fps_mode": "cfr",
        "r": float(meta.fps),
        "c:v": encoder,
        "c:a": "copy",
        "c:s": "copy",
    }
    out_kwargs.update(_quality_args(encoder, crf))
    preset_arg = _encoder_preset_arg(encoder, preset)
    if preset_arg:
        out_kwargs["preset"] = preset_arg

    def _run() -> None:
        (
            ffmpeg_python.input(str(source))
            .output(str(dest), **out_kwargs)
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )

    try:
        _run()
    except ffmpeg_python.Error as exc:
        stderr = (exc.stderr or b"").decode(errors="replace")
        if "fps_mode" in stderr:
            # Ancient ffmpeg: retry with the legacy -vsync spelling.
            out_kwargs["vsync"] = out_kwargs.pop("fps_mode")
            try:
                _run()
            except ffmpeg_python.Error as retry_exc:
                raise PureFrameError(
                    "CFR conversion failed: "
                    f"{(retry_exc.stderr or b'').decode(errors='replace')[-3000:]}"
                ) from retry_exc
        else:
            raise PureFrameError(f"CFR conversion failed: {stderr[-3000:]}") from exc
    return dest


def probe_video(path: Path) -> VideoMetadata:
    res = probe(path)
    meta = extract_metadata(res)
    if meta.container.split(",")[0].strip() in _OVERCOUNTING_CONTAINERS:
        counted = _count_video_packets(path)
        if counted > 0 and counted != meta.total_frames:
            meta = meta.model_copy(update={"total_frames": counted})
    return meta
