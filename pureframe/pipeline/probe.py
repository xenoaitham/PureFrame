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
"""

import subprocess
from pathlib import Path

from pureframe.utils.ffmpeg import VideoMetadata, extract_metadata, probe

# Containers whose frame-count metadata is known to overcount. Extend
# when a format proves untrustworthy; the recount is a demux-only pass,
# so the cost is seconds even on long files.
_OVERCOUNTING_CONTAINERS = {"avi"}


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


def probe_video(path: Path) -> VideoMetadata:
    res = probe(path)
    meta = extract_metadata(res)
    if meta.container.split(",")[0].strip() in _OVERCOUNTING_CONTAINERS:
        counted = _count_video_packets(path)
        if counted > 0 and counted != meta.total_frames:
            meta = meta.model_copy(update={"total_frames": counted})
    return meta
