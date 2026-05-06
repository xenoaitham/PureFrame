from pathlib import Path
from pureframe.utils.ffmpeg import probe, extract_metadata, VideoMetadata


def probe_video(path: Path) -> VideoMetadata:
    res = probe(path)
    return extract_metadata(res)
