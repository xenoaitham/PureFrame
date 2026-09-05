from pathlib import Path

from pureframe.utils.ffmpeg import VideoMetadata, extract_metadata, probe


def probe_video(path: Path) -> VideoMetadata:
    res = probe(path)
    return extract_metadata(res)
