from pureframe.pipeline.probe import probe_video

def test_probe(synthetic_video):
    meta = probe_video(synthetic_video)
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.fps == 24
    assert meta.has_audio is True
    # duration should be around 15.0
    assert 14.5 <= meta.duration_seconds <= 15.5
    assert meta.total_frames == 15 * 24
