"""Inference cache: verdicts are keyed on (content hash, config hash).

The checkpoint store keyed jobs on (path, config) only — replace the file at
that path and every later run silently reused the stale verdicts. Jobs now
also fold a streaming SHA-256 of the file's bytes into the key, so:

* same file + same config  → cached verdicts, no model calls;
* changed file             → cache miss;
* changed config           → cache miss;
* ``--no-cache``           → salted key, never reads or poisons the cache.

Direct ``Config(...)`` constructions (tests, programmatic use) leave the
fingerprint empty and keep matching pre-fingerprint checkpoint keys.
"""

import subprocess
from pathlib import Path

from pureframe.checkpoint import content_fingerprint
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.shots import Action
from tests.conftest import frame_has_marker, generate_three_shot_clip

BOX = (60, 40, 260, 200)


def _mock_detector(monkeypatch) -> list[int]:
    """NudityDetector.detect_batch with a call counter; returns the counter."""
    from pureframe.pipeline.detect.nudity import Detection, NudityDetector

    calls = [0]

    def mocked_detect_batch(self, frames_bgr):
        calls[0] += 1
        return [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.99, box=BOX)]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    monkeypatch.setattr(NudityDetector, "detect_batch", mocked_detect_batch)
    return calls


def _dense_sampling(monkeypatch):
    import pureframe.cli

    original = pureframe.cli.get_settings

    def dense(profile, **kwargs):
        s = original(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense)


def _config(clip: Path, out: Path, **kwargs) -> Config:
    return Config.from_cli(
        input_path=clip,
        output_path=out,
        profile=HardwareProfile.CPU,
        no_clip=True,
        no_audio=True,
        **kwargs,
    )


def _flagged_shot_indices(plan) -> list[int]:
    return [v.shot_index for v in plan.verdicts if v.action != Action.NONE]


class TestContentFingerprint:
    def test_known_bytes(self, tmp_path):
        import hashlib

        f = tmp_path / "x.bin"
        f.write_bytes(b"pureframe")
        assert content_fingerprint(f) == hashlib.sha256(b"pureframe").hexdigest()

    def test_multichunk_file(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(b"a" * (2 * (1 << 20) + 17))
        import hashlib

        assert content_fingerprint(f) == hashlib.sha256(f.read_bytes()).hexdigest()

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"one")
        b.write_bytes(b"two")
        assert content_fingerprint(a) != content_fingerprint(b)


def test_same_file_and_config_hit_the_cache(three_shot_video, tmp_path, monkeypatch):
    import pureframe.cli

    calls = _mock_detector(monkeypatch)
    _dense_sampling(monkeypatch)
    monkeypatch.setattr(
        pureframe.cli, "execute_render", lambda p, c, smart=True, timers=None: None
    )

    cfg = _config(three_shot_video, tmp_path / "out.mp4")
    pureframe.cli.process_file(cfg)
    first_calls = calls[0]
    assert first_calls > 0

    # A second plan for the same content+config (fresh output path so the
    # DONE-skip in process_file can't be what saves us) must not call the
    # model at all, and must carry the same verdicts.
    plan1 = pureframe.cli.generate_plan(cfg)
    verdicts1 = [(v.shot_index, v.action) for v in plan1.verdicts]

    calls[0] = 0
    plan2 = pureframe.cli.generate_plan(
        _config(three_shot_video, tmp_path / "out2.mp4")
    )

    assert calls[0] == 0, "model ran although the checkpoint had every verdict"
    assert [(v.shot_index, v.action) for v in plan2.verdicts] == verdicts1
    assert _flagged_shot_indices(plan2) == [1]


def test_changed_file_is_a_cache_miss(three_shot_video, tmp_path, monkeypatch):
    import pureframe.cli

    calls = _mock_detector(monkeypatch)
    _dense_sampling(monkeypatch)
    monkeypatch.setattr(
        pureframe.cli, "execute_render", lambda p, c, smart=True, timers=None: None
    )

    clip = tmp_path / "clip.mp4"
    generate_three_shot_clip(clip, ["-c:v", "libx264", "-crf", "28"])

    cfg = _config(clip, tmp_path / "out.mp4")
    pureframe.cli.process_file(cfg)
    assert calls[0] > 0

    # Same path, different bytes: re-mux with a comment atom — valid MP4,
    # changed SHA-256.
    bumped = tmp_path / "bumped.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-c",
            "copy",
            "-metadata",
            "comment=pureframe-cache-bust",
            str(bumped),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    bumped.replace(clip)
    assert content_fingerprint(clip) != cfg.content_fingerprint

    calls[0] = 0
    plan = pureframe.cli.generate_plan(_config(clip, tmp_path / "out3.mp4"))
    assert calls[0] > 0, "stale verdicts were reused after the file changed"
    assert _flagged_shot_indices(plan) == [1]


def test_changed_config_is_a_cache_miss(three_shot_video, tmp_path, monkeypatch):
    import pureframe.cli

    calls = _mock_detector(monkeypatch)
    _dense_sampling(monkeypatch)
    monkeypatch.setattr(
        pureframe.cli, "execute_render", lambda p, c, smart=True, timers=None: None
    )

    cfg = _config(three_shot_video, tmp_path / "out.mp4")
    pureframe.cli.process_file(cfg)
    assert calls[0] > 0

    calls[0] = 0
    pureframe.cli.generate_plan(
        _config(
            three_shot_video, tmp_path / "out4.mp4", threshold_overrides={"nudity": 0.6}
        )
    )
    assert calls[0] > 0, "a config change must not reuse the old job's verdicts"


def test_no_cache_bypasses_and_does_not_poison(three_shot_video, tmp_path, monkeypatch):
    import pureframe.cli
    from pureframe.checkpoint import CheckpointStore

    calls = _mock_detector(monkeypatch)
    _dense_sampling(monkeypatch)
    monkeypatch.setattr(
        pureframe.cli, "execute_render", lambda p, c, smart=True, timers=None: None
    )

    cfg = _config(three_shot_video, tmp_path / "out.mp4")
    pureframe.cli.process_file(cfg)
    assert calls[0] > 0

    # --no-cache run: model runs again despite a fully cached job…
    calls[0] = 0
    bypass = _config(
        three_shot_video, tmp_path / "out5.mp4", no_cache=True, cache_salt="s1"
    )
    assert bypass.config_hash == cfg.config_hash  # the flag is not a config change
    pureframe.cli.generate_plan(bypass)
    assert calls[0] > 0

    # …and the cache is intact: a normal run afterwards still hits it.
    calls[0] = 0
    pureframe.cli.generate_plan(_config(three_shot_video, tmp_path / "out6.mp4"))
    assert calls[0] == 0

    # Distinct salts mean distinct jobs: two bypass runs never share state.
    store: CheckpointStore = pureframe.cli.get_store()
    jobs_for_input = [
        j
        for j in store.list_unfinished() + _all_jobs(store)
        if j.input_path == str(three_shot_video.absolute())
    ]
    salted = [j for j in jobs_for_input if "+nocache-" in j.config_hash]
    assert len(salted) >= 1
    assert all(j.config_hash.count("+nocache-") == 1 for j in salted)


def _all_jobs(store):
    cursor = store.conn.execute("SELECT * FROM jobs")
    from pureframe.checkpoint import Job

    return [Job(**dict(r)) for r in cursor]


def test_fingerprint_folds_into_hash_only_when_set(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00" * 32)

    plain = Config(input_path=video)
    fingerprinted = Config(input_path=video, content_fingerprint="ab" * 32)
    assert plain.config_hash != fingerprinted.config_hash

    # no_cache / cache_salt never change the identity of a configuration.
    a = Config(input_path=video, no_cache=True, cache_salt="x")
    b = Config(input_path=video, no_cache=False, cache_salt="")
    assert a.config_hash == b.config_hash == plain.config_hash
