"""Lazy-context tests: the audio classifier must run only when the CLIP
scene signal is at/above the thresholds that make its score matter.

fuse() consults audio only in the two sexual-act branches, both of which
require the scene signal to clear its own threshold first - so below them,
skipping the expensive per-shot PANNs run is provably verdict-neutral.
"""

import tempfile
from pathlib import Path

from pureframe.config import Config
from pureframe.pipeline.detect.audio import AudioContext
from pureframe.pipeline.detect.scene_clip import ShotContext
from pureframe.pipeline.fuse import context_audio_needed, fuse


def _cfg(**kwargs) -> Config:
    # from_cli validates that the input exists - provide throwaway files.
    tmp = tempfile.mkdtemp(prefix="audio_gate_")
    input_path = Path(tmp) / "in.mp4"
    input_path.write_bytes(b"fake")
    return Config.from_cli(
        input_path=input_path, output_path=Path(tmp) / "out.mp4", **kwargs
    )


def _scene(explicit_act=0.0, implied_sex=0.0, kissing=0.0, safe=0.0) -> ShotContext:
    return ShotContext(
        explicit_act_score=explicit_act,
        implied_sex_score=implied_sex,
        kissing_score=kissing,
        safe_score=safe,
    )


def test_audio_not_needed_for_neutral_scene():
    assert context_audio_needed(_scene(), _cfg()) is False


def test_audio_needed_when_scene_explicit_act_clears_threshold():
    # explicit_act threshold = 0.40 at defaults (live-action, medium)
    assert context_audio_needed(_scene(explicit_act=0.41), _cfg()) is True


def test_audio_needed_when_implied_sex_clears_threshold():
    # implied_sex threshold = 0.45 at defaults
    assert context_audio_needed(_scene(implied_sex=0.46), _cfg()) is True


def test_audio_not_needed_below_thresholds():
    assert (
        context_audio_needed(_scene(explicit_act=0.39, implied_sex=0.44), _cfg())
        is False
    )


def test_strict_mode_raises_the_bar():
    # 0.85 multiplier: 0.40 -> 0.34; a score between the two needs audio at
    # default strictness (0.41 >= 0.40) but NOT at strict (0.41 < 0.34)?
    # Careful: strict LOWERS the threshold (multiplier < 1), widening the
    # audio-needed band. A score that fails the normal threshold still fails
    # strict; a score between strict and normal thresholds gains audio need.
    cfg = _cfg()
    assert (
        context_audio_needed(_scene(explicit_act=0.35), cfg, strict_mode=False) is False
    )
    assert (
        context_audio_needed(_scene(explicit_act=0.35), cfg, strict_mode=True) is True
    )


def test_content_type_multiplier_applies():
    # Anime multiplies thresholds 1.4x: explicit_act thresh = 0.40 * 1.4 = 0.56
    cfg = _cfg(content_type="anime")
    assert context_audio_needed(_scene(explicit_act=0.50), cfg) is False
    assert context_audio_needed(_scene(explicit_act=0.60), cfg) is True


def test_fuse_neutral_audio_matches_real_audio_for_below_threshold_scene():
    """Behavioral parity: with the scene signal below thresholds, a neutral
    AudioContext and a loud one produce the identical verdict."""
    shot = _make_shot()
    dets = [[]]
    scene = _scene(explicit_act=0.2, kissing=0.1, safe=0.2)
    loud = AudioContext(
        moaning_score=0.99, sexual_audio_score=0.99, music_score=0.0, speech_score=0.0
    )
    silent = AudioContext(
        moaning_score=0.0, sexual_audio_score=0.0, music_score=0.0, speech_score=0.0
    )
    cfg = _cfg()

    v_loud = fuse(shot, dets, scene, loud, cfg)
    v_silent = fuse(shot, dets, scene, silent, cfg)

    assert v_loud.action == v_silent.action
    assert v_loud.category == v_silent.category


def _make_shot():
    from pureframe.pipeline.shots import Shot

    return Shot(
        index=0,
        start_frame=0,
        end_frame=48,
        start_time=0.0,
        end_time=2.0,
    )
