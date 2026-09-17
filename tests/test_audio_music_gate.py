"""Audio gating: music masking and marginal-score benefit of the doubt.

Unit tests pin the math on synthetic spectra (a pure tone is flatness
zero, white noise is flatness one) and the gate's decision matrix. The
end-to-end pair runs the real spectral pre-pass over two clips that
differ only in their audio - a loud sine tone must mask a marginal
sexual-audio score that white noise lets through - with the CLIP scene
and PANNs classifier faked, so the spectral math under test is real.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pureframe.pipeline.detect.audio import AudioContext
from pureframe.pipeline.detect.audio_gate import (
    MARGINAL_DROP_FACTOR,
    MUSIC_RAISE_FACTOR,
    audio_gate,
    flatness_of_spectrum,
    music_dominant,
    segment_audio_profile,
)


def _ctx(
    moaning=0.0,
    sexual=0.0,
    music=0.0,
    speech=0.0,
    rms_db=None,
    flatness=None,
) -> AudioContext:
    return AudioContext(
        moaning_score=moaning,
        sexual_audio_score=sexual,
        music_score=music,
        speech_score=speech,
        rms_db=rms_db,
        spectral_flatness=flatness,
    )


class TestSpectralMath:
    def test_pure_tone_is_tonal(self):
        rng = np.random.default_rng(11)
        samples = np.sin(np.linspace(0, 2 * np.pi * 40, 8192)).astype(np.float64)
        samples += rng.normal(0, 1e-4, 8192)  # not bit-exact: realistic tone
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(8192))) ** 2
        assert flatness_of_spectrum(spectrum) < 0.05

    def test_white_noise_is_flat(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(0, 0.5, 8192)
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(8192))) ** 2
        assert flatness_of_spectrum(spectrum) > 0.5

    def test_floor_keeps_silent_bins_from_zeroing(self):
        spectrum = np.zeros(64)
        spectrum[10] = 1.0
        assert flatness_of_spectrum(spectrum) > 0.0


class TestSegmentProfile:
    def test_tonal_file_reads_loud_and_tonal(self, tmp_path):
        tone = tmp_path / "tone.mkv"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:a",
                "aac",
                str(tone),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        profile = segment_audio_profile(tone, 0.0, 2.0)
        assert profile is not None
        rms_db, flatness = profile
        assert rms_db > -30.0  # loud
        assert flatness < 0.15  # tonal

    def test_silent_file_reads_quiet(self, tmp_path):
        silence = tmp_path / "silence.mkv"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono:duration=2",
                "-c:a",
                "aac",
                str(silence),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        profile = segment_audio_profile(silence, 0.0, 2.0)
        assert profile is not None
        assert profile[0] < -30.0  # quiet: never music-dominant


class TestGateMatrix:
    BARS = (0.35, 0.30)

    def test_loud_tonal_audio_raises_both_bars(self):
        ctx = _ctx(moaning=0.9, sexual=0.9, rms_db=-6.0, flatness=0.03)
        moan_mult, sex_mult = audio_gate(ctx, False, *self.BARS)
        assert moan_mult == MUSIC_RAISE_FACTOR
        assert sex_mult == MUSIC_RAISE_FACTOR

    def test_quiet_tonal_audio_is_not_music_dominant(self):
        ctx = _ctx(moaning=0.0, sexual=0.0, rms_db=-45.0, flatness=0.03)
        assert music_dominant(ctx) is False

    def test_loud_noise_is_not_music_dominant(self):
        ctx = _ctx(moaning=0.0, sexual=0.0, rms_db=-6.0, flatness=0.8)
        assert music_dominant(ctx) is False

    def test_missing_profile_never_raises(self):
        ctx = _ctx(moaning=0.0, sexual=0.0)
        assert audio_gate(ctx, False, *self.BARS) == (1.0, 1.0)

    def test_marginal_score_in_sexual_scene_drops(self):
        ctx = _ctx(moaning=0.28, sexual=0.0)  # 0.28 in [0.175, 0.35)
        moan_mult, sex_mult = audio_gate(ctx, True, *self.BARS)
        assert moan_mult == MARGINAL_DROP_FACTOR
        assert sex_mult == MARGINAL_DROP_FACTOR

    def test_marginal_score_without_sexual_scene_stays(self):
        ctx = _ctx(moaning=0.28, sexual=0.0)
        assert audio_gate(ctx, False, *self.BARS) == (1.0, 1.0)

    def test_score_over_the_bar_is_not_marginal(self):
        ctx = _ctx(moaning=0.5, sexual=0.0)
        assert audio_gate(ctx, True, *self.BARS) == (1.0, 1.0)

    def test_music_masking_wins_over_the_drop(self):
        # Marginal score AND loud tonal music: raise, never lower.
        ctx = _ctx(moaning=0.28, sexual=0.28, rms_db=-6.0, flatness=0.03)
        moan_mult, sex_mult = audio_gate(ctx, True, *self.BARS)
        assert moan_mult == MUSIC_RAISE_FACTOR
        assert sex_mult == MUSIC_RAISE_FACTOR


def _build_clip(path: Path, audio_filter: str) -> Path:
    """6 s grey clip whose only difference is the audio filter."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0xC0C0C0:size=320x240:rate=15:duration=6",
            "-f",
            "lavfi",
            "-i",
            audio_filter,
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


class _SexualScene:
    def __init__(self, settings):
        self.enabled = True

    def classify_shot(self, frame):
        from pureframe.pipeline.detect.scene_clip import ShotContext

        return ShotContext(
            explicit_act_score=0.7,
            implied_sex_score=0.1,
            kissing_score=0.0,
            safe_score=0.05,
        )

    def unload(self):
        pass


class _MarginalAudio:
    """PANNs stand-in: scores just under the medium bars (0.35 / 0.30)."""

    def __init__(self, settings, enabled=True):
        self.enabled = enabled

    def classify_segment(self, path, start_sec, end_sec):
        return _ctx(moaning=0.30, sexual=0.34)

    def unload(self):
        pass


class TestEndToEnd:
    @pytest.fixture(scope="session")
    def tone_clip(self, tmp_path_factory):
        return _build_clip(
            tmp_path_factory.mktemp("gate") / "tone.mp4",
            "sine=frequency=440:duration=6",
        )

    @pytest.fixture(scope="session")
    def noise_clip(self, tmp_path_factory):
        return _build_clip(
            tmp_path_factory.mktemp("gate") / "noise.mp4",
            "anoisesrc=color=white:duration=6",
        )

    def _plan(self, video, tmp_path, monkeypatch):
        import pureframe.cli
        from pureframe.config import Config
        from pureframe.hardware import HardwareProfile
        from pureframe.pipeline.shots import Shot

        monkeypatch.setattr(pureframe.cli, "SceneClassifier", _SexualScene)
        monkeypatch.setattr(pureframe.cli, "AudioClassifier", _MarginalAudio)
        monkeypatch.setattr(
            pureframe.cli,
            "detect_shots",
            lambda *a, **k: [
                Shot(
                    index=0,
                    start_frame=0,
                    end_frame=90,
                    start_time=0.0,
                    end_time=6.0,
                )
            ],
        )
        config = Config.from_cli(
            input_path=video,
            output_path=tmp_path / "out.mp4",
            profile=HardwareProfile.CPU,
            no_clip=False,
            no_audio=False,
        )
        return pureframe.cli.generate_plan(config)

    def test_loud_music_masks_marginal_audio(self, tone_clip, tmp_path, monkeypatch):
        plan = self._plan(tone_clip, tmp_path, monkeypatch)
        assert all(v.action.name == "NONE" for v in plan.verdicts), (
            "a loud pure tone must raise the audio bars over the marginal scores"
        )

    def test_noise_does_not_mask(self, noise_clip, tmp_path, monkeypatch):
        plan = self._plan(noise_clip, tmp_path, monkeypatch)
        flagged = [v for v in plan.verdicts if v.action.name != "NONE"]
        assert flagged, "noise is not music: the marginal score flags normally"
        assert flagged[0].category.name == "SEXUAL_ACT_VISIBLE"
