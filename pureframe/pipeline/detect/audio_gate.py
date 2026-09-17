"""Audio gating: music masking and marginal-score adjustments.

Two honest adjustments to the audio thresholds, both bounded by what
PANNs can actually hear:

- Loud tonal audio (music) masks quiet explicit cues, so when a shot's
  spectrum is dominated by loud tonal content - high RMS, low spectral
  flatness as the cheap music proxy - the audio thresholds rise. A
  raised bar can only suppress false positives on music-heavy scenes;
  it can never manufacture a flag.
- When the CLIP scene reads sexual and an audio score lands just under
  its bar, the bar drops slightly: whispered or quiet cues in a sexual
  scene deserve one notch of benefit of the doubt. The scene signal
  guards this branch - without it, scores are never lowered.

Foreign-language gaps stay exactly as they are: PANNs hears sounds, not
words, and this gate does not change that.
"""

from __future__ import annotations

import numpy as np

from pureframe.pipeline.detect.audio import AudioContext

# Loud tonal audio: at least this loud (dBFS) and this tonal (spectral
# flatness below the cap) counts as music dominating the mix.
MUSIC_RMS_FLOOR_DB = -30.0
MUSIC_FLATNESS_MAX = 0.15
MUSIC_RAISE_FACTOR = 1.25

# A sexual scene's marginal audio scores (half the bar, under it) get
# this much benefit of the doubt.
MARGINAL_BAND = 0.5
MARGINAL_DROP_FACTOR = 0.8


def flatness_of_spectrum(power_spectrum: np.ndarray) -> float:
    """Spectral flatness of one power spectrum: 0 = tonal, 1 = noise.

    Geometric mean over arithmetic mean of the power bins; a tiny floor
    keeps silent bins from zeroing the geometric mean.
    """
    spectrum = np.asarray(power_spectrum, dtype=np.float64) + 1e-12
    return float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))


def segment_audio_profile(
    path, start_sec: float, end_sec: float
) -> tuple[float, float] | None:
    """(rms_db, spectral_flatness) of one shot's audio, cheaply.

    A 16 kHz mono decode of the segment - no model - feeding the same
    features a DAW would read off a meter. None when there is nothing
    decodable, which leaves the gate neutral rather than guessing.
    """
    import ffmpeg as ffmpeg_python

    try:
        out, _ = (
            ffmpeg_python.input(str(path), ss=f"{start_sec:.3f}", to=f"{end_sec:.3f}")
            .output("pipe:", format="s16le", ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except Exception:
        return None
    if not out:
        return None
    samples = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    frame = 8192
    usable = (samples.size // frame) * frame
    if usable == 0:
        return None
    rms = float(np.sqrt(np.mean(samples**2)))
    rms_db = float(20.0 * np.log10(max(rms, 1e-9)))
    windowed = samples[:usable].reshape(-1, frame) * np.hanning(frame)
    spectra = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    flatness = float(np.mean([flatness_of_spectrum(spectrum) for spectrum in spectra]))
    return rms_db, flatness


def music_dominant(audio_ctx: AudioContext) -> bool:
    """True when loud tonal audio dominates the mix.

    Both features must be present: a missing profile (undecodable audio,
    classifier substitute) never raises a threshold.
    """
    if audio_ctx.rms_db is None or audio_ctx.spectral_flatness is None:
        return False
    return (
        audio_ctx.rms_db >= MUSIC_RMS_FLOOR_DB
        and audio_ctx.spectral_flatness <= MUSIC_FLATNESS_MAX
    )


def audio_gate(
    audio_ctx: AudioContext,
    scene_is_sexual: bool,
    moaning_thresh: float,
    sexual_thresh: float,
) -> tuple[float, float]:
    """Threshold multipliers (moaning, sexual) for this shot.

    Music dominance raises both bars; a sexual scene with a marginal
    score lowers both slightly. The two never stack - music masking
    wins, because raising a bar on loud music is the conservative
    direction while lowering one there would compound the masking.
    """
    if music_dominant(audio_ctx):
        return MUSIC_RAISE_FACTOR, MUSIC_RAISE_FACTOR
    if scene_is_sexual:
        marginal_moaning = (
            MARGINAL_BAND * moaning_thresh <= audio_ctx.moaning_score < moaning_thresh
        )
        marginal_sexual = (
            MARGINAL_BAND * sexual_thresh
            <= audio_ctx.sexual_audio_score
            < sexual_thresh
        )
        if marginal_moaning or marginal_sexual:
            return MARGINAL_DROP_FACTOR, MARGINAL_DROP_FACTOR
    return 1.0, 1.0
