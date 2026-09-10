import io
import logging
import subprocess
from pathlib import Path

import librosa
import numpy as np
from pydantic import BaseModel

from pureframe.hardware import ProfileSettings

logger = logging.getLogger(__name__)


class AudioContext(BaseModel):
    moaning_score: float
    sexual_audio_score: float
    music_score: float
    speech_score: float


# PANNs inference cost scales linearly with segment length; a long shot's
# audio character is conveyed by its middle, so analysis is capped there.
AUDIO_ANALYSIS_WINDOW_SECONDS = 10.0


def analysis_window(
    start_sec: float,
    end_sec: float,
    window: float = AUDIO_ANALYSIS_WINDOW_SECONDS,
) -> tuple[float, float]:
    """Center-capped analysis window for a shot's audio."""
    if end_sec - start_sec <= window:
        return start_sec, end_sec
    mid = (start_sec + end_sec) / 2
    return mid - window / 2, mid + window / 2


class AudioClassifier:
    # Fallback AudioSet-527 indices used when name resolution fails. These
    # are approximate and should be verified against the active panns_inference
    # label map at runtime.
    _DEFAULT_LABEL_IDX = {
        "speech": 0,
        "music": 137,
        "moan": 25,
        "sigh": 26,
        "pant": 45,
        "smack": 467,
    }

    # Substring patterns used to discover indices in the panns_inference
    # label list at startup. First match wins.
    _LABEL_PATTERNS = {
        "speech": ("Speech",),
        "music": ("Music",),
        "moan": ("Moan", "Groan", "Female speech"),
        "sigh": ("Sigh",),
        "pant": ("Pant", "Breath"),
        "smack": ("Smack", "Kiss"),
    }

    def __init__(self, settings: ProfileSettings, enabled: bool = True):
        self.enabled = enabled

        # When disabled we skip the heavyweight PANNs checkpoint download
        # entirely. panns_inference.SoundEventDetection.__init__ shells out to
        # `wget` on first use, which can hang in CI when there is no network
        # or DNS resolution is slow. classify_segment short-circuits via
        # self.enabled so the model is never needed.
        if not enabled:
            self.device = "cpu"
            self.sed = None
            self.label_idx = {}
            return

        from panns_inference import SoundEventDetection

        # Respect GPU availability instead of pinning to CPU. The previous
        # implementation hardcoded ``cpu`` even on CUDA-capable machines,
        # making PANNs inference dramatically slower than needed. The pinned
        # CUDA device (--device) is honored when present.
        try:
            from pureframe.hardware import torch_device_str

            self.device = torch_device_str(settings.cuda_device)
        except Exception:
            self.device = "cpu"
        logger.info(f"Loading PANNs SoundEventDetection model on {self.device}")

        self.sed = SoundEventDetection(checkpoint_path=None, device=self.device)

        self.label_idx = self._resolve_label_indices()

    def _resolve_label_indices(self) -> dict[str, int]:
        """Resolve PANNs label indices by name with safe fallback.

        The previous implementation used hardcoded integer indices for the
        AudioSet 527-class label set. Those indices are fragile — they were
        likely copied without verification and ``smack: 467`` in particular
        does not correspond to a kiss/smack sound in the canonical AudioSet
        label order. We now look the labels up by name at startup, logging
        which patterns failed so users can override via subclass.
        """
        labels: list[str] = []
        try:
            from panns_inference import labels as panns_labels  # type: ignore

            labels = list(panns_labels)
        except Exception:
            try:
                from panns_inference.config import (
                    labels as panns_labels,  # type: ignore
                )

                labels = list(panns_labels)
            except Exception:
                labels = []

        if not labels:
            logger.warning(
                "Could not resolve PANNs label list; using legacy hardcoded indices."
            )
            return dict(self._DEFAULT_LABEL_IDX)

        resolved: dict[str, int] = {}
        for key, patterns in self._LABEL_PATTERNS.items():
            idx: int | None = None
            for pat in patterns:
                for i, lbl in enumerate(labels):
                    if pat.lower() in lbl.lower():
                        idx = i
                        break
                if idx is not None:
                    break
            if idx is None:
                idx = self._DEFAULT_LABEL_IDX[key]
                logger.warning(
                    f"PANNs label for '{key}' not found, falling back to index {idx}"
                )
            resolved[key] = idx
        return resolved

    def classify_segment(
        self, audio_path: Path, start_sec: float, end_sec: float
    ) -> AudioContext:
        if not self.enabled:
            return AudioContext(
                moaning_score=0.0,
                sexual_audio_score=0.0,
                music_score=0.0,
                speech_score=0.0,
            )

        try:
            start_sec, end_sec = analysis_window(start_sec, end_sec)
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_sec),
                "-to",
                str(end_sec),
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "32000",
                "-f",
                "wav",
                "pipe:1",
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or len(res.stdout) == 0:
                return AudioContext(
                    moaning_score=0.0,
                    sexual_audio_score=0.0,
                    music_score=0.0,
                    speech_score=0.0,
                )

            audio, _ = librosa.load(io.BytesIO(res.stdout), sr=32000)

            if len(audio) < 32000:
                audio = np.pad(audio, (0, 32000 - len(audio)))

            framewise_output = self.sed.inference(audio[None, :])
            framewise = framewise_output[0]

            max_probs = np.max(framewise, axis=0)

            moaning = float(
                max(
                    max_probs[self.label_idx["moan"]], max_probs[self.label_idx["sigh"]]
                )
            )
            panting = float(max_probs[self.label_idx["pant"]])
            smacking = float(max_probs[self.label_idx["smack"]])

            sexual_audio = moaning * 0.6 + panting * 0.2 + smacking * 0.2

            return AudioContext(
                moaning_score=moaning,
                sexual_audio_score=sexual_audio,
                music_score=float(max_probs[self.label_idx["music"]]),
                speech_score=float(max_probs[self.label_idx["speech"]]),
            )

        except Exception as e:
            logger.error(f"Audio classification failed: {e}")
            return AudioContext(
                moaning_score=0.0,
                sexual_audio_score=0.0,
                music_score=0.0,
                speech_score=0.0,
            )

    def unload(self):
        if hasattr(self, "sed"):
            del self.sed
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
