import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from pureframe.pipeline.detect.audio import AudioClassifier
from pureframe.hardware import get_settings, HardwareProfile

pytestmark = pytest.mark.slow


def test_audio_classifier_silence(tmp_path):
    settings = get_settings(HardwareProfile.CPU)
    settings.detection_resolution = 300
    classifier = AudioClassifier(settings)
    
    audio_path = tmp_path / "silence.wav"
    sf.write(str(audio_path), np.zeros(32000, dtype=np.float32), 32000)
    
    ctx = classifier.classify_segment(audio_path, 0.0, 1.0)
    assert ctx.moaning_score < 0.1
    assert ctx.sexual_audio_score < 0.1


def test_audio_classifier_tone(tmp_path):
    settings = get_settings(HardwareProfile.CPU)
    settings.detection_resolution = 300
    classifier = AudioClassifier(settings)
    
    audio_path = tmp_path / "tone.wav"
    t = np.linspace(0, 1, 32000, False)
    tone = np.sin(1000 * 2 * np.pi * t).astype(np.float32)
    sf.write(str(audio_path), tone, 32000)
    
    ctx = classifier.classify_segment(audio_path, 0.0, 1.0)
    assert ctx.moaning_score < 0.1
    assert ctx.sexual_audio_score < 0.1
