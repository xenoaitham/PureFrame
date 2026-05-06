import os
import io
import subprocess
import numpy as np
import librosa
from pathlib import Path
from pydantic import BaseModel
from pureframe.hardware import ProfileSettings, HardwareProfile
import logging

logger = logging.getLogger(__name__)

class AudioContext(BaseModel):
    moaning_score: float
    sexual_audio_score: float
    music_score: float
    speech_score: float

class AudioClassifier:
    def __init__(self, settings: ProfileSettings):
        self.enabled = True
        
        from panns_inference import SoundEventDetection
        self.device = "cpu"
        logger.info(f"Loading PANNs SoundEventDetection model on {self.device}")
        
        self.sed = SoundEventDetection(checkpoint_path=None, device=self.device)
        
        self.label_idx = {
            "speech": 0,
            "music": 137,
            "moan": 25,
            "sigh": 26,
            "pant": 45,
            "smack": 467
        }
        
    def classify_segment(self, audio_path: Path, start_sec: float, end_sec: float) -> AudioContext:
        if not self.enabled:
            return AudioContext(moaning_score=0.0, sexual_audio_score=0.0, music_score=0.0, speech_score=0.0)
            
        try:
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_sec), '-to', str(end_sec),
                '-i', str(audio_path), '-ac', '1', '-ar', '32000', '-f', 'wav', 'pipe:1'
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or len(res.stdout) == 0:
                return AudioContext(moaning_score=0.0, sexual_audio_score=0.0, music_score=0.0, speech_score=0.0)
                
            audio, _ = librosa.load(io.BytesIO(res.stdout), sr=32000)
            
            if len(audio) < 32000:
                audio = np.pad(audio, (0, 32000 - len(audio)))
                
            framewise_output = self.sed.inference(audio[None, :])
            framewise = framewise_output[0]
            
            max_probs = np.max(framewise, axis=0)
            
            moaning = float(max(max_probs[self.label_idx["moan"]], max_probs[self.label_idx["sigh"]]))
            panting = float(max_probs[self.label_idx["pant"]])
            smacking = float(max_probs[self.label_idx["smack"]])
            
            sexual_audio = moaning * 0.6 + panting * 0.2 + smacking * 0.2
            
            return AudioContext(
                moaning_score=moaning,
                sexual_audio_score=sexual_audio,
                music_score=float(max_probs[self.label_idx["music"]]),
                speech_score=float(max_probs[self.label_idx["speech"]])
            )
            
        except Exception as e:
            logger.error(f"Audio classification failed: {e}")
            return AudioContext(moaning_score=0.0, sexual_audio_score=0.0, music_score=0.0, speech_score=0.0)

    def unload(self):
        if hasattr(self, 'sed'):
            del self.sed
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
