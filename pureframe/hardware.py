from enum import Enum
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class HardwareProfile(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CPU = "CPU"

class ProfileSettings(BaseModel):
    profile: HardwareProfile
    detection_resolution: int          # max edge in px, downscale before detection
    detection_batch_size: int
    use_fp16: bool
    keep_models_loaded: bool           # if False, unload model between stages
    sample_keyframes_per_shot: int
    densify_every_n_frames: int        # 1 = every frame, 3 = every 3rd
    onnx_providers: list[str]          # e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]

def detect_profile() -> HardwareProfile:
    try:
        import torch
        if not torch.cuda.is_available():
            return HardwareProfile.CPU
        # Get free VRAM in GB
        free_vram = torch.cuda.mem_get_info(0)[0] / (1024**3)
        if free_vram >= 11:
            return HardwareProfile.HIGH
        elif free_vram >= 6:
            return HardwareProfile.MEDIUM
        elif free_vram >= 3:
            return HardwareProfile.LOW
        else:
            return HardwareProfile.CPU
    except ImportError:
        logger.warning("Torch not found, falling back to CPU profile.")
        return HardwareProfile.CPU
    except Exception as e:
        logger.warning(f"Error detecting hardware profile: {e}, falling back to CPU profile.")
        return HardwareProfile.CPU

def get_settings(profile: HardwareProfile) -> ProfileSettings:
    if profile == HardwareProfile.HIGH:
        return ProfileSettings(
            profile=profile,
            detection_resolution=1080,
            detection_batch_size=32,
            use_fp16=True,
            keep_models_loaded=True,
            sample_keyframes_per_shot=5,
            densify_every_n_frames=1,
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    elif profile == HardwareProfile.MEDIUM:
        return ProfileSettings(
            profile=profile,
            detection_resolution=720,
            detection_batch_size=16,
            use_fp16=True,
            keep_models_loaded=True,
            sample_keyframes_per_shot=3,
            densify_every_n_frames=2,
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    elif profile == HardwareProfile.LOW:
        return ProfileSettings(
            profile=profile,
            detection_resolution=540,
            detection_batch_size=4,
            use_fp16=True,
            keep_models_loaded=False,
            sample_keyframes_per_shot=3,
            densify_every_n_frames=3,
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    else: # CPU
        return ProfileSettings(
            profile=profile,
            detection_resolution=480,
            detection_batch_size=1,
            use_fp16=False,
            keep_models_loaded=False,
            sample_keyframes_per_shot=2,
            densify_every_n_frames=5,
            onnx_providers=["CPUExecutionProvider"]
        )
