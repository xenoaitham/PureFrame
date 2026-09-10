import logging
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HardwareProfile(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CPU = "CPU"


class ProfileSettings(BaseModel):
    profile: HardwareProfile
    detection_resolution: int  # max edge in px, downscale before detection
    detection_batch_size: int
    use_fp16: bool
    keep_models_loaded: bool  # if False, unload models after the plan finishes
    sample_keyframes_per_shot: int
    densify_every_n_frames: int  # 1 = every frame, 3 = every 3rd
    onnx_providers: list[str]  # e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # Frames skipped between analyzed frames during scene detection. Scene
    # boundaries stay real (PySceneDetect reports actual positions); only
    # their precision degrades to ±frame_skip frames, which the renderer's
    # 0.5s segment padding already covers.
    scene_frame_skip: int = 0
    # Software-encoder speed preset (libx264/libx265). ffmpeg's default
    # "medium" costs 2-3x encode time for negligible quality gains at these
    # CRFs; faster presets suit a censoring pipeline where the source is
    # re-encoded once and fidelity is bounded by the blur anyway.
    encoder_preset: str = "medium"
    # CUDA device index for the ML models (0-based). Pinning a device makes
    # ONNX Runtime run the CUDA EP on that GPU (provider options) and puts
    # the torch models on cuda:<n>. Verdicts don't depend on it, so it is
    # deliberately excluded from the checkpoint config hash.
    cuda_device: int = 0


def onnx_providers_for(base: list[str], cuda_device: int) -> list:
    """Pin the CUDA EP to *cuda_device* when the profile has it.

    onnxruntime picks GPU 0 unless told otherwise; ``{"device_id": n}`` is
    the provider-option form NudeDetector passes through to InferenceSession.
    """
    if cuda_device == 0 or "CUDAExecutionProvider" not in base:
        return base
    pinned: list = []
    for provider in base:
        if provider == "CUDAExecutionProvider":
            pinned.append(("CUDAExecutionProvider", {"device_id": str(cuda_device)}))
        else:
            pinned.append(provider)
    return pinned


def torch_device_str(cuda_device: int) -> str:
    """torch device string for the models, honoring the pinned CUDA device."""
    try:
        import torch

        if torch.cuda.is_available() and cuda_device < torch.cuda.device_count():
            return f"cuda:{cuda_device}"
    except Exception:
        pass
    return "cpu"


def detect_profile(cuda_device: int = 0) -> HardwareProfile:
    try:
        import torch

        if not torch.cuda.is_available():
            return HardwareProfile.CPU
        # Only guard against out-of-range indexes when the runtime gives us
        # a real count (callers may mock torch partially).
        try:
            device_count = torch.cuda.device_count()
        except Exception:
            device_count = None
        if isinstance(device_count, int) and cuda_device >= device_count:
            logger.warning(
                "CUDA device %s does not exist (%s GPU(s) found) — profiling device 0.",
                cuda_device,
                device_count,
            )
            cuda_device = 0
        # Get free VRAM in GB on the pinned device
        free_vram = torch.cuda.mem_get_info(cuda_device)[0] / (1024**3)
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
        logger.warning(
            f"Error detecting hardware profile: {e}, falling back to CPU profile."
        )
        return HardwareProfile.CPU


def get_settings(profile: HardwareProfile, cuda_device: int = 0) -> ProfileSettings:
    if profile == HardwareProfile.HIGH:
        return ProfileSettings(
            profile=profile,
            detection_resolution=1080,
            detection_batch_size=32,
            use_fp16=True,
            keep_models_loaded=True,
            sample_keyframes_per_shot=5,
            densify_every_n_frames=1,
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            scene_frame_skip=0,
            encoder_preset="medium",
            cuda_device=cuda_device,
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
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            scene_frame_skip=1,
            encoder_preset="faster",
            cuda_device=cuda_device,
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
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            scene_frame_skip=2,
            encoder_preset="veryfast",
            cuda_device=cuda_device,
        )
    else:  # CPU
        return ProfileSettings(
            profile=profile,
            detection_resolution=480,
            detection_batch_size=1,
            use_fp16=False,
            keep_models_loaded=False,
            sample_keyframes_per_shot=2,
            densify_every_n_frames=5,
            onnx_providers=["CPUExecutionProvider"],
            scene_frame_skip=2,
            encoder_preset="veryfast",
            cuda_device=cuda_device,
        )
