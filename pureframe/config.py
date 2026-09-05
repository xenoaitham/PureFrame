from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .hardware import HardwareProfile


class ContentType(str, Enum):
    LIVE_ACTION = "live-action"
    ANIMATION = "animation"
    ANIME = "anime"
    LOW_LIGHT = "low-light"


class Strictness(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"


class BlurMode(str, Enum):
    """Render style for localized censoring boxes."""

    BLUR = "blur"
    BOX = "box"
    PIXELATE = "pixelate"


# Threshold multipliers per content type
CONTENT_TYPE_MULTIPLIERS = {
    ContentType.LIVE_ACTION: 1.0,
    ContentType.ANIMATION: 1.3,
    ContentType.ANIME: 1.4,
    ContentType.LOW_LIGHT: 0.85,
}

# Strictness presets: (nudity_threshold, clip_threshold, audio_threshold)
STRICTNESS_PRESETS = {
    Strictness.LOW: (0.75, 0.70, 0.80),
    Strictness.MEDIUM: (0.55, 0.50, 0.60),
    Strictness.HIGH: (0.35, 0.35, 0.40),
}


class Config(BaseSettings):
    input_path: Path
    output_path: Path | None = None
    profile: HardwareProfile | None = None
    nudity_threshold: float = 0.55
    clip_threshold: float = 0.50
    audio_threshold: float = 0.60
    box_padding_pct: float = 0.12
    box_color: tuple[int, int, int] = (0, 0, 0)
    blur_mode: BlurMode = BlurMode.BLUR
    blur_kernel: int = 51  # odd; bigger = more blur
    blur_sigma: float = 25.0
    pixelate_blocks: int = 16  # number of mosaic blocks across the box's long edge
    output_codec: str = "h264"
    output_crf: int = 20
    log_level: str = "INFO"

    # Phase 2 additions
    strict: bool = False
    no_clip: bool = False
    no_audio: bool = False

    # Phase 3 additions
    content_type: ContentType = ContentType.LIVE_ACTION
    strictness: Strictness = Strictness.MEDIUM
    force: bool = False

    model_config = SettingsConfigDict(env_prefix="PUREFRAME_")

    @classmethod
    def from_cli(cls, **kwargs) -> "Config":
        config = cls(**kwargs)
        if not config.input_path.exists() or not config.input_path.is_file():
            raise ValueError(
                f"Input path does not exist or is not a file: {config.input_path}"
            )
        if config.output_path is None:
            config.output_path = config.input_path.with_name(
                f"{config.input_path.stem}.pureframe{config.input_path.suffix}"
            )
        return config

    def get_effective_thresholds(self) -> tuple[float, float, float]:
        """Return (nudity, clip, audio) thresholds adjusted for content type and strictness."""
        if self.strictness == Strictness.CUSTOM:
            base_nudity = self.nudity_threshold
            base_clip = self.clip_threshold
            base_audio = self.audio_threshold
        else:
            base_nudity, base_clip, base_audio = STRICTNESS_PRESETS[self.strictness]

        # Apply content-type multiplier
        mult = CONTENT_TYPE_MULTIPLIERS[self.content_type]
        return (
            min(base_nudity * mult, 0.99),
            min(base_clip * mult, 0.99),
            min(base_audio * mult, 0.99),
        )

    @property
    def config_hash(self) -> str:
        import hashlib
        import json

        data = {
            "profile": getattr(self.profile, "value", str(self.profile)),
            "nudity_threshold": self.nudity_threshold,
            "clip_threshold": self.clip_threshold,
            "audio_threshold": self.audio_threshold,
            "box_padding_pct": self.box_padding_pct,
            "box_color": list(self.box_color),
            "blur_mode": self.blur_mode.value,
            "blur_kernel": self.blur_kernel,
            "blur_sigma": self.blur_sigma,
            "pixelate_blocks": self.pixelate_blocks,
            "output_codec": self.output_codec,
            "output_crf": self.output_crf,
            "strict": self.strict,
            "no_clip": self.no_clip,
            "no_audio": self.no_audio,
            "content_type": self.content_type.value,
            "strictness": self.strictness.value,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
