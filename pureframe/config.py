import hashlib
import json
from enum import Enum
from pathlib import Path

from pydantic import field_validator
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

# Per-category threshold names in (nudity, clip, audio) order — the order
# get_effective_thresholds() returns and the keys a --thresholds file uses.
THRESHOLD_CATEGORIES = ("nudity", "clip", "audio")


def load_thresholds_file(path: Path) -> dict[str, float]:
    """Parse a ``--thresholds`` JSON file: ``{"nudity": 0.4, "clip": 0.5}``.

    Any subset of :data:`THRESHOLD_CATEGORIES` is allowed; keys and ranges
    are validated when the dict lands in :class:`Config`.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of category → threshold")
    overrides: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: threshold for {key!r} must be a number")
        overrides[str(key)] = float(value)
    return overrides


class Config(BaseSettings):
    input_path: Path
    output_path: Path | None = None
    profile: HardwareProfile | None = None
    nudity_threshold: float = 0.55
    clip_threshold: float = 0.50
    audio_threshold: float = 0.60
    # Explicit per-category base thresholds (--threshold-nudity/-clip/-audio
    # or a --thresholds file). A set category replaces the strictness
    # preset's value for that category; the others keep the preset. The
    # content-type multiplier and --strict still apply on top, exactly as
    # they do for the presets.
    threshold_overrides: dict[str, float] = {}
    box_padding_pct: float = 0.12
    box_color: tuple[int, int, int] = (0, 0, 0)
    blur_mode: BlurMode = BlurMode.BLUR
    blur_kernel: int = 51  # odd; bigger = more blur
    blur_sigma: float = 25.0
    pixelate_blocks: int = 16  # number of mosaic blocks across the box's long edge
    # Applies to H.264 sources; other codecs keep their own so censored
    # segments concat with the stream-copied ones and fit the input container.
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

    # Speed offensive: int8-quantize the NudeNet model on CPU-only profiles
    # (2-4x CPU inference; accuracy gated by the eval-parity CI job).
    quantize_cpu: bool = True

    # Checkpoint cache: SHA-256 of the input's bytes, filled in by from_cli.
    # Folds into config_hash so a replaced file is a cache miss; direct
    # constructions (tests, programmatic use) leave it empty, which matches
    # the pre-fingerprint checkpoint keys.
    content_fingerprint: str = ""
    # --no-cache escape: bypass cached jobs and verdicts entirely. Excluded
    # from config_hash on purpose — the flag changes cache *reads*, not the
    # detection configuration it describes.
    no_cache: bool = False
    cache_salt: str = ""
    # CUDA device index for the ML models (--device, 0-based). Excluded from
    # config_hash: it is a performance knob, not a detection decision — a
    # cached verdict from GPU 0 is valid on GPU 1.
    device: int | None = None

    model_config = SettingsConfigDict(env_prefix="PUREFRAME_")

    @field_validator("threshold_overrides")
    @classmethod
    def _validate_threshold_overrides(cls, value: dict[str, float]) -> dict:
        for key, threshold in value.items():
            if key not in THRESHOLD_CATEGORIES:
                raise ValueError(
                    f"unknown threshold category {key!r}; "
                    f"expected one of {', '.join(THRESHOLD_CATEGORIES)}"
                )
            if not 0.0 < threshold <= 1.0:
                raise ValueError(
                    f"threshold for {key!r} must be in (0, 1], got {threshold}"
                )
        return value

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
        if not config.content_fingerprint:
            from pureframe.checkpoint import content_fingerprint

            config.content_fingerprint = content_fingerprint(config.input_path)
        return config

    def get_effective_thresholds(self) -> tuple[float, float, float]:
        """Return (nudity, clip, audio) thresholds adjusted for content type and strictness.

        Base values come from the strictness preset (or the three
        ``*_threshold`` fields under ``custom``), with any
        ``threshold_overrides`` replacing their category; the content-type
        multiplier then scales all three, capped at 0.99.
        """
        if self.strictness == Strictness.CUSTOM:
            bases = [self.nudity_threshold, self.clip_threshold, self.audio_threshold]
        else:
            bases = list(STRICTNESS_PRESETS[self.strictness])

        for i, category in enumerate(THRESHOLD_CATEGORIES):
            if category in self.threshold_overrides:
                bases[i] = self.threshold_overrides[category]

        # Apply content-type multiplier
        mult = CONTENT_TYPE_MULTIPLIERS[self.content_type]
        nudity, clip, audio = (min(base * mult, 0.99) for base in bases)
        return (nudity, clip, audio)

    @property
    def config_hash(self) -> str:
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
            "quantize_cpu": self.quantize_cpu,
        }
        # Only when set: empty fingerprints (direct constructions,
        # pre-fingerprint checkpoints) and empty override maps must keep
        # hashing to the same value as before. no_cache and cache_salt are
        # deliberately excluded — they bypass cache reads and don't describe
        # the configuration.
        if self.threshold_overrides:
            data["threshold_overrides"] = dict(sorted(self.threshold_overrides.items()))
        if self.content_fingerprint:
            data["content_fingerprint"] = self.content_fingerprint
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
