from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from .hardware import HardwareProfile

class Config(BaseSettings):
    input_path: Path
    output_path: Path | None = None
    profile: HardwareProfile | None = None
    nudity_threshold: float = 0.55
    box_padding_pct: float = 0.12
    box_color: tuple[int, int, int] = (0, 0, 0)
    output_codec: str = "h264"
    output_crf: int = 20
    log_level: str = "INFO"
    
    # Phase 2 additions
    strict: bool = False
    no_clip: bool = False
    no_audio: bool = False

    model_config = SettingsConfigDict(env_prefix="PUREFRAME_")

    @classmethod
    def from_cli(cls, **kwargs) -> "Config":
        config = cls(**kwargs)
        if not config.input_path.exists() or not config.input_path.is_file():
            raise ValueError(f"Input path does not exist or is not a file: {config.input_path}")
        if config.output_path is None:
            config.output_path = config.input_path.with_name(f"{config.input_path.stem}.pureframe{config.input_path.suffix}")
        return config

    @property
    def config_hash(self) -> str:
        import hashlib
        import json
        
        # We hash the parameters that affect the output visually or detection wise
        data = {
            "profile": getattr(self.profile, "value", str(self.profile)),
            "nudity_threshold": self.nudity_threshold,
            "box_padding_pct": self.box_padding_pct,
            "strict": self.strict,
            "no_clip": self.no_clip,
            "no_audio": self.no_audio
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
