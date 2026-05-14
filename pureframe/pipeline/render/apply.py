from pathlib import Path

from pureframe.config import Config
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.render.overlay import build_overlay_callback
from pureframe.utils.ffmpeg import select_hw_encoder, write_video_with_overlay


def apply_censoring(
    input_path: Path,
    output_path: Path,
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
) -> None:
    """Full re-encode renderer.

    Uses the shared overlay callback so that BLACK_BOX actions render the
    user-configured censor style (blur / pixelate / solid box) instead of
    the previous hardcoded solid rectangle.
    """
    encoder = select_hw_encoder(profile_settings.profile, config.output_codec)
    overlay_callback = build_overlay_callback(frame_actions, config, profile_settings)

    write_video_with_overlay(
        input_path=input_path,
        output_path=output_path,
        overlay_callback=overlay_callback,
        settings=profile_settings,
        encoder=encoder,
        crf=config.output_crf,
    )
