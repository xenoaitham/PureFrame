from pathlib import Path

from pureframe.config import Config
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.render.overlay import build_overlay_callback
from pureframe.utils.ffmpeg import (
    probe_video_codec,
    select_render_encoder,
    write_video_with_overlay,
)


def apply_censoring(
    input_path: Path,
    output_path: Path,
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
    input_codec: str | None = None,
) -> None:
    """Full re-encode renderer.

    Uses the shared overlay callback so that BLACK_BOX actions render the
    user-configured censor style (blur / pixelate / solid box) instead of
    the previous hardcoded solid rectangle. The encoder follows the source
    codec (``input_codec``, probed when omitted) so the result muxes into
    the input's own container — WebM and AVI reject a hardcoded H.264.
    """
    if input_codec is None:
        input_codec = probe_video_codec(input_path)
    encoder = select_render_encoder(
        profile_settings.profile, config.output_codec, input_codec
    )
    overlay_callback = build_overlay_callback(frame_actions, config, profile_settings)

    write_video_with_overlay(
        input_path=input_path,
        output_path=output_path,
        overlay_callback=overlay_callback,
        settings=profile_settings,
        encoder=encoder,
        crf=config.output_crf,
        preset=profile_settings.encoder_preset,
    )
