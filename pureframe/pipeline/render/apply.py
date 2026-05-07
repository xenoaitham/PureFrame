from pathlib import Path
import cv2
import numpy as np

from pureframe.config import Config
from pureframe.hardware import ProfileSettings
from pureframe.utils.ffmpeg import write_video_with_overlay, select_hw_encoder

from pureframe.pipeline.shots import Action


def apply_censoring(
    input_path: Path,
    output_path: Path,
    frame_actions: dict[int, dict],
    config: Config,
    profile_settings: ProfileSettings,
) -> None:
    encoder = select_hw_encoder(profile_settings.profile, config.output_codec)

    def overlay_callback(frame_idx: int, frame_bgr: np.ndarray) -> np.ndarray:
        frame_data = frame_actions.get(frame_idx)
        if not frame_data:
            return frame_bgr

        action = frame_data.get("action", Action.NONE)

        if action == Action.FULL_FRAME_BLUR:
            # Apply Gaussian Blur (kernel size 99)
            return cv2.GaussianBlur(frame_bgr, (99, 99), 30)

        elif action == Action.BLACK_BOX:
            boxes = frame_data.get("boxes", [])
            if not boxes:
                return frame_bgr

            h, w = frame_bgr.shape[:2]
            det_res = profile_settings.detection_resolution

            dw, dh = w, h
            if dw > dh and dw > det_res:
                dh = int(dh * (det_res / dw))
                dw = det_res
            elif dh > dw and dh > det_res:
                dw = int(dw * (det_res / dh))
                dh = det_res
            dw = dw - (dw % 2)
            dh = dh - (dh % 2)

            scale_w = w / dw if dw > 0 else 1.0
            scale_h = h / dh if dh > 0 else 1.0

            for box in boxes:
                x1, y1, x2, y2 = box
                nx1 = int(x1 * scale_w)
                ny1 = int(y1 * scale_h)
                nx2 = int(x2 * scale_w)
                ny2 = int(y2 * scale_h)
                cv2.rectangle(frame_bgr, (nx1, ny1), (nx2, ny2), config.box_color, -1)

        return frame_bgr

    write_video_with_overlay(
        input_path=input_path,
        output_path=output_path,
        overlay_callback=overlay_callback,
        settings=profile_settings,
        encoder=encoder,
        crf=config.output_crf,
    )
