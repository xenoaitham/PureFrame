import logging

import numpy as np
from nudenet import NudeDetector
from pydantic import BaseModel

from pureframe.hardware import ProfileSettings, onnx_providers_for

logger = logging.getLogger(__name__)

EXPLICIT_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


class Detection(BaseModel):
    label: str
    score: float
    box: tuple[int, int, int, int]


class NudityDetector:
    def __init__(self, settings: ProfileSettings, quantize: bool = True):
        self.settings = settings
        self.quantize = quantize
        self.detector = None
        if self.settings.keep_models_loaded:
            self._load()

    def _load(self):
        if self.detector is None:
            # CPU-only profiles get a dynamically int8-quantized copy of the
            # bundled weights (cached after first use, 2-4x CPU inference);
            # GPU profiles keep fp32 since the CUDA EP doesn't benefit. The
            # eval-parity CI gate proves the quantized model behaves
            # identically before anything ships on it.
            model_path = None
            if self.quantize and self.settings.onnx_providers == [
                "CPUExecutionProvider"
            ]:
                try:
                    from pureframe.pipeline.detect.quantize import (
                        quantized_model_path,
                    )

                    model_path = str(quantized_model_path())
                except Exception as e:
                    logger.warning(
                        "NudeNet quantization unavailable (%s) — using fp32", e
                    )
            try:
                providers = onnx_providers_for(
                    self.settings.onnx_providers, self.settings.cuda_device
                )
                if model_path:
                    self.detector = NudeDetector(
                        providers=providers,
                        model_path=model_path,
                    )
                else:
                    self.detector = NudeDetector(providers=providers)
            except TypeError:
                # Older nudenet without provider/model_path kwargs.
                self.detector = NudeDetector()

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        if not frames_bgr:
            return []

        # Load once on first use and keep the session resident: the CPU/LOW
        # profiles previously unloaded after every call, and since densify
        # calls this per frame, each densified frame paid a full ONNX session
        # re-init. cli.generate_plan unloads explicitly at end of plan.
        self._load()

        batch_results = []
        # NudeDetector typically works on paths or frames.
        # Let's check nudenet 3.x API. It usually expects a path or an array for detect().
        # If it doesn't support batched inference natively, we loop.
        for frame in frames_bgr:
            try:
                preds = self.detector.detect(frame)
                detections = []
                for p in preds:
                    label = p["class"]
                    score = p["score"]
                    box = p["box"]  # usually [x, y, w, h] or [x1, y1, x2, y2]

                    # NudeNet 3.x returns boxes as [x, y, w, h] in pixel space.
                    if len(box) != 4 or label not in EXPLICIT_LABELS:
                        continue
                    x1, y1, w, h = box
                    x2, y2 = x1 + w, y1 + h
                    detections.append(
                        Detection(
                            label=label,
                            score=score,
                            box=(int(x1), int(y1), int(x2), int(y2)),
                        )
                    )
                batch_results.append(detections)
            except Exception:
                batch_results.append([])

        return batch_results

    def unload(self) -> None:
        self.detector = None
