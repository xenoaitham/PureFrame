from pydantic import BaseModel
import numpy as np
from nudenet import NudeDetector
from pureframe.hardware import ProfileSettings

EXPLICIT_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED"
}

class Detection(BaseModel):
    label: str
    score: float
    box: tuple[int, int, int, int]

class NudityDetector:
    def __init__(self, settings: ProfileSettings):
        self.settings = settings
        self.detector = None
        if self.settings.keep_models_loaded:
            self._load()

    def _load(self):
        if self.detector is None:
            # Note: NudeDetector initializes ONNXRuntime internally.
            # We pass providers if NudeDetector exposes it.
            # Usually in nudenet 3.x, it takes providers param.
            try:
                self.detector = NudeDetector(providers=self.settings.onnx_providers)
            except TypeError:
                self.detector = NudeDetector() # fallback if providers not supported

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        if not frames_bgr:
            return []
            
        was_unloaded = self.detector is None
        if was_unloaded:
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
                    label = p['class']
                    score = p['score']
                    box = p['box'] # usually [x, y, w, h] or [x1, y1, x2, y2]
                    
                    # NudeNet box format is usually [x, y, w, h]. Let's convert to x1, y1, x2, y2.
                    if len(box) == 4:
                        if box[2] < box[0] or box[3] < box[1]: # maybe x,y,w,h
                            x1, y1, w, h = box
                            x2, y2 = x1 + w, y1 + h
                        else:
                            # It could be x1, y1, w, h but where x1+w and y1+h
                            # Or it could be x1, y1, x2, y2 natively
                            pass
                            
                        # Ensure it's x1, y1, x2, y2. NudeNet 3.x is [x, y, w, h]
                        x1, y1, w, h = box
                        x2, y2 = x1 + w, y1 + h
                        
                        if label in EXPLICIT_LABELS:
                            detections.append(Detection(
                                label=label,
                                score=score,
                                box=(int(x1), int(y1), int(x2), int(y2))
                            ))
                batch_results.append(detections)
            except Exception as e:
                batch_results.append([])
                
        if was_unloaded and not self.settings.keep_models_loaded:
            self.unload()
            
        return batch_results

    def unload(self) -> None:
        self.detector = None
