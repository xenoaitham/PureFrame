import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SceneDetector:
    def __init__(self, device="cpu"):
        self.device = device
        # Use a small model by default, could be configurable via profile
        model_id = "openai/clip-vit-base-patch32"
        logger.info(f"Loading CLIP model {model_id} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.prompts = [
            "a sex scene",
            "people making out",
            "two people kissing passionately",
            "a normal conversation",
            "an action scene",
            "a person talking to the camera",
            "a landscape",
        ]

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict[str, float]:
        # Convert BGR to RGB
        frame_rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(frame_rgb)

        inputs = self.processor(
            text=self.prompts, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        return {prompt: float(prob) for prompt, prob in zip(self.prompts, probs)}
