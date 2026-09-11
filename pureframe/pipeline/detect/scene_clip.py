import logging

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

from pureframe.hardware import HardwareProfile, ProfileSettings

logger = logging.getLogger(__name__)


def _features_to_tensor(features):
    """Return the projected feature tensor from a ``get_*_features`` call.

    transformers 4.x returns a bare tensor; 5.x returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` has been replaced
    with the projected features. Supporting both keeps the pyproject floor
    (>=4.30) honest without pinning.
    """
    pooled = getattr(features, "pooler_output", None)
    return pooled if pooled is not None else features


class ShotContext(BaseModel):
    explicit_act_score: float
    implied_sex_score: float
    kissing_score: float
    safe_score: float


PROMPT_SETS = {
    "explicit_act": [
        "a sex scene",
        "two people having sex",
        "naked people in bed",
        "intimate sexual contact",
    ],
    "implied_sex_context": [
        "a closeup of a woman moaning during sex",
        "a sex scene with bodies under a blanket",
        "intimate moaning expression",
        "a person in throes of sex",
    ],
    "kissing": [
        "two people kissing passionately",
        "an open mouth kiss",
        "making out on a bed",
    ],
    "safe_neutral": [
        "people having a normal conversation",
        "a person walking outdoors",
        "a meeting in an office",
        "people eating dinner",
        "a fight scene",
    ],
}


class SceneClassifier:
    def __init__(self, settings: ProfileSettings):
        self.enabled = True
        if settings.profile == HardwareProfile.CPU:
            self.enabled = False
            return

        model_name = "openai/clip-vit-large-patch14"
        if settings.profile == HardwareProfile.LOW:
            model_name = "openai/clip-vit-base-patch32"

        from pureframe.hardware import torch_device_str

        self.device = torch_device_str(settings.cuda_device)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        if getattr(settings, "use_fp16", False):
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()

        # Precompute text embeddings
        self.text_inputs = []
        self.category_names = []
        for cat, prompts in PROMPT_SETS.items():
            for p in prompts:
                self.text_inputs.append(p)
                self.category_names.append(cat)

        inputs = self.processor(
            text=self.text_inputs, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            # Public API instead of poking at model internals -
            # text_model()/vision_model() output layouts have shifted across
            # transformers releases; get_*_features() is the stable contract
            # and already applies the projection heads.
            self.text_embeds = _features_to_tensor(
                self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            )
            self.text_embeds = self.text_embeds / self.text_embeds.norm(
                p=2, dim=-1, keepdim=True
            )

    def classify_shot(self, representative_frame_bgr: np.ndarray) -> ShotContext:
        if not self.enabled:
            return ShotContext(
                explicit_act_score=0.0,
                implied_sex_score=0.0,
                kissing_score=0.0,
                safe_score=1.0,
            )

        image = Image.fromarray(representative_frame_bgr[:, :, ::-1])  # BGR to RGB
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        if next(self.model.parameters()).dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            image_embeds = _features_to_tensor(
                self.model.get_image_features(pixel_values=inputs["pixel_values"])
            )
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            # Cosine similarity per prompt (both sides L2-normalized).
            # The previous implementation softmax'd over ALL prompts together
            # and summed per category, which made categories with more prompts
            # mechanically score higher. We now compute INDEPENDENT per-category
            # scores by taking the maximum cosine similarity over each
            # category's prompt set and mapping it to [0, 1].
            cosine = torch.matmul(image_embeds, self.text_embeds.t()).cpu().numpy()[0]

        scores: dict[str, float] = {cat: 0.0 for cat in PROMPT_SETS.keys()}
        for sim, cat in zip(cosine, self.category_names):
            # Map cosine similarity from [-1, 1] to [0, 1]; keep the max
            # match per category so a single strong prompt drives the score.
            mapped = float(max(0.0, min(1.0, (sim + 1.0) * 0.5)))
            if mapped > scores[cat]:
                scores[cat] = mapped

        return ShotContext(
            explicit_act_score=scores["explicit_act"],
            implied_sex_score=scores["implied_sex_context"],
            kissing_score=scores["kissing"],
            safe_score=scores["safe_neutral"],
        )

    def unload(self):
        if hasattr(self, "model"):
            if hasattr(self.model, "cpu"):
                self.model.cpu()
            del self.model
            del self.processor
        if hasattr(self, "text_embeds"):
            del self.text_embeds
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
