import torch
from transformers import CLIPProcessor, CLIPModel
from pydantic import BaseModel
import numpy as np
from PIL import Image
from pureframe.hardware import ProfileSettings, HardwareProfile
import logging

logger = logging.getLogger(__name__)

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
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if getattr(settings, 'use_fp16', False):
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
                
        inputs = self.processor(text=self.text_inputs, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            self.text_embeds = self.model.text_projection(text_outputs[1])
            self.text_embeds = self.text_embeds / self.text_embeds.norm(p=2, dim=-1, keepdim=True)
            
    def classify_shot(self, representative_frame_bgr: np.ndarray) -> ShotContext:
        if not self.enabled:
            return ShotContext(explicit_act_score=0.0, implied_sex_score=0.0, kissing_score=0.0, safe_score=1.0)
            
        image = Image.fromarray(representative_frame_bgr[:, :, ::-1]) # BGR to RGB
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        if next(self.model.parameters()).dtype == torch.float16:
            inputs['pixel_values'] = inputs['pixel_values'].half()
            
        with torch.no_grad():
            image_outputs = self.model.vision_model(**inputs)
            image_embeds = self.model.visual_projection(image_outputs[1])
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            
            logits_per_image = torch.matmul(image_embeds, self.text_embeds.t()) * self.model.logit_scale.exp()
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
        scores = {cat: 0.0 for cat in PROMPT_SETS.keys()}
        for prob, cat in zip(probs, self.category_names):
            scores[cat] += prob
            
        return ShotContext(
            explicit_act_score=scores["explicit_act"],
            implied_sex_score=scores["implied_sex_context"],
            kissing_score=scores["kissing"],
            safe_score=scores["safe_neutral"]
        )

    def unload(self):
        if hasattr(self, 'model'):
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            del self.processor
        if hasattr(self, 'text_embeds'):
            del self.text_embeds
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
