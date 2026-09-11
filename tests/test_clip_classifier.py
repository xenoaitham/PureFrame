import cv2
import numpy as np
import pytest

from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.detect.scene_clip import SceneClassifier

pytestmark = pytest.mark.slow


def test_clip_classifier_synthetic():
    settings = get_settings(HardwareProfile.LOW)
    settings.detection_resolution = 300
    classifier = SceneClassifier(settings)

    frame = np.ones((300, 300, 3), dtype=np.uint8) * 200
    # draw something that looks vaguely like a person on a bed
    cv2.rectangle(frame, (50, 200), (250, 250), (100, 100, 100), -1)
    cv2.ellipse(frame, (150, 150), (40, 100), 0, 0, 360, (0, 0, 0), -1)

    ctx = classifier.classify_shot(frame)

    # Per-category max-cosine maps each category independently to [0, 1]
    # (no cross-category softmax - categories with more prompts should not
    # automatically dominate). Each score is a bounded confidence, so we
    # assert ranges rather than a partition-of-unity.
    for score in (
        ctx.explicit_act_score,
        ctx.implied_sex_score,
        ctx.kissing_score,
        ctx.safe_score,
    ):
        assert 0.0 <= score <= 1.0
    assert ctx.safe_score > 0.0
