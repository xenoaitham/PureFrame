import pytest
import numpy as np
import cv2
from pureframe.pipeline.detect.scene_clip import SceneClassifier
from pureframe.hardware import get_settings, HardwareProfile

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

    total = (
        ctx.explicit_act_score
        + ctx.implied_sex_score
        + ctx.kissing_score
        + ctx.safe_score
    )
    assert np.isclose(total, 1.0, atol=1e-3)
    assert ctx.safe_score > 0.0
