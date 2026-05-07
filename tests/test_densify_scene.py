"""Tests for densify_shot and SceneDetector with mocked dependencies."""

from unittest.mock import patch, MagicMock
import numpy as np

from pureframe.pipeline.shots import Shot
from pureframe.pipeline.detect.nudity import Detection


class TestDensifyShot:
    def test_densify_produces_detections(self, synthetic_video):
        """Test densify_shot with a mocked NudityDetector."""
        from pureframe.hardware import get_settings, HardwareProfile

        settings = get_settings(HardwareProfile.CPU)

        # Create a mock detector that returns empty detections
        mock_detector = MagicMock()
        mock_detector.detect_batch.return_value = [[]]

        shot = Shot(index=0, start_frame=0, end_frame=30, start_time=0.0, end_time=1.25)

        from pureframe.pipeline.densify import densify_shot

        result = densify_shot(
            shot=shot,
            video_path=synthetic_video,
            detector=mock_detector,
            settings=settings,
            threshold=0.5,
        )

        assert isinstance(result, dict)
        assert 0 in result  # First frame should be present
        assert 29 in result  # Last frame should be present

    def test_densify_with_detections(self, synthetic_video):
        """Test densify with a detector that returns actual detections."""
        from pureframe.hardware import get_settings, HardwareProfile

        settings = get_settings(HardwareProfile.CPU)

        det = Detection(label="EXPOSED_BREAST_F", score=0.85, box=(50, 50, 150, 150))
        mock_detector = MagicMock()
        mock_detector.detect_batch.return_value = [[det]]

        shot = Shot(index=0, start_frame=0, end_frame=20, start_time=0.0, end_time=0.83)

        from pureframe.pipeline.densify import densify_shot

        result = densify_shot(
            shot=shot,
            video_path=synthetic_video,
            detector=mock_detector,
            settings=settings,
            threshold=0.5,
        )

        # Should have detections since score > threshold
        has_dets = any(len(dets) > 0 for dets in result.values())
        assert has_dets

    def test_densify_threshold_filters(self, synthetic_video):
        """Test that threshold filters out low-confidence detections."""
        from pureframe.hardware import get_settings, HardwareProfile

        settings = get_settings(HardwareProfile.CPU)

        low_det = Detection(label="EXPOSED_BREAST_F", score=0.3, box=(50, 50, 150, 150))
        mock_detector = MagicMock()
        mock_detector.detect_batch.return_value = [[low_det]]

        shot = Shot(index=0, start_frame=0, end_frame=10, start_time=0.0, end_time=0.42)

        from pureframe.pipeline.densify import densify_shot

        result = densify_shot(
            shot=shot,
            video_path=synthetic_video,
            detector=mock_detector,
            settings=settings,
            threshold=0.5,
        )

        # All detections should be empty (score 0.3 < threshold 0.5)
        all_empty = all(len(dets) == 0 for dets in result.values())
        assert all_empty


class TestSceneDetectorMocked:
    def test_scene_detector_init(self):
        """Test SceneDetector creation with mocked model loading."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("pureframe.pipeline.detect.scene.CLIPModel") as MockCLIP:
            with patch("pureframe.pipeline.detect.scene.CLIPProcessor") as MockProc:
                MockCLIP.from_pretrained.return_value.to.return_value = mock_model
                MockProc.from_pretrained.return_value = mock_processor

                from pureframe.pipeline.detect.scene import SceneDetector

                detector = SceneDetector(device="cpu")
                assert detector.device == "cpu"
                assert len(detector.prompts) == 7

    def test_analyze_frame_returns_scores(self):
        """Test analyze_frame with mocked model inference."""
        import torch

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Mock the model output
        mock_outputs = MagicMock()
        logits = torch.tensor([[1.0, 0.5, 0.3, 0.1, 0.2, 0.05, 0.01]])
        mock_outputs.logits_per_image = logits
        mock_model.__call__ = MagicMock(return_value=mock_outputs)

        mock_processor.return_value = {"input_ids": torch.zeros(1), "pixel_values": torch.zeros(1)}

        with patch("pureframe.pipeline.detect.scene.CLIPModel") as MockCLIP:
            with patch("pureframe.pipeline.detect.scene.CLIPProcessor") as MockProc:
                MockCLIP.from_pretrained.return_value.to.return_value = mock_model
                MockProc.from_pretrained.return_value = mock_processor

                from pureframe.pipeline.detect.scene import SceneDetector

                detector = SceneDetector(device="cpu")
                detector.model = mock_model

                # Create a fake frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Mock the processor to return tensors
                detector.processor = MagicMock()
                detector.processor.return_value = {
                    "input_ids": torch.zeros(7, 10),
                    "pixel_values": torch.zeros(1, 3, 224, 224),
                }
                mock_model.return_value = mock_outputs

                scores = detector.analyze_frame(frame)
                assert isinstance(scores, dict)
                assert len(scores) == 7
                assert all(isinstance(v, float) for v in scores.values())
