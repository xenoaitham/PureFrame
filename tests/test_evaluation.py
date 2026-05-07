"""Tests for the evaluation benchmark module."""

import json
import pytest

from pureframe.eval import (
    SceneResult,
    EvaluationReport,
    SYNTHETIC_SCENARIOS,
    _generate_synthetic_frame,
)


class TestSceneResult:
    def test_true_positive(self):
        r = SceneResult(
            scene_id="T1",
            genre="test",
            description="test",
            expected_explicit=True,
            detected_explicit=True,
            confidence=0.9,
            detection_time_ms=10.0,
        )
        assert r.true_positive is True
        assert r.false_positive is False
        assert r.true_negative is False
        assert r.false_negative is False

    def test_true_negative(self):
        r = SceneResult(
            scene_id="T2",
            genre="test",
            description="test",
            expected_explicit=False,
            detected_explicit=False,
            confidence=0.1,
            detection_time_ms=10.0,
        )
        assert r.true_negative is True

    def test_false_positive(self):
        r = SceneResult(
            scene_id="T3",
            genre="test",
            description="test",
            expected_explicit=False,
            detected_explicit=True,
            confidence=0.7,
            detection_time_ms=10.0,
        )
        assert r.false_positive is True

    def test_false_negative(self):
        r = SceneResult(
            scene_id="T4",
            genre="test",
            description="test",
            expected_explicit=True,
            detected_explicit=False,
            confidence=0.3,
            detection_time_ms=10.0,
        )
        assert r.false_negative is True


class TestEvaluationReport:
    def test_compute_metrics_perfect(self):
        report = EvaluationReport()
        report.results = [
            SceneResult("1", "g", "d", True, True, 0.9, 1.0),
            SceneResult("2", "g", "d", False, False, 0.1, 1.0),
            SceneResult("3", "g", "d", True, True, 0.8, 1.0),
            SceneResult("4", "g", "d", False, False, 0.2, 1.0),
        ]
        report.compute_metrics()
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0
        assert report.false_positive_rate == 0.0
        assert report.accuracy == 1.0

    def test_compute_metrics_with_errors(self):
        report = EvaluationReport()
        report.results = [
            SceneResult("1", "a", "d", True, True, 0.9, 1.0),  # TP
            SceneResult("2", "a", "d", False, True, 0.6, 1.0),  # FP
            SceneResult("3", "b", "d", True, False, 0.3, 1.0),  # FN
            SceneResult("4", "b", "d", False, False, 0.1, 1.0),  # TN
        ]
        report.compute_metrics()
        assert report.precision == 0.5  # 1 TP / (1 TP + 1 FP)
        assert report.recall == 0.5  # 1 TP / (1 TP + 1 FN)
        assert report.total_scenes == 4
        assert "a" in report.genre_metrics
        assert "b" in report.genre_metrics

    def test_compute_metrics_empty(self):
        report = EvaluationReport()
        report.compute_metrics()
        assert report.precision == 0.0
        assert report.accuracy == 0.0
        assert report.total_scenes == 0

    def test_to_dict(self):
        report = EvaluationReport(version="test", timestamp="2026-01-01")
        report.results = [
            SceneResult("1", "g", "d", True, True, 0.9, 1.0),
        ]
        report.compute_metrics()
        d = report.to_dict()
        assert d["version"] == "test"
        assert "aggregate_metrics" in d
        assert "results" in d
        assert len(d["results"]) == 1

    def test_save(self, tmp_path):
        report = EvaluationReport(version="test", timestamp="now")
        report.results = [
            SceneResult("1", "g", "d", True, True, 0.9, 1.0),
        ]
        report.compute_metrics()
        out = tmp_path / "report.json"
        report.save(out)
        data = json.loads(out.read_text())
        assert data["version"] == "test"


class TestSyntheticScenarios:
    def test_scenario_count(self):
        assert len(SYNTHETIC_SCENARIOS) == 50

    def test_genres_covered(self):
        genres = set(s["genre"] for s in SYNTHETIC_SCENARIOS)
        assert "live-action" in genres
        assert "animation" in genres
        assert "anime" in genres
        assert "low-light" in genres
        assert "music-video" in genres
        assert "documentary" in genres
        assert "family" in genres
        assert "edge-case" in genres

    def test_has_explicit_and_safe(self):
        explicit = [s for s in SYNTHETIC_SCENARIOS if s["explicit"]]
        safe = [s for s in SYNTHETIC_SCENARIOS if not s["explicit"]]
        assert len(explicit) >= 15
        assert len(safe) >= 25

    def test_unique_ids(self):
        ids = [s["id"] for s in SYNTHETIC_SCENARIOS]
        assert len(ids) == len(set(ids))


class TestSyntheticFrameGeneration:
    @pytest.mark.parametrize(
        "frame_type",
        [
            "high_skin_ratio",
            "medium_skin_ratio",
            "low_skin_ratio",
            "no_skin",
            "cartoon_explicit",
            "cartoon_safe",
            "anime_explicit",
            "anime_safe",
            "anime_borderline",
            "dark_skin",
            "dark_no_skin",
            "flash_frame",
            "reflected_explicit",
            "already_censored",
            "screen_in_screen",
            "bw_explicit",
        ],
    )
    def test_frame_generation(self, frame_type):
        frame = _generate_synthetic_frame(frame_type)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype.name == "uint8"

    def test_unknown_type_returns_black(self):
        frame = _generate_synthetic_frame("totally_unknown")
        assert frame.shape == (480, 640, 3)
        # Should be all zeros (black)
        assert frame.sum() == 0


class TestRunSyntheticBenchmark:
    """Test run_synthetic_benchmark with a mock detector."""

    def test_benchmark_with_mock_detector(self):
        from unittest.mock import MagicMock
        from pureframe.eval import run_synthetic_benchmark

        # Create a fake detection result
        fake_det = MagicMock()
        fake_det.score = 0.85
        fake_det.label = "EXPOSED_BREAST_F"

        # Mock detector that returns one detection per frame
        detector = MagicMock()
        detector.detect_batch.return_value = [[fake_det]]

        report = run_synthetic_benchmark(detector=detector, threshold=0.5)

        assert report.total_scenes == 50
        assert report.version  # Should have a version string
        assert report.timestamp
        # All scenes detected as explicit (mock always returns explicit)
        # So explicit scenes → TP, safe scenes → FP
        assert report.precision > 0
        assert report.recall == 1.0  # All explicit detected
        assert len(report.threshold_analysis) == 7  # 7 threshold levels

    def test_benchmark_with_no_detections(self):
        from unittest.mock import MagicMock
        from pureframe.eval import run_synthetic_benchmark

        detector = MagicMock()
        detector.detect_batch.return_value = [[]]  # no detections

        report = run_synthetic_benchmark(detector=detector, threshold=0.5)

        assert report.total_scenes == 50
        # Nothing detected → all safe scenes = TN, all explicit = FN
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.accuracy > 0  # TN count > 0

    def test_threshold_sweep_values(self):
        from unittest.mock import MagicMock
        from pureframe.eval import run_synthetic_benchmark

        fake_det = MagicMock()
        fake_det.score = 0.65
        fake_det.label = "EXPOSED_GENITALIA_F"

        detector = MagicMock()
        detector.detect_batch.return_value = [[fake_det]]

        report = run_synthetic_benchmark(detector=detector, threshold=0.5)

        # Verify threshold sweep
        thresholds = [t["threshold"] for t in report.threshold_analysis]
        assert thresholds == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # At threshold 0.3, score 0.65 should detect → more positives
        # At threshold 0.9, score 0.65 should NOT detect → fewer positives
        t03 = report.threshold_analysis[0]  # threshold=0.3
        t09 = report.threshold_analysis[-1]  # threshold=0.9
        assert t03["tp"] >= t09["tp"]

    def test_report_serialization(self, tmp_path):
        from unittest.mock import MagicMock
        from pureframe.eval import run_synthetic_benchmark

        detector = MagicMock()
        detector.detect_batch.return_value = [[]]

        report = run_synthetic_benchmark(detector=detector)
        out = tmp_path / "bench.json"
        report.save(out)

        data = json.loads(out.read_text())
        assert data["total_scenes"] == 50
        assert "aggregate_metrics" in data
        assert "threshold_analysis" in data
        assert len(data["results"]) == 50
