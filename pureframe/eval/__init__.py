"""PureFrame Evaluation Benchmark - synthetic + real-world quality metrics.

This module generates a comprehensive evaluation report by testing PureFrame's
detection pipeline against synthetic test scenarios and (when available) annotated
video datasets.

Usage:
    python -m pureframe.eval.benchmark --output report.json
    pureframe evaluate [--output report.json]

Metrics computed:
    - Precision: TP / (TP + FP) - how many flagged frames were actually explicit
    - Recall: TP / (TP + FN) - how many explicit frames were correctly flagged
    - F1 Score: 2 * (P * R) / (P + R) - harmonic mean
    - False Positive Rate: FP / (FP + TN)
    - Processing Speed: frames/second on CPU vs GPU
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneResult:
    """Result for a single test scene."""

    scene_id: str
    genre: str
    description: str
    expected_explicit: bool
    detected_explicit: bool
    confidence: float
    detection_time_ms: float
    true_positive: bool = False
    true_negative: bool = False
    false_positive: bool = False
    false_negative: bool = False
    # Detection signature: every label -> max score the model emitted for
    # this scenario, explicit or not. This is the fingerprint the eval-parity
    # gate compares - sensitive to ANY model-behavior change (quantization,
    # version bumps, preprocessing drift), not just explicit-label hits.
    labels: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.expected_explicit and self.detected_explicit:
            self.true_positive = True
        elif not self.expected_explicit and not self.detected_explicit:
            self.true_negative = True
        elif not self.expected_explicit and self.detected_explicit:
            self.false_positive = True
        elif self.expected_explicit and not self.detected_explicit:
            self.false_negative = True


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    version: str = ""
    timestamp: str = ""
    total_scenes: int = 0
    results: list = field(default_factory=list)

    # Aggregate metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0
    accuracy: float = 0.0

    # Per-genre breakdown
    genre_metrics: dict = field(default_factory=dict)

    # Per-threshold analysis
    threshold_analysis: list = field(default_factory=list)

    def compute_metrics(self):
        tp = sum(1 for r in self.results if r.true_positive)
        tn = sum(1 for r in self.results if r.true_negative)
        fp = sum(1 for r in self.results if r.false_positive)
        fn = sum(1 for r in self.results if r.false_negative)

        self.total_scenes = len(self.results)
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1_score = (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )
        self.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        self.accuracy = (tp + tn) / self.total_scenes if self.total_scenes > 0 else 0.0

        # Per-genre
        genres = set(r.genre for r in self.results)
        for genre in genres:
            genre_results = [r for r in self.results if r.genre == genre]
            g_tp = sum(1 for r in genre_results if r.true_positive)
            g_tn = sum(1 for r in genre_results if r.true_negative)
            g_fp = sum(1 for r in genre_results if r.false_positive)
            g_fn = sum(1 for r in genre_results if r.false_negative)
            g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0.0
            g_rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
            g_f1 = (
                2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
            )

            self.genre_metrics[genre] = {
                "total": len(genre_results),
                "tp": g_tp,
                "tn": g_tn,
                "fp": g_fp,
                "fn": g_fn,
                "precision": round(g_prec, 4),
                "recall": round(g_rec, 4),
                "f1": round(g_f1, 4),
            }

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "timestamp": self.timestamp,
            "total_scenes": self.total_scenes,
            "aggregate_metrics": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "false_positive_rate": round(self.false_positive_rate, 4),
                "accuracy": round(self.accuracy, 4),
            },
            "genre_metrics": self.genre_metrics,
            "threshold_analysis": self.threshold_analysis,
            "results": [asdict(r) for r in self.results],
        }
        return d

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Evaluation report saved to {path}")


# ── Synthetic Test Scenarios ──────────────────────────────────────────────────


SYNTHETIC_SCENARIOS = [
    # Live-action explicit
    {
        "id": "LA-001",
        "genre": "live-action",
        "desc": "Full nudity in well-lit bedroom scene",
        "explicit": True,
        "frame_type": "high_skin_ratio",
    },
    {
        "id": "LA-002",
        "genre": "live-action",
        "desc": "Brief partial nudity in shower scene",
        "explicit": True,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "LA-003",
        "genre": "live-action",
        "desc": "Beach scene with swimwear",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "LA-004",
        "genre": "live-action",
        "desc": "Romantic kissing scene",
        "explicit": False,
        "frame_type": "low_skin_ratio",
    },
    {
        "id": "LA-005",
        "genre": "live-action",
        "desc": "Conversation in office",
        "explicit": False,
        "frame_type": "no_skin",
    },
    {
        "id": "LA-006",
        "genre": "live-action",
        "desc": "Explicit sex scene with multiple actors",
        "explicit": True,
        "frame_type": "high_skin_ratio",
    },
    {
        "id": "LA-007",
        "genre": "live-action",
        "desc": "Medical examination scene",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "LA-008",
        "genre": "live-action",
        "desc": "Topless male workout scene",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "LA-009",
        "genre": "live-action",
        "desc": "Night club scene with revealing outfits",
        "explicit": False,
        "frame_type": "low_skin_ratio",
    },
    {
        "id": "LA-010",
        "genre": "live-action",
        "desc": "Breastfeeding scene",
        "explicit": True,
        "frame_type": "medium_skin_ratio",
    },
    # Animation
    {
        "id": "AN-001",
        "genre": "animation",
        "desc": "Animated explicit scene",
        "explicit": True,
        "frame_type": "cartoon_explicit",
    },
    {
        "id": "AN-002",
        "genre": "animation",
        "desc": "Cartoon characters in swimwear",
        "explicit": False,
        "frame_type": "cartoon_safe",
    },
    {
        "id": "AN-003",
        "genre": "animation",
        "desc": "Pixar-style family scene",
        "explicit": False,
        "frame_type": "cartoon_safe",
    },
    {
        "id": "AN-004",
        "genre": "animation",
        "desc": "Adult animation with nudity",
        "explicit": True,
        "frame_type": "cartoon_explicit",
    },
    {
        "id": "AN-005",
        "genre": "animation",
        "desc": "Animated bath scene (children's show)",
        "explicit": False,
        "frame_type": "cartoon_safe",
    },
    # Anime
    {
        "id": "AE-001",
        "genre": "anime",
        "desc": "Ecchi fanservice scene",
        "explicit": True,
        "frame_type": "anime_explicit",
    },
    {
        "id": "AE-002",
        "genre": "anime",
        "desc": "Hot springs episode (steam censored)",
        "explicit": False,
        "frame_type": "anime_borderline",
    },
    {
        "id": "AE-003",
        "genre": "anime",
        "desc": "Battle scene with torn clothing",
        "explicit": False,
        "frame_type": "anime_borderline",
    },
    {
        "id": "AE-004",
        "genre": "anime",
        "desc": "Hentai explicit content",
        "explicit": True,
        "frame_type": "anime_explicit",
    },
    {
        "id": "AE-005",
        "genre": "anime",
        "desc": "Slice of life school scene",
        "explicit": False,
        "frame_type": "anime_safe",
    },
    {
        "id": "AE-006",
        "genre": "anime",
        "desc": "Beach episode with bikinis",
        "explicit": False,
        "frame_type": "anime_borderline",
    },
    {
        "id": "AE-007",
        "genre": "anime",
        "desc": "Transformation sequence (magical girl)",
        "explicit": False,
        "frame_type": "anime_borderline",
    },
    {
        "id": "AE-008",
        "genre": "anime",
        "desc": "Gore/violence action scene",
        "explicit": False,
        "frame_type": "anime_safe",
    },
    # Low-light / Dark scenes
    {
        "id": "DK-001",
        "genre": "low-light",
        "desc": "Explicit scene in dark room",
        "explicit": True,
        "frame_type": "dark_skin",
    },
    {
        "id": "DK-002",
        "genre": "low-light",
        "desc": "Horror scene in darkness",
        "explicit": False,
        "frame_type": "dark_no_skin",
    },
    {
        "id": "DK-003",
        "genre": "low-light",
        "desc": "Candlelit intimate scene",
        "explicit": True,
        "frame_type": "dark_skin",
    },
    {
        "id": "DK-004",
        "genre": "low-light",
        "desc": "Night surveillance footage",
        "explicit": False,
        "frame_type": "dark_no_skin",
    },
    {
        "id": "DK-005",
        "genre": "low-light",
        "desc": "Movie theater scene",
        "explicit": False,
        "frame_type": "dark_no_skin",
    },
    # Music videos
    {
        "id": "MV-001",
        "genre": "music-video",
        "desc": "Music video with explicit nudity",
        "explicit": True,
        "frame_type": "high_skin_ratio",
    },
    {
        "id": "MV-002",
        "genre": "music-video",
        "desc": "Pop music video with dancers in costumes",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "MV-003",
        "genre": "music-video",
        "desc": "R&B video with provocative dancing",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "MV-004",
        "genre": "music-video",
        "desc": "Rock concert footage",
        "explicit": False,
        "frame_type": "no_skin",
    },
    # Documentary
    {
        "id": "DC-001",
        "genre": "documentary",
        "desc": "Nature documentary - animals",
        "explicit": False,
        "frame_type": "no_skin",
    },
    {
        "id": "DC-002",
        "genre": "documentary",
        "desc": "Art documentary with classical nude paintings",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "DC-003",
        "genre": "documentary",
        "desc": "Medical documentary with surgical footage",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "DC-004",
        "genre": "documentary",
        "desc": "Tribal documentary with indigenous nudity",
        "explicit": True,
        "frame_type": "high_skin_ratio",
    },
    # Family / Safe content
    {
        "id": "FM-001",
        "genre": "family",
        "desc": "Children's cartoon",
        "explicit": False,
        "frame_type": "cartoon_safe",
    },
    {
        "id": "FM-002",
        "genre": "family",
        "desc": "Family dinner scene",
        "explicit": False,
        "frame_type": "no_skin",
    },
    {
        "id": "FM-003",
        "genre": "family",
        "desc": "Kids playing at pool",
        "explicit": False,
        "frame_type": "low_skin_ratio",
    },
    {
        "id": "FM-004",
        "genre": "family",
        "desc": "Animated movie - Finding Nemo type",
        "explicit": False,
        "frame_type": "cartoon_safe",
    },
    # Edge cases
    {
        "id": "EC-001",
        "genre": "edge-case",
        "desc": "Mannequin / statue nudity",
        "explicit": False,
        "frame_type": "medium_skin_ratio",
    },
    {
        "id": "EC-002",
        "genre": "edge-case",
        "desc": "Body paint covering nudity",
        "explicit": False,
        "frame_type": "high_skin_ratio",
    },
    {
        "id": "EC-003",
        "genre": "edge-case",
        "desc": "Quick flash frame (1/24s exposure)",
        "explicit": True,
        "frame_type": "flash_frame",
    },
    {
        "id": "EC-004",
        "genre": "edge-case",
        "desc": "Scene reflected in mirror",
        "explicit": True,
        "frame_type": "reflected_explicit",
    },
    {
        "id": "EC-005",
        "genre": "edge-case",
        "desc": "Blurred/pixelated already-censored content",
        "explicit": False,
        "frame_type": "already_censored",
    },
    {
        "id": "EC-006",
        "genre": "edge-case",
        "desc": "Video call showing explicit content on screen-in-screen",
        "explicit": True,
        "frame_type": "screen_in_screen",
    },
    {
        "id": "EC-007",
        "genre": "edge-case",
        "desc": "Nude art photography in black and white",
        "explicit": True,
        "frame_type": "bw_explicit",
    },
    {
        "id": "EC-008",
        "genre": "edge-case",
        "desc": "Sports scene - wrestling",
        "explicit": False,
        "frame_type": "high_skin_ratio",
    },
    {
        "id": "EC-009",
        "genre": "edge-case",
        "desc": "Wardrobe malfunction (brief accidental exposure)",
        "explicit": True,
        "frame_type": "flash_frame",
    },
    {
        "id": "EC-010",
        "genre": "edge-case",
        "desc": "Graphic violence (non-sexual)",
        "explicit": False,
        "frame_type": "no_skin",
    },
]


def _generate_synthetic_frame(frame_type: str) -> np.ndarray:
    """Generate a deterministic synthetic test frame with known properties.

    The layout pass paints the per-type regions; a photographic-texture pass
    (blur + seeded noise + gradient) follows so the frames resemble camera
    output - real NudeNet barely responds to flat color patches. The seed is
    derived from the frame type so every run produces identical inputs, a
    requirement for the eval-parity gate.
    """
    import zlib

    import cv2

    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    if frame_type == "high_skin_ratio":
        # Large skin-colored region (potential explicit)
        frame[100:400, 150:500] = [180, 200, 230]  # skin tone in BGR

    elif frame_type == "medium_skin_ratio":
        # Moderate skin region
        frame[150:350, 200:400] = [170, 195, 225]
        frame[0:100, :] = [200, 180, 160]  # blue sky

    elif frame_type == "low_skin_ratio":
        # Small skin region (face only)
        frame[:] = [40, 60, 80]  # dark background
        frame[200:260, 280:340] = [175, 200, 230]  # small face

    elif frame_type == "no_skin":
        # Landscape / no people
        frame[:240] = [200, 180, 140]  # sky
        frame[240:] = [50, 100, 60]  # grass

    elif frame_type in ("cartoon_explicit", "anime_explicit"):
        # Bright colors with skin tones
        frame[:] = [255, 240, 230]
        frame[100:380, 200:440] = [180, 200, 240]

    elif frame_type in ("cartoon_safe", "anime_safe"):
        # Bright cartoon colors, no skin
        frame[:] = [255, 200, 100]
        frame[100:200, 100:300] = [100, 200, 255]

    elif frame_type == "anime_borderline":
        # Mix of skin tones and bright colors
        frame[:] = [220, 210, 200]
        frame[150:300, 200:400] = [185, 205, 235]

    elif frame_type in ("dark_skin", "dark_no_skin"):
        # Very dark frame
        base = 30 if frame_type == "dark_no_skin" else 50
        frame[:] = [base, base, base]
        if "skin" in frame_type:
            frame[200:350, 250:400] = [80, 90, 100]

    elif frame_type == "flash_frame":
        # Single bright frame
        frame[:] = [200, 210, 230]

    elif frame_type == "reflected_explicit":
        # Half frame normal, half with skin tones (mirror)
        frame[:, :320] = [100, 100, 100]
        frame[:, 320:] = [175, 200, 230]

    elif frame_type == "already_censored":
        # Pixelated/blurred region
        frame[:] = [128, 128, 128]

    elif frame_type == "screen_in_screen":
        # Small rectangle with skin tones in corner
        frame[:] = [60, 60, 60]
        frame[50:150, 400:550] = [180, 200, 230]

    elif frame_type == "bw_explicit":
        # Black and white skin tones
        gray = 180
        frame[:] = [gray, gray, gray]
        frame[100:350, 150:450] = [200, 200, 200]

    # Photographic texture pass - identical for every frame type.
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    rng = np.random.default_rng(zlib.crc32(frame_type.encode("utf-8")))
    noise = rng.integers(-10, 11, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gradient = np.linspace(18, -18, w, dtype=np.int16).reshape(1, w, 1)
    frame = np.clip(frame.astype(np.int16) + gradient, 0, 255).astype(np.uint8)

    return frame


def run_synthetic_benchmark(
    detector=None,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Run the synthetic benchmark suite.

    Args:
        detector: Optional NudityDetector instance. If None, creates one.
        threshold: Detection confidence threshold.

    Returns:
        Complete EvaluationReport with metrics.
    """
    from datetime import datetime
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        ver = pkg_version("pureframe")
    except PackageNotFoundError:
        ver = "dev"

    report = EvaluationReport(
        version=ver,
        timestamp=datetime.now().isoformat(),
    )

    if detector is None:
        from pureframe.hardware import HardwareProfile, get_settings
        from pureframe.pipeline.detect.nudity import NudityDetector

        # CPU settings: deterministic across machines, no GPU variance in
        # the numbers this gate compares.
        detector = NudityDetector(get_settings(HardwareProfile.CPU))

    # Explicit labels that should trigger censoring - the canonical set from
    # the detector module (NudeNet 3.x names; the old 2.x names here made
    # every scenario score as safe).
    from pureframe.pipeline.detect.nudity import EXPLICIT_LABELS

    explicit_labels = EXPLICIT_LABELS

    for scenario in SYNTHETIC_SCENARIOS:
        frame = _generate_synthetic_frame(scenario["frame_type"])

        start = time.perf_counter()
        detections = detector.detect_batch([frame])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Check if any detection crosses threshold with explicit label
        detected = any(
            d.score >= threshold and d.label in explicit_labels for d in detections
        )
        max_conf = max(
            (d.score for d in detections if d.label in explicit_labels),
            default=0.0,
        )

        # Detection signature over ALL labels (not just explicit ones).
        label_scores: dict[str, float] = {}
        for d in detections:
            label_scores[d.label] = max(label_scores.get(d.label, 0.0), d.score)

        result = SceneResult(
            scene_id=scenario["id"],
            genre=scenario["genre"],
            description=scenario["desc"],
            expected_explicit=scenario["explicit"],
            detected_explicit=detected,
            confidence=max_conf,
            detection_time_ms=elapsed_ms,
            labels=label_scores,
        )
        report.results.append(result)

    report.compute_metrics()

    # Threshold sweep
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = sum(1 for r in report.results if r.expected_explicit and r.confidence >= t)
        fp = sum(
            1 for r in report.results if not r.expected_explicit and r.confidence >= t
        )
        fn = sum(1 for r in report.results if r.expected_explicit and r.confidence < t)
        tn = sum(
            1 for r in report.results if not r.expected_explicit and r.confidence < t
        )

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        report.threshold_analysis.append(
            {
                "threshold": t,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    return report
