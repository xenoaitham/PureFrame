"""Plugin corners a real user hits on day two: resuming an interrupted
plan with a plugin enabled, running with ``--no-cache`` beside a
plugin-enabled cache entry, and applying a plugin-flagged plan on a
machine where the plugin is not installed.

The fixture plugin classifies each frame by the clip's shot shades
(dark gray / busy testsrc2 / light gray), so the tests can tell a
resumed shot (loaded from the checkpoint store, never re-detected) from
an analyzed one without depending on how detect_batch batches its
calls: one batch per analyzed shot plus more during the densify pass of
a flagged shot.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from pureframe.cli import generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile
from pureframe.pipeline.detect.nudity import Detection, NudityDetector
from pureframe.pipeline.shots import Action, Category, ShotVerdict
from pureframe.plugin_api import PluginRegistration
from tests.conftest import frame_has_marker

WEAPON = "WEAPON_VISIBLE"
BOX = (60, 40, 260, 200)


def _shade(frame_bgr) -> str:
    """Which three-shot clip band a frame belongs to."""
    mean = float(np.mean(frame_bgr))
    spread = float(np.std(frame_bgr))
    if spread < 6.0:
        if mean < 90:
            return "dark"  # shot 0 (0x303030)
        if mean > 150:
            return "light"  # shot 2 (0xC0C0C0)
    return "busy"  # shot 1 (testsrc2)


class CountingDetector:
    """Fixture plugin: flags the marker frame, tallies the shot shades it
    was asked about."""

    label_categories = {"pistol": WEAPON}
    label_thresholds = {"pistol": 0.6}

    #: Class-level so tests can observe instances generate_plan constructs.
    shades: ClassVar[dict[str, int]] = {}
    marker_frames: ClassVar[int] = 0

    @classmethod
    def reset(cls):
        cls.shades = {"dark": 0, "light": 0, "busy": 0}
        cls.marker_frames = 0

    def __init__(self, settings):
        self.settings = settings
        self.unloaded = False

    def detect_batch(self, frames_bgr):
        for f in frames_bgr:
            self.shades[_shade(f)] += 1
            if frame_has_marker(f):
                CountingDetector.marker_frames += 1
        return [
            [Detection(label="pistol", score=0.9, box=BOX)]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    def unload(self):
        self.unloaded = True


def _registration() -> PluginRegistration:
    return PluginRegistration(
        name="weapons",
        cls=CountingDetector,
        label_categories={"pistol": WEAPON},
        label_thresholds={"pistol": 0.6},
    )


def _patched_registry(monkeypatch, registration):
    from pureframe import plugin_api

    monkeypatch.setattr(
        plugin_api, "discover", lambda: {registration.name: registration}
    )


def _patch_empty_registry(monkeypatch):
    from pureframe import plugin_api

    monkeypatch.setattr(plugin_api, "discover", lambda: {})


def _dense_settings(monkeypatch):
    """Sample densely so the marker frame lands among the keyframes."""
    import pureframe.cli

    original = pureframe.cli.get_settings

    def dense(profile, **kwargs):
        s = original(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense)


def _config(three_shot_video, tmp_path, **kwargs) -> Config:
    kwargs.setdefault("no_clip", True)
    kwargs.setdefault("no_audio", True)
    return Config.from_cli(
        input_path=three_shot_video,
        output_path=tmp_path / "out.mp4",
        profile=HardwareProfile.CPU,
        enabled_plugins=["weapons"],
        **kwargs,
    )


def _silence_nudity(monkeypatch):
    monkeypatch.setattr(
        NudityDetector, "detect_batch", lambda self, frames: [[] for _ in frames]
    )


class TestResumeWithPluginEnabled:
    def test_interrupted_plan_resumes_from_the_store(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        """An interrupted first run left shot 0 analyzed. Resuming with the
        plugin enabled must continue the SAME job, reuse the stored verdict,
        and detect only the remaining shots."""
        _silence_nudity(monkeypatch)
        _dense_settings(monkeypatch)
        _patched_registry(monkeypatch, _registration())

        config = _config(three_shot_video, tmp_path)
        job = mock_store.find_or_create_job(
            config.input_path, config.output_path, config
        )
        mock_store.update_status(job.id, "DETECTING", total_shots=3)
        mock_store.save_verdict(
            job.id,
            ShotVerdict(
                shot_index=0,
                category=Category.SAFE,
                action=Action.NONE,
                confidence=1.0,
                reasoning="saved by the interrupted first run",
            ),
        )

        CountingDetector.reset()
        plan = generate_plan(config)

        verdicts = {v.shot_index: v for v in plan.verdicts}
        assert set(verdicts) == {0, 1, 2}
        # Shot 0 came from the store, not from a fresh detection: its frames
        # were never handed to the plugin again.
        assert verdicts[0].action == Action.NONE
        assert verdicts[0].reasoning == "saved by the interrupted first run"
        assert CountingDetector.shades["dark"] == 0, (
            "a resumed shot must not be re-analyzed"
        )
        # The marker lives in the middle shot: the plugin flagged it.
        flagged = [v for v in verdicts.values() if v.action != Action.NONE]
        assert len(flagged) == 1
        assert flagged[0].category == Category.PLUGIN_BOX
        assert flagged[0].plugin_category == WEAPON
        assert CountingDetector.marker_frames >= 1
        # Shot 2 was fully analyzed on the resume.
        assert CountingDetector.shades["light"] == 10


class TestNoCacheWithPlugin:
    def test_no_cache_reanalyzes_and_leaves_the_cache_entry_alone(
        self, three_shot_video, tmp_path, monkeypatch, mock_store
    ):
        _silence_nudity(monkeypatch)
        _dense_settings(monkeypatch)
        _patched_registry(monkeypatch, _registration())

        # First run populates the checkpoint store: all three shots (the
        # flagged shot's densify pass adds extra busy frames beyond the 10
        # sampled keyframes).
        CountingDetector.reset()
        first = generate_plan(_config(three_shot_video, tmp_path))
        assert CountingDetector.shades["dark"] == 10
        assert CountingDetector.shades["light"] == 10
        assert CountingDetector.shades["busy"] >= 10
        first_flagged = [v for v in first.verdicts if v.action != Action.NONE]
        assert len(first_flagged) == 1
        job_id_first = mock_store.list_unfinished()[-1].id

        # --no-cache: salted config hash, full re-analysis, identical result.
        CountingDetector.reset()
        salted = generate_plan(
            _config(three_shot_video, tmp_path, no_cache=True, cache_salt="abc123")
        )
        assert CountingDetector.shades["dark"] == 10, (
            "--no-cache must re-analyze every shot"
        )
        assert CountingDetector.shades["light"] == 10
        salted_flagged = [v for v in salted.verdicts if v.action != Action.NONE]
        assert len(salted_flagged) == 1
        assert salted_flagged[0].plugin_category == WEAPON
        assert salted_flagged[0].boxes, "salted rerun still densifies boxes"
        assert len(mock_store.list_unfinished()) == 2, (
            "the salted run must be a separate job, not an overwrite"
        )

        # A plain re-run after that still rides the first run's cache entry.
        CountingDetector.reset()
        cached = generate_plan(_config(three_shot_video, tmp_path))
        assert CountingDetector.shades == {"dark": 0, "light": 0, "busy": 0}, (
            "a plain re-run must reuse the unsalted cache entry"
        )
        assert cached.verdicts == first.verdicts
        assert mock_store.list_unfinished()[-1].id == job_id_first


class TestPluginPlanRendersWithoutPlugin:
    def test_saved_plan_applies_on_a_machine_without_the_plugin(
        self, three_shot_video, tmp_path, monkeypatch
    ):
        """Generate and save a plan with the plugin, then drop the plugin
        from the registry entirely and render the saved JSON. Boxes are
        data; the render must not need the code that produced them."""
        import cv2

        _silence_nudity(monkeypatch)
        _dense_settings(monkeypatch)
        _patched_registry(monkeypatch, _registration())

        config = _config(three_shot_video, tmp_path)
        plan = generate_plan(config)
        plan_path = tmp_path / "plugin_flagged.censorplan.json"
        plan.serialize(plan_path)

        # The plugin is gone from this machine (and with it the module the
        # registry would import).
        _patch_empty_registry(monkeypatch)
        from pureframe.plugin_api import discover

        assert discover() == {}

        from pureframe.cli import execute_render

        out_path = tmp_path / "rendered_without_plugin.mp4"
        render_config = Config.from_cli(
            input_path=three_shot_video,
            output_path=out_path,
            profile=HardwareProfile.CPU,
            no_clip=True,
            no_audio=True,
        )
        execute_render(plan, render_config)

        assert out_path.exists() and out_path.stat().st_size > 0

        flagged = [v for v in plan.verdicts if v.action != Action.NONE][0]
        shot = plan.shots[flagged.shot_index]
        mid = (shot.start_frame + shot.end_frame) // 2

        def grab(path, frame_idx):
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            cap.release()
            assert ok
            return frame

        def lap_variance(frame):
            x1, y1, x2, y2 = BOX
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Container intact: the render preserves the frame count.
        src_cap = cv2.VideoCapture(str(three_shot_video))
        out_cap = cv2.VideoCapture(str(out_path))
        src_count = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_count = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_cap.release()
        out_cap.release()
        assert out_count == src_count

        # Untouched segments stay untouched: shot 0 was never flagged, and
        # the smart renderer stream-copies clean segments. The small
        # tolerance only covers a hypothetical full-re-encode fallback.
        clean_noise = np.abs(
            grab(out_path, 30).astype(int) - grab(three_shot_video, 30).astype(int)
        ).mean()
        assert clean_noise < 3.0, (
            f"an unflagged shot must survive the render (diff {clean_noise:.2f})"
        )

        # The blur itself lands at the flagged shot's midpoint. The box
        # holds the sharp magenta marker on busy testsrc2, so its Laplacian
        # variance collapses under the Gaussian blur; re-encode noise alone
        # would keep the variance near the source's.
        out_lap = lap_variance(grab(out_path, mid))
        src_lap = lap_variance(grab(three_shot_video, mid))
        assert src_lap > 50.0, "fixture expects a structured region to censor"
        assert out_lap < 0.05 * src_lap, (
            f"plugin boxes must blur the region on a plugin-less machine "
            f"(src lap {src_lap:.1f}, out lap {out_lap:.1f})"
        )
