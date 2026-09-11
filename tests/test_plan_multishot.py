"""Every shot of a multi-shot video must be analyzed.

Regression guard for the plan loop's prefetch worker (v0.2.0–v0.2.1): its
queue helper returned ``None`` on success, the shot loop read that as
"stop", and so only the *first* shot of any video was ever analyzed - a
movie with hundreds of shots came back with a single verdict and nothing
censored past it. Every fixture the suite had was single-shot, so nothing
caught it.
"""

import threading
from queue import Queue

from pureframe.cli import _extraction_worker, generate_plan
from pureframe.config import Config
from pureframe.hardware import HardwareProfile, get_settings
from pureframe.pipeline.probe import probe_video
from pureframe.pipeline.shots import Action, detect_shots
from pureframe.utils.timing import PhaseTimers
from tests.conftest import THREE_SHOT_MARKER_FRAMES, frame_has_marker

BOX = (60, 40, 260, 200)


def _mock_detector(monkeypatch):
    from pureframe.pipeline.detect.nudity import Detection, NudityDetector

    def mocked_detect_batch(self, frames_bgr):
        return [
            [Detection(label="FEMALE_BREAST_EXPOSED", score=0.99, box=BOX)]
            if frame_has_marker(f)
            else []
            for f in frames_bgr
        ]

    monkeypatch.setattr(NudityDetector, "detect_batch", mocked_detect_batch)


def _dense_sampling(monkeypatch):
    import pureframe.cli

    original = pureframe.cli.get_settings

    def dense(profile, **kwargs):
        s = original(profile)
        s.sample_keyframes_per_shot = 10
        return s

    monkeypatch.setattr(pureframe.cli, "get_settings", dense)


def test_generate_plan_analyzes_every_shot(three_shot_video, tmp_path, monkeypatch):
    _mock_detector(monkeypatch)
    _dense_sampling(monkeypatch)

    config = Config.from_cli(
        input_path=three_shot_video,
        output_path=tmp_path / "out.mp4",
        profile=HardwareProfile.CPU,
        no_clip=True,
        no_audio=True,
    )
    plan = generate_plan(config)

    assert len(plan.shots) == 3
    # One verdict per shot - not just the first.
    assert sorted(v.shot_index for v in plan.verdicts) == [0, 1, 2]

    flagged = [v for v in plan.verdicts if v.action != Action.NONE]
    assert [v.shot_index for v in flagged] == [1]
    middle = plan.shots[1]
    assert middle.start_frame <= min(THREE_SHOT_MARKER_FRAMES)
    assert middle.end_frame > max(THREE_SHOT_MARKER_FRAMES)

    # The flagged shot carries tracked boxes inside its own frame range.
    assert flagged[0].boxes
    box_frames = {b.frame_idx for b in flagged[0].boxes}
    assert box_frames <= set(range(middle.start_frame, middle.end_frame))
    assert box_frames & set(THREE_SHOT_MARKER_FRAMES)


def _run_worker(shots, completed, clip, stop_event=None, maxsize=8):
    settings = get_settings(HardwareProfile.CPU)
    meta = probe_video(clip)
    config = Config(input_path=clip, output_path=clip.with_suffix(".out.mp4"))
    queue: Queue = Queue(maxsize=maxsize)
    stop_event = stop_event or threading.Event()
    _extraction_worker(
        shots, completed, config, settings, meta, PhaseTimers(), queue, stop_event
    )
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def test_extraction_worker_delivers_every_shot(three_shot_video):
    shots = detect_shots(three_shot_video, frame_skip=2)
    assert len(shots) == 3

    items = _run_worker(shots, set(), three_shot_video)

    assert items[-1] is None  # end-of-stream sentinel
    delivered = [item for item in items[:-1] if not isinstance(item, Exception)]
    assert [item[0].index for item in delivered] == [0, 1, 2]
    for _shot, kf_indices, frames in delivered:
        assert kf_indices and set(kf_indices) == set(frames)


def test_extraction_worker_passes_completed_shots_through(three_shot_video):
    shots = detect_shots(three_shot_video, frame_skip=2)

    items = _run_worker(shots, {0}, three_shot_video)

    delivered = items[:-1]
    assert [item[0].index for item in delivered] == [0, 1, 2]
    # Already-checkpointed shots are forwarded empty so the consumer can
    # advance its progress bar without re-extracting anything.
    assert delivered[0][1] == [] and delivered[0][2] == {}
    assert delivered[1][1]


def test_extraction_worker_stops_when_asked(three_shot_video):
    shots = detect_shots(three_shot_video, frame_skip=2)
    stop = threading.Event()
    stop.set()

    items = _run_worker(shots, set(), three_shot_video, stop_event=stop, maxsize=1)

    # Nothing may be queued once the consumer has signalled shutdown - not
    # even the sentinel - so a torn-down plan loop never blocks on a full queue.
    assert items == []
