"""Scoring real footage against hand-marked expected ranges.

The synthetic corpus and the eval-parity gate pin the detector's
behavior on generated clips; neither can say how PureFrame does on real
movies. This module closes that gap for one clip at a time: the owner
marks the ranges that should be censored in a small JSON file, the real
pipeline plans the clip, and the two are compared as time coverage per
category.

Granularity is shot-level, on purpose: the pipeline flags whole shots
(the renderer then blurs per-frame boxes inside them), so "the tool
flagged this shot but the mark only covered half of it" is exactly the
kind of miss this scoring surfaces.

Records are privacy-safe by construction, mirroring ``bench --real``:
the clip is identified by SHA-256 and basic metadata, the mark file by
its SHA-256 - no path, name or raw timestamp ever lands in the JSONL.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pureframe.checkpoint import content_fingerprint
from pureframe.pipeline.probe import VideoMetadata, probe_video

# Marks file schema:
#   {"ranges": [{"start": 12.5, "end": 18.0, "category": "NUDITY_EXPLICIT"},
#               {"start": 40.0, "end": 41.5}]}
# start/end are seconds; category is optional and defaults to "ANY"
# (matches a flagged shot of any category). Category names are the plan's
# Category values (NUDITY_EXPLICIT, KISS_INTENSE, ...).


@dataclass(frozen=True)
class MarkedRange:
    start: float
    end: float
    category: str = "ANY"


@dataclass(frozen=True)
class FlaggedRange:
    start: float
    end: float
    category: str


@dataclass
class CategoryScore:
    category: str
    expected_seconds: float = 0.0
    flagged_seconds: float = 0.0
    true_positive_seconds: float = 0.0
    missed: list[tuple[float, float]] = field(default_factory=list)
    extra: list[tuple[float, float]] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        if self.flagged_seconds <= 0:
            return None
        return min(self.true_positive_seconds / self.flagged_seconds, 1.0)

    @property
    def recall(self) -> float | None:
        if self.expected_seconds <= 0:
            return None
        return min(self.true_positive_seconds / self.expected_seconds, 1.0)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)


def load_marks(path: Path) -> list[MarkedRange]:
    """Parse and validate a hand-marked ranges JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ranges = data.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError(f"{path}: expected a non-empty 'ranges' list")
    marks: list[MarkedRange] = []
    for i, raw in enumerate(ranges):
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: range {i} needs numeric start/end") from e
        if not 0 <= start < end:
            raise ValueError(f"{path}: range {i} needs 0 <= start < end")
        category = str(raw.get("category", "ANY")).strip() or "ANY"
        marks.append(MarkedRange(start=start, end=end, category=category))
    return marks


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent intervals into a sorted disjoint union."""
    if not intervals:
        return []
    out: list[list[float]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def _overlap_seconds(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total length covered by both disjoint-sorted interval lists."""
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _subtract(
    intervals: list[tuple[float, float]], cover: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Parts of disjoint-sorted *intervals* not covered by disjoint-sorted *cover*."""
    remaining: list[tuple[float, float]] = []
    for start, end in intervals:
        cursor = start
        for c_start, c_end in cover:
            if c_end <= cursor or c_start >= end:
                continue
            if c_start > cursor:
                remaining.append((cursor, min(c_start, end)))
            cursor = max(cursor, c_end)
            if cursor >= end:
                break
        if cursor < end:
            remaining.append((cursor, end))
    return remaining


def score_ranges(
    expected: list[MarkedRange],
    flagged: list[FlaggedRange],
    tolerance: float = 0.5,
) -> list[CategoryScore]:
    """Per-category time-coverage precision/recall.

    *tolerance* widens the expected ranges before intersecting, absorbing
    shot-boundary snapping at the clip's cut points; 0.5 s is a sane
    default for movie cuts, 0.0 turns it off.
    """
    categories = sorted({m.category for m in expected} | {f.category for f in flagged})
    # "ANY" marks fold into every concrete category's expected side, so an
    # "ANY" score row could never carry flagged time - only emit one when
    # it is the sole category (everything missed).
    concrete = [c for c in categories if c != "ANY"]
    if not concrete and "ANY" in categories:
        concrete = ["ANY"]
    scores: list[CategoryScore] = []
    for category in concrete:
        # Expected time reports the marks as written; the tolerance only
        # decides what counts as a hit.
        raw_exp = _union(
            [(m.start, m.end) for m in expected if m.category in (category, "ANY")]
        )
        expected_seconds = sum(end - start for start, end in raw_exp)
        widened = _union(
            [
                (max(0.0, m.start - tolerance), m.end + tolerance)
                for m in expected
                if m.category in (category, "ANY")
            ]
        )
        flg = _union([(f.start, f.end) for f in flagged if f.category == category])
        flagged_seconds = sum(end - start for start, end in flg)
        # Raw hit time against the widened marks: precision measures the
        # flagged time the marks vouch for (tolerance widens what they
        # vouch for), recall is capped at 1.0 by the properties since the
        # widened marks can extend past the raw ones.
        tp = _overlap_seconds(flg, widened)
        # Symmetric slack: a mark counts as covered when a flagged range
        # reaches it within the tolerance, and a flagged range counts as
        # extra when no mark reaches it within the tolerance.
        widened_flagged = _union(
            [
                (max(0.0, s - tolerance), e + tolerance)
                for s, e in [
                    (f.start, f.end) for f in flagged if f.category == category
                ]
            ]
        )
        missed = _subtract(raw_exp, widened_flagged)
        extra = _subtract(flg, widened)
        scores.append(
            CategoryScore(
                category=category,
                expected_seconds=expected_seconds,
                flagged_seconds=flagged_seconds,
                true_positive_seconds=tp,
                missed=missed,
                extra=extra,
            )
        )
    return scores


def flagged_ranges_from_plan(plan) -> list[FlaggedRange]:
    """Flagged shot ranges (seconds) from a CensorPlan, with categories."""
    out: list[FlaggedRange] = []
    for verdict in plan.verdicts:
        if verdict.action.value == "NONE":
            continue
        shot = next((s for s in plan.shots if s.index == verdict.shot_index), None)
        if shot is None:
            continue
        category = (
            verdict.plugin_category
            if verdict.category.value == "PLUGIN_BOX" and verdict.plugin_category
            else verdict.category.value
        )
        out.append(
            FlaggedRange(
                start=shot.start_time,
                end=shot.end_time,
                category=category,
            )
        )
    return out


def _plan_via_cli(
    video: Path,
    plan_path: Path,
    profile: str | None,
    strictness: str | None,
    content_type: str | None,
    extra_args: list[str],
) -> str:
    """Run the real `pureframe plan` command in-process (the same surface
    users invoke, and the same flow `bench --real` drives via CliRunner);
    returns the CLI stdout for error reporting."""
    from typer.testing import CliRunner

    from pureframe.cli import app

    argv = [
        "plan",
        str(video),
        "--output",
        str(plan_path),
        *(["--profile", profile] if profile else []),
        *(["--strictness", strictness] if strictness else []),
        *(["--content-type", content_type] if content_type else []),
        *extra_args,
    ]
    result = CliRunner().invoke(app, argv)
    if result.exit_code != 0:
        raise SystemExit(f"pureframe plan failed on the clip:\n{result.output[-2000:]}")
    return result.output


def run_real_eval(
    video: Path,
    marks_path: Path,
    tolerance: float = 0.5,
    profile: str | None = None,
    strictness: str | None = None,
    content_type: str | None = None,
    extra_args: list[str] | None = None,
    jsonl_path: Path | None = None,
) -> tuple[list[CategoryScore], dict]:
    """Plan the clip with the real CLI, score it against the marks, and
    append a privacy-safe JSONL record. Returns (scores, summary dict)."""
    from pureframe.bench import capture_environment

    marks = load_marks(marks_path)
    meta: VideoMetadata = probe_video(video)

    flagged = _plan_shot_verdicts(
        video, profile, strictness, content_type, list(extra_args or [])
    )

    scores = score_ranges(marks, flagged, tolerance=tolerance)

    summary: dict = {
        "file": {
            "sha256": content_fingerprint(video),
            "duration_seconds": round(meta.duration_seconds, 2),
            "width": meta.width,
            "height": meta.height,
            "fps": float(meta.fps),
            "container": meta.container,
            "video_codec": meta.video_codec,
        },
        "marks_sha256": content_fingerprint(marks_path),
        "tolerance_seconds": tolerance,
        "config": {
            "profile": profile,
            "strictness": strictness,
            "content_type": content_type,
            "extra_args": list(extra_args or []),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_category": {
            s.category: {
                "expected_seconds": round(s.expected_seconds, 2),
                "flagged_seconds": round(s.flagged_seconds, 2),
                "true_positive_seconds": round(s.true_positive_seconds, 2),
                "precision": None if s.precision is None else round(s.precision, 4),
                "recall": None if s.recall is None else round(s.recall, 4),
                "f1": None if s.f1 is None else round(s.f1, 4),
            }
            for s in scores
        },
        "environment": capture_environment(),
    }

    sink = jsonl_path if jsonl_path else Path.cwd() / "pureframe_real_eval.jsonl"
    with open(sink, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "real-eval", **summary}) + "\n")

    return scores, summary


def _plan_shot_verdicts(
    video: Path,
    profile: str | None,
    strictness: str | None,
    content_type: str | None,
    extra_args: list[str],
) -> list[FlaggedRange]:
    """Plan the clip through the real CLI and return its flagged ranges."""
    import shutil
    import tempfile

    from pureframe.pipeline.render.plan import CensorPlan

    workdir = Path(tempfile.mkdtemp(prefix="pureframe_real_eval_"))
    plan_path = workdir / "plan.censorplan.json"
    try:
        _plan_via_cli(video, plan_path, profile, strictness, content_type, extra_args)
        plan = CensorPlan.load(plan_path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return flagged_ranges_from_plan(plan)


def format_report(scores: list[CategoryScore], summary: dict) -> str:
    """Human-readable scoring report for one clip."""
    f = summary["file"]
    lines = [
        "Real-footage evaluation",
        f"  clip: sha256 {f['sha256'][:16]}... "
        f"({f['duration_seconds']} s, {f['width']}x{f['height']}, "
        f"{f['container']}/{f['video_codec']})",
        f"  marks: sha256 {summary['marks_sha256'][:16]}..., "
        f"tolerance {summary['tolerance_seconds']} s",
        "",
        f"  {'category':<28} {'expected':>9} {'flagged':>9} {'hit':>8} "
        f"{'precision':>9} {'recall':>7} {'f1':>7}",
    ]
    for s in scores:
        p = "-" if s.precision is None else f"{s.precision:.3f}"
        r = "-" if s.recall is None else f"{s.recall:.3f}"
        f1 = "-" if s.f1 is None else f"{s.f1:.3f}"
        lines.append(
            f"  {s.category:<28} {s.expected_seconds:>8.1f}s "
            f"{s.flagged_seconds:>8.1f}s {s.true_positive_seconds:>7.1f}s "
            f"{p:>9} {r:>7} {f1:>7}"
        )
        for start, end in s.missed:
            lines.append(f"    missed: {start:8.2f} - {end:8.2f}")
        for start, end in s.extra:
            lines.append(f"    extra:  {start:8.2f} - {end:8.2f}")
    return "\n".join(lines)
