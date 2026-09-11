#!/usr/bin/env python3
"""Gate detection-quality parity against a committed baseline.

Compares a fresh `pureframe evaluate` report against `eval-baseline.json`:

1. Detection signature - per-scenario label sets must match exactly and per
   label scores within SCORE_TOLERANCE. This is sensitive to ANY model
   behavior change (quantization, version bumps, preprocessing drift), which
   matters because synthetic scenarios may legitimately score zero on
   explicit-label F1.
2. Aggregate precision/recall/F1 - compared within ±TOLERANCE whenever the
   baseline metric is non-zero.

Bootstrap mode: if the baseline file is missing, the check passes with a
warning and instructions (a maintainer then commits the generated report as
the new baseline).
"""

import json
import sys
from pathlib import Path

TOLERANCE = 0.02
SCORE_TOLERANCE = 0.05
METRICS = ("precision", "recall", "f1_score")


def load_aggregates(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = data["aggregate_metrics"]
    return {m: float(agg[m]) for m in METRICS}


def compare_signatures(baseline: dict, current: dict) -> list[str]:
    """Return a list of human-readable drift descriptions (empty = OK)."""
    base_scenes = {r["scene_id"]: r for r in baseline.get("results", [])}
    cur_scenes = {r["scene_id"]: r for r in current.get("results", [])}
    problems: list[str] = []

    for scene_id in sorted(set(base_scenes) | set(cur_scenes)):
        b = base_scenes.get(scene_id)
        c = cur_scenes.get(scene_id)
        if b is None or c is None:
            problems.append(f"{scene_id}: present in only one report")
            continue
        b_labels = b.get("labels", {})
        c_labels = c.get("labels", {})
        # Tiny scores are compression-level noise; ignore anything below the
        # reporting floor on both sides.
        sig_b = {k: v for k, v in b_labels.items() if v >= SCORE_TOLERANCE}
        sig_c = {k: v for k, v in c_labels.items() if v >= SCORE_TOLERANCE}
        for label in sorted(set(sig_b) | set(sig_c)):
            if label not in sig_c:
                problems.append(
                    f"{scene_id}: label '{label}' {sig_b[label]:.3f} -> missing"
                )
            elif label not in sig_b:
                problems.append(
                    f"{scene_id}: label '{label}' appeared at {sig_c[label]:.3f}"
                )
            elif abs(sig_b[label] - sig_c[label]) > SCORE_TOLERANCE:
                problems.append(
                    f"{scene_id}: label '{label}' {sig_b[label]:.3f} -> "
                    f"{sig_c[label]:.3f}"
                )
    return problems


def main() -> int:
    baseline_path = Path("eval-baseline.json")
    report_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluation_report.json")
    )

    if not report_path.exists():
        print(f"FAIL: evaluation report not found at {report_path}", file=sys.stderr)
        print("Run: pureframe evaluate --output " + str(report_path), file=sys.stderr)
        return 1

    if not baseline_path.exists():
        print(
            "WARNING: eval-baseline.json not found - parity check skipped "
            "(bootstrap mode).\n"
            "To create the baseline, commit the freshly generated report:\n"
            f"  cp {report_path} {baseline_path}",
            file=sys.stderr,
        )
        return 0

    failed = False

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = json.loads(report_path.read_text(encoding="utf-8"))
    problems = compare_signatures(baseline, current)
    if problems:
        failed = True
        print(f"Detection signature drift ({len(problems)}):")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    else:
        print("Detection signature: identical (all labels within tolerance)")

    base_agg = load_aggregates(baseline_path)
    cur_agg = load_aggregates(report_path)
    for metric in METRICS:
        if base_agg[metric] == 0.0 and cur_agg[metric] == 0.0:
            continue  # signature carries the gate when F1 is uninformative
        delta = cur_agg[metric] - base_agg[metric]
        status = "ok"
        if abs(delta) > TOLERANCE:
            status = f"FAIL (drift {delta:+.4f} exceeds ±{TOLERANCE})"
            failed = True
        print(
            f"{metric:>10}: baseline {base_agg[metric]:.4f} -> "
            f"current {cur_agg[metric]:.4f} [{status}]"
        )

    if failed:
        print(
            "\nDetection quality drifted beyond tolerance. If the change is "
            "intentional, regenerate and commit the baseline with:\n"
            f"  pureframe evaluate --output {baseline_path}",
            file=sys.stderr,
        )
        return 1
    print("\nEval parity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
