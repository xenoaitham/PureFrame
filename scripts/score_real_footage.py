#!/usr/bin/env python3
"""Score one real clip against hand-marked expected ranges.

Runs the real `pureframe plan` on a user-supplied clip and compares the
flagged shot ranges against a small hand-marked JSON of the ranges that
should be censored, per category. Prints a human-readable report and
appends a privacy-safe JSONL record (clip identified by SHA-256 and
metadata only, never a path) so numbers accumulate across clips and
runs, mirroring `bench --real`.

Marks file format:
    {"ranges": [
        {"start": 12.5, "end": 18.0, "category": "NUDITY_EXPLICIT"},
        {"start": 40.0, "end": 41.5}
    ]}

`start`/`end` are seconds into the clip. `category` is optional and
defaults to "ANY" (matches flagged shots of any category); otherwise it
is one of the plan's category names (NUDITY_EXPLICIT, KISS_INTENSE,
...). The workflow doc is docs/real-footage-eval.md.

Usage:
    uv run python scripts/score_real_footage.py clip.mkv marks.json \
        [--tolerance 0.5] [--profile CPU] [--strictness medium] \
        [--content-type live-action] [--jsonl my-results.jsonl] \
        [extra pureframe plan flags...]
"""

import argparse
import sys
from pathlib import Path

from pureframe.eval.real_footage import (
    format_report,
    load_marks,
    run_real_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a real clip against hand-marked expected ranges."
    )
    parser.add_argument("video", type=Path, help="the clip to score")
    parser.add_argument("marks", type=Path, help="hand-marked ranges JSON")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="seconds of slack around each mark for hit counting (default 0.5)",
    )
    parser.add_argument("--profile", default=None, help="hardware profile override")
    parser.add_argument("--strictness", default=None, help="strictness preset")
    parser.add_argument("--content-type", default=None, help="content type profile")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="append the privacy-safe record here (default pureframe_real_eval.jsonl)",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="extra flags passed through to `pureframe plan` (e.g. --strict --no-clip)",
    )
    args = parser.parse_args()

    load_marks(args.marks)  # fail fast on a malformed marks file
    scores, summary = run_real_eval(
        args.video,
        args.marks,
        tolerance=args.tolerance,
        profile=args.profile,
        strictness=args.strictness,
        content_type=args.content_type,
        extra_args=args.extra,
        jsonl_path=args.jsonl,
    )
    print(format_report(scores, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
