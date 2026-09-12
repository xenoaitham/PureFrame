# Real-footage evaluation

How PureFrame actually does on real movies - not the synthetic corpus
the CI gate pins, but the content you play. One clip at a time, scored
against marks you make by hand, with numbers that accumulate locally
without ever leaving your machine.

This lives outside the automated test suite on purpose: the footage is
yours, the marks are yours, and nothing copyrighted ships in the repo.
The harness only stores hashes and metric lines.

## The workflow

1. **Pick one clip.** A few minutes of representative content beats a
   whole movie for the first pass; you will be marking it by hand.

2. **Mark the ranges that should be censored.** Create a small JSON
   file next to your notes (wherever you like - it is never uploaded):

   ```json
   {
     "ranges": [
       {"start": 12.5, "end": 18.0, "category": "NUDITY_EXPLICIT"},
       {"start": 40.0, "end": 41.5, "category": "KISS_INTENSE"},
       {"start": 77.0, "end": 79.0}
     ]
   }
   ```

   - `start`/`end` are seconds into the clip. Mark generously: the
     union of everything a careful human would want covered.
   - `category` is optional. Without it the mark counts against every
     category ("ANY"); with it, only against that category's score.
     Category names are the plan's: `NUDITY_EXPLICIT`,
     `SEXUAL_ACT_VISIBLE`, `SEXUAL_CONTEXT_NO_NUDITY`, `KISS_INTENSE`,
     `KISS_LIGHT`, `VIOLENCE_GORE`, or a plugin category such as
     `MOTION_VISIBLE`.

   An `ffprobe`-free way to find timestamps: open the clip in any
   player with a frame-accurate seek, note the cut points, round
   outward.

3. **Score it:**

   ```bash
   uv run python scripts/score_real_footage.py clip.mkv marks.json
   ```

   This runs the real `pureframe plan` on the clip (models download on
   first use, as always), compares the flagged shot ranges against
   your marks, prints a report, and appends one JSONL line to
   `pureframe_real_eval.jsonl` in the working directory.

4. **Iterate.** Re-run with different settings and compare:

   ```bash
   uv run python scripts/score_real_footage.py clip.mkv marks.json \
     --profile CPU --strictness high
   uv run python scripts/score_real_footage.py clip.mkv marks.json \
     --profile CPU --content-type animation
   uv run python scripts/score_real_footage.py clip.mkv marks.json \
     --enable-plugin motionblob        # pass-through to pureframe plan
   ```

## Reading the report

Per category (and overall through "ANY" marks), time-based precision
and recall over shot ranges:

- **expected** - total seconds you marked.
- **flagged** - total seconds the planner flagged (shots are flagged
  whole; the renderer then blurs per-frame boxes inside them).
- **hit** - flagged time covered by a mark.
- **precision** - hit / flagged: how much of what the tool flagged you
  actually wanted censored. Low precision = false positives (benign
  shots getting flagged).
- **recall** - hit / expected: how much of what you marked the tool
  caught. Low recall = false negatives (content slipping through).
- **missed / extra** - the concrete ranges behind the numbers, for
  eyeballing what happened at those timestamps.

`--tolerance` (default 0.5 s) widens each mark before hit counting, so
a shot boundary that snaps a second past your mark is not punished as
a miss; `--tolerance 0` turns this off. Reported expected seconds
always stay the marks as written.

Granularity is shot-level by design: "flagged this shot but my mark
only covered half" is precisely the miss this scoring surfaces. Frame
level truth is a different (much heavier) marking exercise.

## What lands on disk

```
pureframe_real_eval.jsonl   one line per scoring run
```

Each line carries: the clip's SHA-256 + duration/resolution/fps/codec
(never a path or filename), the marks file's SHA-256, the settings used,
the per-category metrics, and the environment block (same shape as
`bench --real`). Timestamps of the marks themselves are not stored in
the JSONL - the numbers and hashes are. Delete the file whenever you
like; nothing else in the tree depends on it.
