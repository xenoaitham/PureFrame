# Parental guides: giving PureFrame known targets

Detection models are general; a parental guide is specific. For shows
where fans have documented the explicit scenes - IMDb's Parents Guide is
the famous example - you can hand PureFrame that knowledge as a small
marks file. Inside a marked window the planner lowers the nudity bar
(detection gets a second chance at content it might have missed in dark
or brief scenes), and in window mode a marked shot the detectors did not
flag is blurred whole - visibly categorized, reviewed by you before
anything renders.

Two honest limits, stated up front:

- **PureFrame never fetches guides from the network.** The offline
  identity is the product, and guide sites' terms generally forbid
  automated access. You author the marks file locally: open the guide
  page in your browser, save it, and note the timestamps into the JSON -
  a one-time, per-episode chore of a few minutes. Services that offer
  legitimate APIs for timestamped scene data work the same way: export,
  point `--guide` at the file.
- **Guides timestamp sex and nudity, not kissing or tense scenes.** The
  detectors keep doing that work. The guide extends coverage where the
  models are weakest; it does not replace them.

## The format

The same marks JSON the real-footage evaluator uses:

```json
{
  "ranges": [
    {"start": 812.0, "end": 845.0, "category": "NUDITY_EXPLICIT"},
    {"start": 1204.5, "end": 1211.0}
  ]
}
```

`start`/`end` are seconds into the episode; `category` is optional and
informational. Timestamps are usually scene-level - mark generously.

## Using it

```bash
# Hint mode (default): inside marked windows the nudity threshold is
# scaled by 0.7, so borderline detections in exactly those windows now
# flag. Detector-silent shots are left alone.
pureframe plan S04E03.mkv --guide goty-s04e03.json

# Window mode: marked shots the detectors still did not flag become
# whole-shot blur (GUIDE_BOX, shown blue in the editor) - review before
# applying, whitelist what you disagree with.
pureframe process S04E03.mkv --guide goty-s04e03.json --guide-mode window

# The guide bar is tunable:
pureframe plan S04E03.mkv --guide goty-s04e03.json --guide-factor 0.5
```

Guide settings fold into the checkpoint hash: changing the guide (or
dropping it) re-analyzes, like any other detection setting.

## Review is still the contract

Window-mode blur is a whole-shot decision from crowd-sourced evidence -
it is deliberately loud in the plan editor (blue on the timeline,
"Guide-marked" in the shot view) so you flip through every one before
applying. Whitelist a false mark and it stays whitelisted for that plan.
Hint-mode changes are ordinary detector verdicts with a lower bar, so
they show their threshold in the reasoning (`threshold: 0.39`) and you
can see exactly why something flagged.
