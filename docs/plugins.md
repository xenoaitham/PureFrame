# Plugins: custom detectors for PureFrame

Plugins add new censor categories - a different detector model, a
region detector for content PureFrame does not classify (weapons, gore,
logos), or something as plain as "censor whatever moves". They run
in-process next to the built-in detectors and their boxes render exactly
like nudity blur boxes: tracked, smoothed, reviewable in the plan editor.

Everything on this page is **experimental** while the first real plugins
land. The `detect_batch` signature is stable; `__init__` may gain
additive keyword arguments.

## Try the bundled example

`pureframe-plugins-examples` is maintained in this repository under
`examples/` and ships a motion-blob detector: it censors whatever moves,
no model required.

```bash
pip install pureframe-plugins-examples
pureframe plugins list
```

    Discovered PureFrame plugins
    Name        Categories       Labels       Default thresholds
    motionblob  MOTION_VISIBLE   motion_blob  motion_blob=0.45

Enable it for one run:

```bash
pureframe process input.mp4 --enable-plugin motionblob
```

The flag is repeatable (`--enable-plugin a --enable-plugin b`) and works
on `plan` and `process`. Plugins are never enabled implicitly: they
change verdicts, so a run uses exactly the plugins you name.

## Tuning a plugin's threshold

Plugin categories participate in the standard threshold controls, keyed
by the category name from `plugins list`:

```bash
# One-shot, JSON: lower the motion threshold to 0.3 (catch more motion)
pureframe process input.mp4 --enable-plugin motionblob \
  --thresholds '{"MOTION_VISIBLE": 0.3}'

# Or the strictness/content-type machinery applies as usual
pureframe process input.mp4 --enable-plugin motionblob --strictness high
```

The plugin's default (0.45 for motionblob) is the base; an explicit
override replaces it; the content-type multiplier and `--strict` still
apply on top, exactly as they do for nudity.

## What enabling a plugin changes

- Verdicts from plugin categories appear as `PLUGIN_BOX` with the
  plugin's own category recorded; they render per-frame blur boxes and
  never touch the audio classifier - a box provider is visual-only.
- Plugins can add censoring to a shot the built-ins would have passed;
  they can never downgrade a nudity or sexual-context verdict.
- The enabled plugin set is part of the cache key: re-running with the
  same plugins reuses cached analysis, changing the set re-analyzes.

## Plans stay portable

A plan that contains plugin boxes is ordinary JSON. You can edit it on a
machine without the plugin installed, apply it, share it - the boxes are
data, not code. The plugin is only needed to *generate* detections.

## Trust

Installing a plugin runs its code with the same permissions PureFrame
has. Treat plugin packages exactly like the pip packages you already
install: only install plugins you trust. There is no sandboxing, and the
API does not pretend otherwise.

## Writing your own

The developer contract, the fuse() integration points and the design
notes live in [docs/plugin-api.md](plugin-api.md). The motion-blob
detector in `examples/pureframe-plugins-examples/` is the reference
implementation - copy it, change the detection loop, ship.
