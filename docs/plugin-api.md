# Plugin API for custom detectors - developer reference

Status: **implemented** across four merged slices. The motivating use
cases, in order of demand: a face detector for kiss-scene mouth tracking
on animation (where CLIP misfires), region detectors for weapons/gore on
non-English content, and site-specific logo/watermark detection. The
user-facing page is [plugins.md](plugins.md).

## What a detector is

Everything the pipeline needs from a detector already exists in the
codebase - a plugin is just a class honoring the same shape as
`NudityDetector`, with its label mapping declared on the class:

```python
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.detect.nudity import Detection


class MyDetector:
    """Detects one or more visual categories on BGR frames."""

    label_categories = {"pistol": "WEAPON_VISIBLE"}
    label_thresholds = {"pistol": 0.6}

    def __init__(self, settings: ProfileSettings):
        # settings carries detection_resolution, batch size, fp16 flag,
        # keep_models_loaded and the ONNX providers. Load heavy models here
        # lazily or in first detect_batch (see NudityDetector's resident
        # session pattern) - the pipeline may construct the class even when
        # the category never fires.
        ...

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """One Detection list per input frame, in order."""
        ...

    def unload(self) -> None:
        """Drop model memory; called when the profile does not keep models loaded."""
        ...
```

`Detection` is the shared currency: `label`, `score` in [0, 1], and `box`
as `(x1, y1, x2, y2)` in **native frame pixels** (detectors may run
downscaled internally, but report native - the overlay rescales per config
from there).

## Declaring categories

A plugin maps its raw labels to the plan's semantics. Two mechanisms, in
order of preference:

1. **Reuse an existing `Category`.** Any label mapped to `NUDITY_EXPLICIT`
   or another explicit category participates in the existing fuse branches
   exactly like NudeNet labels (including threshold gating and
   densify/tracking). This is for detectors that see the same kind of
   content with different eyes.

2. **Provide a new box-provider category.** Any flagged verdict with boxes
   is a blur source, so a plugin can declare e.g. `WEAPON_VISIBLE`, a
   category name of its own whose verdicts render per-frame boxes without
   touching the nudity logic. The design constraint carried from the
   start: **non-nudity categories are box providers** - they must never
   gate on the audio classifier, and their boxes flow through densify +
   `smooth_detections` untouched.

   Plans keep a single closed `Category.PLUGIN_BOX` value for all plugin
   verdicts, with the plugin's own category name recorded in the optional
   `ShotVerdict.plugin_category` field. One closed member plus an optional
   field keeps plans portable in both directions: old versions load new
   plans (they drop the field), and a plan edited on a machine without
   the plugin still renders, because boxes are data, not code.

Label→category maps are plain dicts declared on the plugin class:

```python
LABEL_CATEGORIES = {"pistol": "WEAPON_VISIBLE", "knife": "WEAPON_VISIBLE"}
LABEL_THRESHOLDS = {"pistol": 0.6, "knife": 0.6}
```

Per-label thresholds ride on the existing per-category threshold controls:
a category's default is the minimum over its labels' declared thresholds
(the lower one keeps the category trigger-ready), `threshold_overrides`
and `--thresholds file.json` accept the plugin's category names the same
way they accept `nudity`/`clip`/`audio` (the validator whitelists
categories contributed by the discovered registry), and the content-type
multiplier plus `--strict` apply on top exactly as for built-ins.

## Registration

Entry points, so installation is the only step:

```toml
[project.entry-points."pureframe.plugins"]
weapons = "pureframe_weapons:MyDetector"
```

`pureframe.plugin_api.discover()` iterates the group and returns
`{name: PluginRegistration}` - a frozen record of the class, its
`label_categories` and its `label_thresholds`, with `threshold_for()`,
`category_names()` and `category_threshold_bases()` helpers. Plugins that
fail to import or break the contract are logged and skipped, so one bad
wheel cannot take the CLI down; nothing is constructed at discovery time.
The CLI grows:

- `pureframe plugins list` - discovered plugins and their categories;
- `--enable-plugin weapons` on `process`/`plan` (repeatable).

No auto-enable: plugins change verdicts, so enabling is explicit. The
enabled set is part of `config_hash` (only when non-empty, so pre-plugin
checkpoints stay valid). The GUI gets a checkbox per discovered plugin
later via the same discovery call.

## Pipeline integration points

1. **Construction** in `generate_plan`, next to `NudityDetector` - inside
   the existing `keep_models_loaded` lifecycle (construct eagerly, load
   lazily, `unload()` with the others).
2. **Per-shot inference** - plugin `detect_batch` runs on the same sampled
   keyframes the nudity detector already extracted (no extra decodes);
   results merge into `batch_dets` per frame.
3. **Fusion** - `fuse()` keeps its nudity and sexual-context branches
   untouched and adds one shared plugin branch after them: any detection at
   or above the category's effective threshold flags the shot `BLACK_BOX`
   with boxes. Placement is deliberate - after the nudity/sexual-context
   branches so a plugin can never *downgrade* a verdict those paths made,
   and before the kiss branches so plugin evidence still censors a shot
   that would otherwise render at `KISS_LIGHT` with no censoring.
4. **Densify + tracking** - the existing densify pass extends to plugin
   detections whose shot was flagged by that plugin; boxes flow through the
   unchanged `smooth_detections` pipeline.
5. **Eval parity** - the CI gate pins NudeNet's detection signature.
   Plugin code cannot drift it (plugins only run when enabled), and a
   `tests/test_plugin_api.py` contract suite runs a fixture plugin through
   the fused pipeline so the integration surface stays honest.

## Checkpoints and plans

`config_hash` gains the enabled-plugin set + plugin config, so cache
semantics (#72) hold. Plans serialize plugin detections as ordinary
boxes/categories - a plan edited or applied on a machine without the plugin
still renders (boxes are data, not code).

## Versioning and safety

- The API is **experimental** until two real plugins exist; expect the
  `detect_batch` signature to be stable but reserve the right to extend
  `__init__` kwargs (additive only).
- Plugins run in-process with full trust - the entry-point group is
  documented as "install plugins you trust", same trust level as the pip
  packages PureFrame already imports. No sandboxing is attempted (and not
  pretended).
- First-party example plugin `pureframe-plugins-examples` (under
  `examples/`) ships a tiny motion-blob detector (lavfi-friendly,
  CI-testable) as the reference.

## Implementation status

All four slices are merged: `pureframe/plugin_api.py` (discovery +
registry + contract tests), the fuse() box-provider branch with
per-category threshold integration, the CLI `plugins list` +
`--enable-plugin` + config-hash extension, and the example plugin package
with `docs/plugins.md` as the user page.
