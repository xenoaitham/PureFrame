# Plugin API for custom detectors - design note

Status: **design**, not yet implemented. This note is the contract a future
implementation should satisfy; code changes land after review of this page.
The motivating use cases, in order of demand: a face detector for
kiss-scene mouth tracking on animation (where CLIP misfires), region
detectors for weapons/gore on non-English content, and site-specific
logo/watermark detection.

## What a detector is

Everything the pipeline needs from a detector already exists in the codebase
- a plugin is just a class honoring the same shape as `NudityDetector`:

```python
from pureframe.hardware import ProfileSettings
from pureframe.pipeline.detect.nudity import Detection


class MyDetector:
    """Detects one or more visual categories on BGR frames."""

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

2. **Provide a new box-provider category.** The post-#63 fuse/plan plumbing
   treats any flagged verdict with boxes as a blur source, so a plugin can
   declare e.g. `WEAPON_VISIBLE` (a new `Category` value) whose verdicts
   render per-frame boxes without touching the nudity logic. The design
   constraint carried from the start: **non-nudity categories are box
   providers** - they must never gate on the audio classifier, and their
   boxes flow through `densify_shot` + `smooth_detections` untouched.

Label→category maps are plain dicts shipped with the plugin:

```python
LABEL_CATEGORIES = {"pistol": "WEAPON_VISIBLE", "knife": "WEAPON_VISIBLE"}
LABEL_THRESHOLDS = {"pistol": 0.6, "knife": 0.6}
```

Per-label thresholds ride on the existing per-category threshold controls
(#69): `THRESHOLD_CATEGORIES` gains the plugin's category names, so
`--thresholds file.json` works uniformly. The `threshold_overrides`
validator whitelists category names - the plugin registry contributes its
categories at validation time.

## Registration

Entry points, so installation is the only step:

```toml
[project.entry-points."pureframe.plugins"]
weapons = "pureframe_weapons:MyDetector"
```

`pureframe.plugin_api.discover()` iterates the group and returns
`{name: (class, LABEL_CATEGORIES, LABEL_THRESHOLDS)}`. The CLI grows:

- `pureframe plugins list` - discovered plugins and their categories;
- `--enable-plugin weapons` on `process`/`plan` (repeatable).

No auto-enable: plugins change verdicts, so enabling is explicit. The GUI
gets a checkbox per discovered plugin later via the same discovery call.

## Pipeline integration points

1. **Construction** in `generate_plan`, next to `NudityDetector` - inside
   the existing `keep_models_loaded` lifecycle (construct eagerly, load
   lazily, `unload()` with the others).
2. **Per-shot inference** - plugin `detect_batch` runs on the same sampled
   keyframes the nudity detector already extracted (no extra decodes);
   results merge into `batch_dets` per frame.
3. **Fusion** - `fuse()` keeps its existing nudity branches untouched and
   appends one branch per box-provider category: any detection ≥ that
   category's effective threshold flags the shot `BLACK_BOX` with boxes.
   A plugin must never *downgrade* a verdict the nudity path already made.
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
- First-party example plugin `pureframe-plugins-examples` ships a tiny
  motion-blob detector (lavfi-friendly, CI-testable) as the reference.

## Implementation slices

1. `pureframe/plugin_api.py` - discovery + registry + contract tests.
2. `fuse()` box-provider branch + threshold-category extension.
3. CLI `plugins list` + `--enable-plugin`, config-hash extension.
4. Example plugin package + `docs/plugins.md` user page (this note becomes
   the developer reference).
