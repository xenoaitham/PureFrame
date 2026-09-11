# Detection Evaluation Report

## Methodology

PureFrame's detection pipeline combines three independent classifiers:

1. **NudeNet** (ONNX) - localized nudity/explicit region detection
2. **CLIP** (ViT-B/32) - scene-level semantic classification
3. **PANNs** (CNN14) - audio event classification (moaning detection)

Results are fused via a weighted voting system with configurable thresholds.

---

## Detection Categories

| Category | Detector | Default Threshold | Labels |
|----------|----------|-------------------|--------|
| Nudity - exposed breasts | NudeNet | 0.55 | `FEMALE_BREAST_EXPOSED` |
| Nudity - exposed genitalia | NudeNet | 0.55 | `FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED` |
| Nudity - exposed buttocks | NudeNet | 0.55 | `BUTTOCKS_EXPOSED` |
| Nudity - exposed anus | NudeNet | 0.55 | `ANUS_EXPOSED` |
| Sexual activity | CLIP | 0.50 | Scene-level semantic match |
| Audio - moaning | PANNs | 0.60 | Audio event classification |

---

## Confidence Calibration Guide

### Threshold Presets

| Preset | Nudity | CLIP | Audio | Best For |
|--------|--------|------|-------|----------|
| **Low** (permissive) | 0.75 | 0.70 | 0.80 | Minimal false positives. May miss some nudity. |
| **Medium** (balanced) | 0.55 | 0.50 | 0.60 | Default. Good balance for most content. |
| **High** (strict) | 0.35 | 0.35 | 0.40 | Aggressive. Higher false positive rate. Best for family viewing. |
| **Custom** | User-defined | User-defined | User-defined | Set `--threshold` manually. |

### Choosing a Threshold

```
More false positives ←----------→ More missed detections

  0.25    0.35    0.45    0.55    0.65    0.75    0.85
   |       |       |       |       |       |       |
 Ultra   Strict  High   Default  Low    Minimal  Off
 strict                 (0.55)
```

**Recommendation:** Start with `medium` (default). If you see false positives on swimwear/skin, increase to `low`. If content slips through, decrease to `high`.

---

## Content-Type Profiles

Different content types have different visual characteristics that affect detection accuracy.

### Live Action (default)

Standard settings. Works well for most movies and TV shows.

| Aspect | Behavior |
|--------|----------|
| Detection resolution | Configured by hardware profile |
| False positive sources | Swimwear, skin-tone backgrounds, tight clothing |
| Recommended threshold | 0.55 (medium) |

### Animation

Animated content has different visual characteristics than live action. Colors are more saturated, shapes are simpler, and skin tones are uniform.

| Aspect | Behavior |
|--------|----------|
| Detection resolution | Same as hardware profile |
| Threshold multiplier | 1.3x (higher threshold to reduce FPs) |
| False positive sources | Character designs, bathhouse scenes, comedy nudity |
| Recommended threshold | 0.65–0.75 |

### Anime

Japanese anime has distinct art styles that can trigger false positives on character designs.

| Aspect | Behavior |
|--------|----------|
| Threshold multiplier | 1.4x |
| False positive sources | Fan service, beach episodes, transformation sequences |
| Recommended threshold | 0.70–0.80 |

### Low-Light

Dark scenes reduce detection confidence. We compensate by lowering thresholds slightly.

| Aspect | Behavior |
|--------|----------|
| Threshold multiplier | 0.85x |
| Confidence adjustment | +10% to compensate for reduced visibility |
| Recommended threshold | 0.45–0.50 |

---

## Known Limitations

### Expected False Positives

| Scenario | Why | Mitigation |
|----------|-----|-----------|
| Swimwear / bikini scenes | Skin exposure triggers nudity detector | Use `low` strictness or whitelist |
| Medical/documentary content | Clinical nudity | Use `low` strictness, review plan |
| Tight/revealing clothing | Skin-tone regions | Increase threshold |
| Artistic nudity (paintings, sculptures) | Detector doesn't distinguish art vs. real | Whitelist specific shots |
| Baby/child bathing scenes | Skin detection is not age-aware | Always review plan manually |
| Dark-skinned subjects in dark lighting | Low contrast can cause misdetection | Use `low-light` content type |

### Expected False Negatives

| Scenario | Why | Mitigation |
|----------|-----|-----------|
| Very dark scenes | Low visibility reduces detection confidence | Use `low-light` content type or `high` strictness |
| Very brief flashes (< 3 frames) | May fall between keyframe samples | Increase `densify_every_n_frames` |
| Obscured/partially covered nudity | Detector requires visible explicit regions | Use CLIP scene classification as backup |
| Animated nudity in non-standard art styles | NudeNet trained primarily on photographic content | Use `anime` content type |

### Important Disclaimers

- PureFrame is a **tool**, not a guarantee. Always review censor plans before sharing output.
- Detection accuracy depends heavily on video quality, lighting, and content type.
- The plan/apply workflow exists specifically for human review - use it.
- No automated system achieves 100% accuracy on content moderation.

---

## Reporting Detection Issues

If you encounter a false positive or false negative:

1. Note the video timestamp and shot index
2. Note the content type and threshold used
3. Open a GitHub issue with:
   - Content type (e.g., "live-action drama")
   - Description of the scene (no explicit screenshots)
   - PureFrame version and settings
   - Whether it was a false positive or false negative

We use these reports to improve detection quality in future releases.
