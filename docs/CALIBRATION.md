# Confidence Calibration Guide

> **TL;DR**: Use `--strictness high` for family viewing, `--strictness medium` for general use, `--strictness low` for minimal false positives. Beach/pool and gym/sports scenes get an automatic bar raise; tune or disable it with `--scene-context-factor`.

## Threshold Presets

PureFrame uses confidence thresholds to decide when a detection is "real" enough to trigger censoring. Higher thresholds mean fewer false positives but risk missing actual explicit content. There is one nudity threshold, one CLIP scene threshold and one audio threshold; every preset sets all three.

### Built-in Strictness Levels

| Strictness | Nudity | CLIP scene | Audio | Use Case |
|-----------|--------|-----------|----------|----------|
| **High** | 0.35 | 0.35 | 0.40 | Family movie night - catches the most, accepts more false positives |
| **Medium** (default) | 0.55 | 0.50 | 0.60 | General use - balanced precision/recall |
| **Low** | 0.75 | 0.70 | 0.80 | Minimal intervention - only flags obvious nudity |
| **Custom** | via `--threshold` / `--threshold-nudity` / `--threshold-clip` / `--threshold-audio` or a `--thresholds` file | | | Full control |

A per-category flag or file entry replaces that category's preset value; the others keep it. The content-type multiplier and `--strict` (an additional 0.85x on every threshold) apply on top either way.

### Content-Type Modifiers

Each content type multiplies all three base thresholds (capped at 0.99):

| Content Type | Multiplier | Effect |
|-------------|------------|--------|
| `live-action` | 1.0x | Standard thresholds |
| `animation` | 1.3x | Higher bar (drawn skin tones trigger the detector more easily) |
| `anime` | 1.4x | Highest bar (anime styling over-triggers) |
| `low-light` | 0.85x | Lower bar, plus the automatic dark-frame normalization described below |
| `art` | 1.5x | Museums and galleries - classical paintings and statues stop flagging |
| `medical` | 1.4x | Clinical footage - surgery and anatomy content stops flagging |

Since 0.2.5, dark scenes get help automatically regardless of content type: when a frame measures dark (mean luma under 70) or near-grayscale with a compressed histogram, the detector sees a range-restored copy while the render keeps the original pixels. Declaring `--content-type low-light` keeps the 0.85x multiplier for content that stays dark in ways the frame-level measurement misses.

### Scene-Context Gate (beach/pool, gym/sports, museum/gallery, medical)

Swimwear, skin-tight clothing, shirtless athletes, classical paintings and surgical footage produce real detector signals - they are just not nudity. CLIP classifies every shot's context; when one of those four benign contexts reads confident (score at or above 0.60) and neither sexual context category is near its own threshold, the shot's nudity threshold scales by `--scene-context-factor` (default 1.4, valid range 1.0 to 2.0; 1.0 disables the gate). The art and medical content types layer on top: they raise the whole run's bar, and the museum/medical scene gate refines it per shot.

The gate can only raise a bar, never lower one, and confident sexual context switches it off entirely - a sex scene at the beach, or real nudity in the same documentary, keeps the strict threshold. The factor is part of the plan's config hash: changing it re-analyzes.

The parental-guide feature composes with it: inside a `--guide` window the guide factor multiplies on top of whatever the context gate decided.

## Recommended Configurations

#### Family Movie Night
```bash
pureframe process movie.mp4 --strictness high --content-type live-action
```
- Catches the most explicit content
- Swimwear scenes at the beach or pool are gated by the scene-context factor; pass `--scene-context-factor 1.0` if you would rather over-flag them
- Borderline cases land in the plan for review before anything renders

#### Anime Watching Session
```bash
pureframe process anime.mkv --strictness high --content-type anime
```
- Anime-specific thresholds handle stylized skin tones
- Catches ecchi/fanservice content
- Hot springs episodes may still trigger (use `plan-whitelist` to review)

#### TV Series Binge
```bash
pureframe process episode.mp4 --strictness medium --content-type live-action
```
- Balanced for shows with intermittent explicit scenes
- Catches full nudity and most partial nudity
- Brief kissing/embrace scenes left uncensored

#### Beach / Sports Footage
```bash
pureframe process surf_day.mp4 --strictness high
```
- The scene-context gate suppresses swimwear flags on confident beach/pool shots
- Verdict reasoning and the plan editor show what flagged; whitelist anything the gate missed
- Real nudity in the same footage still flags at the strictness preset

#### Documentary / Art
```bash
pureframe process documentary.mp4 --strictness low --content-type live-action
```
- Only flags clearly explicit content
- Art with classical nudity generally untouched

#### Art Documentary / Museum Footage
```bash
pureframe process gallery_tour.mp4 --content-type art
```
- The 1.5x bar plus the museum scene gate keep Renaissance paintings and statues unflagged
- Real explicit content in the same film still flags wherever the sexual context is confident

#### Medical / Educational
```bash
pureframe process surgery_lecture.mp4 --content-type medical
```
- The 1.4x bar plus the medical scene gate keep surgical and anatomical footage unflagged
- Same recall guarantee: explicit scenes elsewhere in the recording flag normally

## Threshold Tuning Workflow

If the defaults don't work for your content:

### 1. Generate a plan first
```bash
pureframe plan video.mp4 --output plan.json --strictness medium
```

### 2. Preview what gets flagged
```bash
pureframe preview plan.json --output preview.html
```

### 3. Review and whitelist false positives
```bash
pureframe plan-whitelist plan.json --indices 3,7,12
```

### 4. Apply the refined plan
```bash
pureframe apply plan.json
```

### 5. If too many false positives, raise threshold
```bash
pureframe plan video.mp4 --threshold 0.65
```

### 6. If missing content, lower threshold
```bash
pureframe plan video.mp4 --threshold 0.35
```

## Understanding Confidence Scores

| Score Range | Interpretation | Action |
|-------------|---------------|--------|
| **0.90+** | Model is very confident - almost certainly explicit | Always censor |
| **0.70-0.89** | Strong signal - very likely explicit | Censor at medium/high strictness |
| **0.50-0.69** | Moderate signal - possible explicit content | Censor at high strictness; review at medium |
| **0.30-0.49** | Weak signal - could be skin, swimwear, or artistic nudity | Only censor at high strictness; the second-pass rescan re-examines shots in this band |
| **<0.30** | Background noise - very unlikely to be explicit | Almost never censor |

## Running the Benchmark

To see how thresholds perform across different content types:

```bash
pureframe evaluate --threshold 0.5 --output eval_report.json
```

This runs the synthetic test corpus (52 scenarios across 8 content genres) and shows precision/recall at multiple threshold levels. The committed `eval-baseline.json` pins the detector's exact behavior on every scenario; `scripts/check_eval_parity.py` fails if any score drifts.
