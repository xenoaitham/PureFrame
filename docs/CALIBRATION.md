# Confidence Calibration Guide

> **TL;DR**: Use `--strictness high` for family viewing, `--strictness medium` for general use, `--strictness low` for minimal false positives.

## Threshold Presets

PureFrame uses confidence thresholds to decide when a detection is "real" enough to trigger censoring. Higher thresholds mean fewer false positives but risk missing actual explicit content.

### Built-in Strictness Levels

| Strictness | Nudity | Genitalia | Buttocks | Use Case |
|-----------|--------|-----------|----------|----------|
| **High** | 0.30 | 0.25 | 0.35 | Family movie night - catches everything, may flag swimwear |
| **Medium** | 0.45 | 0.40 | 0.50 | General use - balanced precision/recall |
| **Low** | 0.60 | 0.55 | 0.65 | Minimal intervention - only flags obvious nudity |
| **Custom** | User-defined via `--threshold` | | | Full control |

### Content-Type Modifiers

Each content type applies a multiplier to the base thresholds:

| Content Type | Multiplier | Effect |
|-------------|------------|--------|
| `live-action` | 1.0x | Standard thresholds |
| `animation` | 0.85x | Lower thresholds (animated skin tones are harder to detect) |
| `anime` | 0.80x | Even lower (anime has unique skin rendering) |
| `low-light` | 0.90x | Slightly lower (dark scenes reduce detection confidence) |

### Recommended Configurations

#### Family Movie Night
```bash
pureframe process movie.mp4 --strictness high --content-type live-action
```
- Catches virtually all explicit content
- May flag beach/pool scenes as borderline
- Expected false positive rate: ~5-8%
- Expected false negative rate: <1%

#### Anime Watching Session
```bash
pureframe process anime.mkv --strictness high --content-type anime
```
- Anime-specific thresholds handle stylized skin tones
- Catches ecchi/fanservice content
- Hot springs episodes may trigger (use `plan-whitelist` to review)
- Expected false positive rate: ~10-15%
- Expected false negative rate: <2%

#### TV Series Binge
```bash
pureframe process episode.mp4 --strictness medium --content-type live-action
```
- Balanced for shows like Game of Thrones, Euphoria
- Catches full nudity and most partial nudity
- Brief kissing/embrace scenes left uncensored
- Expected false positive rate: ~2-4%
- Expected false negative rate: ~3-5%

#### Documentary / Educational
```bash
pureframe process documentary.mp4 --strictness low --content-type live-action
```
- Only flags clearly explicit content
- Medical/anatomical content mostly left uncensored
- Art with classical nudity generally untouched
- Expected false positive rate: <1%
- Expected false negative rate: ~5-10%

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
| **0.30-0.49** | Weak signal - could be skin, swimwear, or artistic nudity | Only censor at high strictness |
| **<0.30** | Background noise - very unlikely to be explicit | Almost never censor |

## Running the Benchmark

To see how thresholds perform across different content types:

```bash
pureframe evaluate --threshold 0.5 --output eval_report.json
```

This runs 50 synthetic test scenarios and shows precision/recall at multiple threshold levels.
