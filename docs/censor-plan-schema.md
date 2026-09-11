# Censor Plan Schema

PureFrame uses JSON censor plans as the intermediate format between detection and rendering. This allows human review and editing before any video modification.

## File Format

Censor plans use the `.censorplan.json` extension.

## Schema

```json
{
  "pureframe_version": "0.1.0b4",
  "plan_version": 1,
  "input_metadata": {
    "filename": "movie.mp4",
    "duration_seconds": 5400.0,
    "fps": 24.0,
    "width": 1920,
    "height": 1080,
    "total_frames": 129600,
    "codec": "h264",
    "audio_codec": "aac"
  },
  "config_snapshot": {
    "input_path": "/path/to/movie.mp4",
    "output_path": "/path/to/movie.pureframe.mp4",
    "profile": "CPU",
    "nudity_threshold": 0.55,
    "box_padding_pct": 0.12,
    "box_color": [0, 0, 0],
    "output_codec": "h264",
    "output_crf": 20,
    "strict": false,
    "no_clip": false,
    "no_audio": false
  },
  "shots": [
    {
      "index": 0,
      "start_frame": 0,
      "end_frame": 240,
      "start_time": 0.0,
      "end_time": 10.0
    }
  ],
  "verdicts": [
    {
      "shot_index": 0,
      "action": "BLACK_BOX",
      "category": "NUDITY",
      "confidence": 0.87,
      "reasoning": "FEMALE_BREAST_EXPOSED detected with score 0.87"
    }
  ],
  "total_censored_frames": 240,
  "total_blur_frames": 0,
  "generated_at": "2026-05-07T12:00:00Z"
}
```

## Field Reference

### Root Fields

| Field | Type | Description |
|-------|------|-------------|
| `pureframe_version` | string | PureFrame version that generated this plan |
| `plan_version` | int | Schema version (currently 1) |
| `input_metadata` | object | Video file metadata |
| `config_snapshot` | object | Configuration used during detection |
| `shots` | array | List of detected scene shots |
| `verdicts` | array | Censoring decision per shot |
| `total_censored_frames` | int | Total frames marked for censoring |
| `total_blur_frames` | int | Total frames marked for blur |
| `generated_at` | string | ISO 8601 timestamp |

### Shot Object

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Shot index (0-based) |
| `start_frame` | int | First frame of this shot |
| `end_frame` | int | Last frame of this shot |
| `start_time` | float | Start time in seconds |
| `end_time` | float | End time in seconds |

### Verdict Object

| Field | Type | Description |
|-------|------|-------------|
| `shot_index` | int | References `shots[].index` |
| `action` | enum | `NONE`, `BLACK_BOX`, `FULL_FRAME_BLUR` |
| `category` | enum | Detection category (see below) |
| `confidence` | float | Detection confidence (0.0–1.0) |
| `reasoning` | string | Human-readable explanation |

### Action Values

| Value | Effect |
|-------|--------|
| `NONE` | No censoring (safe or whitelisted) |
| `BLACK_BOX` | Black box overlay on detected regions |
| `FULL_FRAME_BLUR` | Full-frame Gaussian blur |

### Category Values

| Value | Description |
|-------|-------------|
| `NONE` | No explicit content detected |
| `NUDITY` | Exposed nudity detected |
| `SEXUAL_ACTIVITY` | Sexual activity detected via scene classification |
| `AUDIO_MOAN` | Moaning/sexual audio detected |
| `KISSING` | Intimate kissing detected |
| `PARTIAL_NUDITY` | Partial nudity or suggestive content |
| `COMPOSITE` | Multiple detection signals fused |

## Editing Plans

You can edit any verdict manually:

```bash
# Open in editor
pureframe plan-edit plan.json

# Whitelist a specific shot
pureframe plan-whitelist plan.json 5

# Or edit the JSON directly
# Change "action": "BLACK_BOX" to "action": "NONE" to whitelist
```

## Sharing Plans

Censor plans are portable. You can share a plan with others who have the same video file. The plan does not contain any video data - only metadata and censoring decisions.

See [Plan Sharing Guide](plan-sharing.md) for details.
