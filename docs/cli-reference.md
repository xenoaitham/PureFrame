# CLI Reference

PureFrame provides a command-line interface for all operations.

## Global Options

```
pureframe --version, -V    Show version and exit
pureframe --help            Show help message
```

---

## Commands

### `pureframe process`

**Full pipeline:** detect explicit content and render the censored video in one step.

```bash
pureframe process <INPUT> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `INPUT` | path | *required* | Path to input video file |
| `--output, -o` | path | `<input>.pureframe.<ext>` | Output video path |
| `--profile` | enum | auto-detected | Hardware profile: `cpu`, `low`, `medium`, `high` |
| `--threshold` | float | preset | Nudity threshold (0–1]; alias for `--threshold-nudity` |
| `--threshold-nudity` | float | preset | Nudity threshold; replaces the strictness preset's value |
| `--threshold-clip` | float | preset | CLIP scene threshold; replaces the preset's value |
| `--threshold-audio` | float | preset | Audio threshold; replaces the preset's value |
| `--thresholds` | path | none | JSON file with any of `"nudity"`, `"clip"`, `"audio"`; flags win over it |
| `--strict` | flag | false | Enable strict mode (lower thresholds) |
| `--no-clip` | flag | false | Skip CLIP scene classification |
| `--no-audio` | flag | false | Skip audio moaning detection |
| `--content-type` | enum | `live-action` | Content type: `live-action`, `animation`, `anime`, `low-light` |
| `--strictness` | enum | `medium` | Strictness: `low`, `medium`, `high`, `custom` |
| `--verbose, -v` | flag | false | Enable debug logging |

**Examples:**
```bash
# Basic usage
pureframe process movie.mp4

# Custom output and strict mode
pureframe process movie.mp4 --output clean_movie.mp4 --strict

# CPU-only, skip audio detection
pureframe process movie.mp4 --profile cpu --no-audio

# Anime content with high strictness
pureframe process anime.mkv --content-type anime --strictness high

# Per-category thresholds on top of a preset: only nudity is relaxed,
# clip/audio keep the high-strictness values (content-type and --strict
# multipliers still apply on top)
pureframe process movie.mp4 --strictness high --threshold-nudity 0.3

# Thresholds from a file, e.g. {"nudity": 0.4, "clip": 0.6}
pureframe process movie.mp4 --thresholds my_thresholds.json
```

Per-category thresholds replace the strictness preset's base value for that
one category; the other categories keep the preset, and the content-type
multiplier (and `--strict`) still scale the result. Previously `--threshold`
only had an effect combined with `--strictness custom`.

---

### `pureframe plan`

**Detection only:** generate a censor plan JSON without rendering. Useful for reviewing detections before applying.

```bash
pureframe plan <INPUT> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `INPUT` | path | *required* | Path to input video file |
| `--output, -o` | path | `<input>.censorplan.json` | Output plan path |
| `--profile` | enum | auto-detected | Hardware profile |
| `--threshold` | float | preset | Nudity threshold; alias for `--threshold-nudity` |
| `--threshold-nudity` | float | preset | Nudity threshold; replaces the preset's value |
| `--threshold-clip` | float | preset | CLIP scene threshold; replaces the preset's value |
| `--threshold-audio` | float | preset | Audio threshold; replaces the preset's value |
| `--thresholds` | path | none | JSON file with any of `"nudity"`, `"clip"`, `"audio"` |
| `--strict` | flag | false | Enable strict mode |
| `--no-clip` | flag | false | Skip CLIP classification |
| `--no-audio` | flag | false | Skip audio detection |
| `--content-type` | enum | `live-action` | Content type preset |
| `--strictness` | enum | `medium` | Strictness level |
| `--verbose, -v` | flag | false | Enable debug logging |

**Example:**
```bash
pureframe plan movie.mp4 --output my_plan.json
# Review the JSON, then apply with:
pureframe apply movie.mp4 my_plan.json
```

---

### `pureframe apply`

**Render only:** apply a previously generated censor plan to produce the censored video.

```bash
pureframe apply <INPUT> <PLAN> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `INPUT` | path | *required* | Path to original video file |
| `PLAN` | path | *required* | Path to `.censorplan.json` file |
| `--output, -o` | path | `<input>.pureframe.<ext>` | Output video path |
| `--verbose, -v` | flag | false | Enable debug logging |

**Example:**
```bash
pureframe apply movie.mp4 movie.mp4.censorplan.json --output clean.mp4
```

---

### `pureframe plan-edit`

**Edit plan:** open the censor plan JSON in your `$EDITOR` for manual editing, then validate on save.

```bash
pureframe plan-edit <PLAN>
```

**Example:**
```bash
export EDITOR=nano
pureframe plan-edit my_plan.json
```

---

### `pureframe plan-whitelist`

**Whitelist a shot:** mark a specific shot index as safe (action = NONE), skipping censoring for that shot.

```bash
pureframe plan-whitelist <PLAN> <SHOT_INDEX>
```

**Example:**
```bash
# Whitelist shot #3 (it was a false positive)
pureframe plan-whitelist my_plan.json 3
```

---

### `pureframe preview`

**Safe preview:** export flagged frame thumbnails as a contact sheet for review without watching the full video.

```bash
pureframe preview <PLAN> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `PLAN` | path | *required* | Path to `.censorplan.json` file |
| `--output, -o` | path | `<plan>.preview.html` | Output HTML report path |
| `--blur` | flag | true | Apply blur to flagged regions in thumbnails |

---

### `pureframe folder`

**Batch process:** process all video files in a directory.

```bash
pureframe folder <DIRECTORY> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `DIRECTORY` | path | *required* | Directory containing video files |
| `--output-dir, -o` | path | `<dir>/pureframe_output/` | Output directory |
| `--recursive, -r` | flag | false | Process subdirectories |
| `--workers` | int | 1 | Number of parallel workers |

---

### `pureframe jobs list`

**List jobs:** show all tracked processing jobs and their status.

```bash
pureframe jobs list
```

### `pureframe jobs cleanup`

**Cleanup jobs:** remove completed or failed job records.

```bash
pureframe jobs cleanup [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--all` | flag | false | Remove all jobs including pending ones |
| `--failed` | flag | false | Remove only failed jobs |

### `pureframe jobs resume`

**Resume job:** resume a previously interrupted job.

```bash
pureframe jobs resume <JOB_ID>
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PUREFRAME_PROFILE` | auto | Override hardware profile |
| `PUREFRAME_NUDITY_THRESHOLD` | 0.55 | Override nudity threshold |
| `PUREFRAME_LOG_LEVEL` | INFO | Log level: DEBUG, INFO, WARNING, ERROR |
| `EDITOR` | nano | Editor for `plan-edit` command |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (see stderr for details) |
| 2 | Invalid arguments |
