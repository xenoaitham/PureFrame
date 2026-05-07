# Example Commands

## Basic Usage

```bash
# Process a single video file (detect + render in one step)
pureframe process movie.mp4

# Specify output path
pureframe process movie.mp4 --output ~/Videos/clean_movie.mp4

# CPU-only mode (no GPU required)
pureframe process movie.mp4 --profile cpu
```

## Two-Step Workflow (Recommended)

The plan/apply workflow lets you review detections before rendering:

```bash
# Step 1: Generate detection plan
pureframe plan movie.mp4

# Step 2: Review the plan (opens in your editor)
pureframe plan-edit movie.mp4.censorplan.json

# Step 3: Whitelist false positives
pureframe plan-whitelist movie.mp4.censorplan.json 3
pureframe plan-whitelist movie.mp4.censorplan.json 7

# Step 4: Apply the reviewed plan
pureframe apply movie.mp4 movie.mp4.censorplan.json
```

## Content Types

```bash
# Live-action movies and TV (default)
pureframe process movie.mp4 --content-type live-action

# Animated content (higher thresholds to reduce false positives)
pureframe process cartoon.mp4 --content-type animation

# Anime (different threshold profiles for anime art style)
pureframe process anime.mkv --content-type anime

# Dark/low-light scenes (adjusted detection sensitivity)
pureframe process horror_movie.mp4 --content-type low-light
```

## Strictness Levels

```bash
# Low: minimal censoring, only high-confidence explicit content
pureframe process movie.mp4 --strictness low

# Medium: balanced detection (default)
pureframe process movie.mp4 --strictness medium

# High: aggressive censoring, catches more edge cases
pureframe process movie.mp4 --strictness high

# Custom: set your own threshold
pureframe process movie.mp4 --threshold 0.35
```

## Speed Optimization

```bash
# Skip audio detection (faster, no moaning detection)
pureframe process movie.mp4 --no-audio

# Skip CLIP scene classification (faster, nudity-only)
pureframe process movie.mp4 --no-clip

# Skip both (fastest, nudity detection only)
pureframe process movie.mp4 --no-audio --no-clip

# Force high-performance GPU profile
pureframe process movie.mp4 --profile high
```

## Batch Processing

```bash
# Process all videos in a folder
pureframe folder ~/Movies/

# Recursive processing with 2 parallel workers
pureframe folder ~/Movies/ --recursive --workers 2

# Custom output directory
pureframe folder ~/Movies/ --output-dir ~/CleanMovies/
```

## Job Management

```bash
# List all processing jobs
pureframe jobs list

# Resume an interrupted job
pureframe jobs resume <job-id>

# Clean up completed jobs
pureframe jobs cleanup

# Clean up all jobs including pending
pureframe jobs cleanup --all
```

## Safe Preview

```bash
# Generate a preview contact sheet (HTML report)
pureframe preview movie.mp4.censorplan.json

# Preview with blurred flagged regions
pureframe preview movie.mp4.censorplan.json --blur

# Custom output path
pureframe preview movie.mp4.censorplan.json --output review.html
```
