# Architecture

PureFrame is a modular video processing pipeline designed to identify and obscure explicit content locally, without cutting scenes or altering audio timing.

## Core Pipeline

The pipeline is split into three major phases: **Probe**, **Detect**, and **Render**.

### 1. Probe & Sample
- **`probe_video()`**: Uses FFmpeg to extract video metadata (fps, duration, resolution, codecs).
- **`sample_keyframes()`**: Extracts keyframes using SceneDetect to avoid processing every single frame. This saves massive amounts of compute time.

### 2. Detection Engine
The detection engine is a fusion of multiple specialized models:

- **Visual Detection (NudeNet):** Identifies exposed explicit body parts using bounding boxes.
- **Contextual Scene Classification (CLIP):** Classifies the overall scene context (e.g., implied sexual activity, kissing) when bounding boxes alone are insufficient.
- **Audio Classification (PANNs):** Analyzes the audio track for sexual or moaning sounds to disambiguate "safe" flesh (e.g., a wrestling match or a beach) from explicit scenes.
- **Face Detection (Haar/OpenCV):** Helps identify areas to *avoid* blurring or to adjust confidence if faces are primary.

### 3. Densification & Smoothing
- **`densify_shot()`**: Once a keyframe flags positive, the pipeline extracts denser frames around that timestamp to get precise tracking data.
- **`smooth_detections()`**: Applies temporal smoothing to bounding boxes. This prevents the blur from flickering or jumping erratically between frames.

### 4. Render
- **`apply_censoring()`**: Uses FFmpeg filters (`boxblur`) combined with complex filtergraphs to selectively blur the regions identified by the smoothing phase. The original audio track is mapped back directly (stream copy) so sync is perfectly maintained.

## Configuration & State

- **`Config`**: Defines thresholds, hardware profiles (CPU, LOW, MEDIUM, HIGH), and feature flags (e.g., `disable_audio`).
- **Batch Processing (`CheckpointStore`)**: For processing entire directories, a SQLite database tracks state (`PENDING`, `DETECTING`, `RENDERING`, `DONE`). This allows resuming interrupted jobs automatically.
