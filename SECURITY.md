# Security & Privacy

PureFrame is designed with privacy as a first principle. Since it processes personal or potentially sensitive video files, its architecture ensures that your data never leaves your device.

## Privacy Guarantee

*   **Fully Offline:** PureFrame does not require an active internet connection to run detection or rendering.
*   **No Cloud Processing:** No frames, audio snippets, metadata, or filenames are ever sent to a remote server.
*   **No Telemetry:** PureFrame does not include any analytics, tracking, or telemetry mechanisms.

## Model Downloads

The only network activity occurs during the **first execution** when the AI model weights are downloaded from Hugging Face.
*   NudeNet weights
*   CLIP models
*   PANNs audio models

These models total approximately ~500MB. Once downloaded and cached in your local user directory (e.g., `~/.cache/huggingface` and standard app data directories), PureFrame will never make another network request.

## Storage and Temporary Files

*   **Temporary Audio:** During processing, audio tracks are extracted to temporary files on your local drive to run the audio classifier. These temporary files are deleted immediately after the pipeline finishes.
*   **Plan Files:** Censor plans (`.json`) are stored locally. They contain timecodes, scores, and bounding boxes, but do not contain actual video frames.
*   **Configuration:** Custom configuration files are stored locally as standard `toml` or `json`.

## Reporting Security Issues

If you find a security vulnerability, please open an issue on GitHub or reach out to the maintainers directly.
