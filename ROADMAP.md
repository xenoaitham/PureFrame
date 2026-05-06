# Roadmap

PureFrame is currently in **v0.1.0-beta**. The core detection and blurring pipelines are functional. Below is the planned roadmap for upcoming features.

## 🏃 Next Up (v0.2.x)

- [ ] **Hardware Encoding:** Support for NVENC, QSV, and VideoToolbox during the FFmpeg render phase for drastically faster output.
- [ ] **Content-Type Presets:** Add a `--content-type` flag (e.g., `live-action`, `anime`, `cartoon`) to automatically adjust detection thresholds, as animation often triggers false positives on standard models.
- [ ] **GUI Enhancements:** Move the desktop app out of experimental status, complete the visual scrubbing timeline, and allow 1-click whitelisting.

## 🛠 Planned (v0.3.x & Beyond)

- [ ] **Plan Sharing Hub:** A secure, legal way to share `.censorplan.json` files so users don't have to run the heavy detection pipeline themselves.
- [ ] **Advanced Audio Filtering:** Muting or bleeping explicit words using local whisper models (currently only audio context is used for visual disambiguation).
- [ ] **Subtitles Support:** Automatically carry over or adjust subtitle tracks (`.srt`, `.ass`) into the final output.
- [ ] **More Blurring Styles:** Pixelation, solid black bars, or advanced inpainting instead of standard Gaussian blur.
