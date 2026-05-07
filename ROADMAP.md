# Roadmap

## v0.1.x - Current (Beta)

- [x] CLI plan/apply/process workflow
- [x] NudeNet nudity detection
- [x] CLIP scene classification
- [x] PANNs audio moaning detection
- [x] Multi-level hardware profiles (CPU/Low/Medium/High)
- [x] Checkpoint/resume system
- [x] Batch processing
- [x] PyPI beta release
- [x] CI/CD pipeline (Python 3.11/3.12/3.13)
- [x] Content-type profiles (live-action, animation, anime, low-light)
- [x] Strictness levels (low, medium, high, custom)
- [x] Safe preview mode (HTML contact sheet)
- [x] Jobs cleanup --all/--failed
- [ ] Coverage report badge (target 80%+)
- [ ] Cross-platform CI (macOS, Windows)

## v0.2.0 - Stability & Polish

- [ ] Tauri desktop GUI with timeline scrubbing
- [ ] Before/after preview in GUI
- [ ] Smart render (re-encode only affected segments)
- [ ] Cached model inference per video hash
- [ ] Per-category threshold controls in CLI
- [ ] Temporal tracking improvements (reduce box jitter)
- [ ] Expected time estimator
- [ ] More container formats (MKV, WebM, AVI)

## v0.3.0 - Desktop Packaging

- [ ] Windows executable (PyInstaller/Nuitka)
- [ ] macOS .app bundle
- [ ] Linux AppImage
- [ ] First-run onboarding wizard
- [ ] Bundled FFmpeg
- [ ] Bundled ONNX models

## v1.0.0 - Production Release

- [ ] Full evaluation suite with metrics
- [ ] Multi-GPU support
- [ ] Plugin API for custom detectors
- [ ] Real-world benchmark suite
- [ ] Signed releases with checksums
- [ ] Community-contributed censor plans
- [ ] Comparison page vs VidAngel/ClearPlay
- [ ] Demo video

## Future Ideas

- Subtitle-aware detection (avoid censoring text overlays)
- Smart audio ducking
- Multiple censoring styles (pixelate, emoji overlay, etc.)
- Web-based review interface
- Mobile companion app for remote monitoring
