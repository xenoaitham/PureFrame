# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b7] - 2026-05-07

### Added
- **Smart segment rendering** (`pureframe/pipeline/render/smart.py`) - only re-encodes dirty frames, stream-copies clean segments. 2-5x faster on typical content.
- **Evaluation benchmark** (`pureframe evaluate`) - 50 synthetic test scenarios across 8 content genres with precision/recall/F1 metrics and threshold sweep analysis.
- **Confidence calibration guide** (`docs/CALIBRATION.md`) - threshold presets, content-type multipliers, and step-by-step tuning workflow.
- **Known limitations doc** (`docs/KNOWN_LIMITATIONS.md`) - comprehensive FP/FN categories, animation-specific issues, rendering and audio detection gaps.
- **Desktop packaging** (`packaging/build_desktop.py`) - PyInstaller build script for standalone executables on Linux/macOS/Windows.
- **Codecov integration** in CI workflow for automated coverage badge.
- Tests for smart rendering (segment detection, fallback logic, stream copy).
- Tests for evaluation benchmark (SceneResult classification, metrics computation, 50-scenario validation, synthetic frame generation).
- Coverage and test count badges in README.

### Changed
- `execute_render` now uses smart segment rendering by default (falls back to full re-encode if >60% dirty or on error).
- CI timeout increased to 15 minutes; Codecov upload step added.
- README badges: added coverage (83%), test count (165), Codecov link.
- README Known Limitations section now links to full `KNOWN_LIMITATIONS.md`.
- FAQ threshold question now links to `CALIBRATION.md`.
- Documentation table expanded with Calibration and Known Limitations entries.

## [0.1.0b6] - 2026-05-07

### Added
- **165 unit tests** (up from 54), achieving **83% code coverage** (up from 71%).
- Full test coverage for: `checkpoint.py`, `config.py`, `face.py`, `probe.py`, `scene.py`, `logging.py` (all 100%).
- End-to-end render tests for `apply_censoring` covering blur, black box, mixed actions, empty boxes, and custom colors.
- Mocked hardware detection tests covering all VRAM thresholds (HIGH/MEDIUM/LOW/CPU) and exception handling.
- Mocked FFmpeg encoder selection tests covering nvenc, videotoolbox, QSV, AMF fallback chains for h264/hevc.
- Smooth detection tests: interpolation, padding, median filtering, multi-track independence, clipping.
- Densify shot tests with mocked detector: threshold filtering, empty/populated detection output.
- SceneDetector mocked tests: CLIP model init and frame analysis with mocked inference.
- Extended CLI tests: version flags, auto-output, verbose mode, invalid JSON, plan-edit with `$EDITOR`, jobs cleanup flags.
- `frames_iter` tests for FFmpeg frame iteration with and without downscaling.
- HDR/HLG detection tests and multi-audio-stream metadata extraction tests.

### Changed
- `render/apply.py` coverage: 33% → 93%.
- `densify.py` coverage: 23% → 96%.
- `scene.py` coverage: 0% → 100%.
- `smooth.py` coverage: 81% → 98%.
- `hardware.py` coverage: 78% → 96%.
- `ffmpeg.py` coverage: 71% → 92%.

## [0.1.0b5] - 2026-05-07

### Added
- **Content-type profiles**: `--content-type live-action|animation|anime|low-light` with per-type threshold multipliers.
- **Strictness levels**: `--strictness low|medium|high|custom` with per-category threshold presets.
- **Preview command**: `pureframe preview` generates HTML contact sheets of flagged shots for safe review.
- **Partial nudity labels**: Added `FEMALE_BREAST_COVERED`, `BELLY_EXPOSED`, `ARMPITS_EXPOSED` to detection label set.
- Full documentation suite (9 new docs): installation, CLI reference, architecture, evaluation, censor plan schema, privacy, legal, plan sharing, and examples.
- `CONTRIBUTING.md` with complete contributor onboarding guide.
- `.github/ISSUE_TEMPLATE/` with bug report and feature request templates.
- Golden-file test for censor plan JSON schema validation.
- Config model tests: 20 new tests for content-type, strictness, hashing, and factory.
- Fusion engine tests: 10 new tests for threshold-aware detection across content types and strictness levels.
- CLI feature tests: 7 new tests for preview, content-type, strictness, and jobs cleanup flags.
- Coverage reporting in CI with `pytest-cov` and artifact upload.

### Changed
- Fusion engine now uses effective thresholds from `Config.get_effective_thresholds()` (composing content-type × strictness).
- Preview command uses `config_snapshot.input_path` instead of removed `VideoMetadata.filename` field.
- `jobs cleanup` now supports `--all` (wipe everything) and `--failed` (only failed jobs) flags.
- Updated README with collapsible FAQ, new badges, documentation table, content-type/strictness docs.
- Professional `ROADMAP.md` with phased milestones and checked completed items.

### Fixed
- Removed unused `use std::path::Path;` import from Tauri backend (`gui/src-tauri/src/lib.rs`).
- Fixed ruff lint warnings: removed f-string prefix from strings without placeholders.
- Fixed `preview` command crash on `plan.input_metadata.filename` (field doesn't exist on `VideoMetadata`).

## [0.1.0b4] - 2026-05-07

### Fixed
- **Root cause CI fix:** `select_hw_encoder` was receiving `ProfileSettings` object instead of `HardwareProfile` enum, causing it to skip the CPU guard and select `h264_nvenc` on CI runners without a GPU. Fixed by passing `profile_settings.profile`.
- Removed deprecated `-vsync 0` ffmpeg argument that caused warnings on newer ffmpeg versions.
- Removed colorspace pass-through kwargs - let ffmpeg autodetect input colorspace.
- Added even-dimension enforcement for yuv420p encoding compatibility.
- Added `BrokenPipeError` handler with stderr tail capture for better diagnostics.

### Changed
- Improved ffmpeg error reporting: encoder crashes now include the last 3000 chars of stderr.
- Default frame rate fallback to 24.0 if metadata reports invalid fps.

## [0.1.0b3] - 2026-05-07

### Fixed
- Synchronized `pyproject.toml` version with PyPI release.
- Fixed README badge version mismatch.
- Fixed README Markdown table formatting (tables were compressed into single lines).
- Fixed FAQ section formatting.
- Cleaned up compressed YAML/Python/TOML files that had been flattened.

### Changed
- Updated all documentation files to proper multi-line formatting.

## [0.1.0b2] - 2026-05-07

### Added
- Initial beta release.
- CLI `plan`/`apply`/`process` workflow.
- Local explicit-content detection using NudeNet.
- JSON censor plan review workflow with `plan-edit` and `plan-whitelist`.
- FFmpeg-based blur rendering pipeline.
- Audio moaning detection via PANNs.
- Scene-level CLIP classification.
- Multi-level hardware profiles: CPU, Low, Medium, High.
- Checkpoint/resume system with SQLite job store.
- Batch processing support.
- Rich CLI progress output.
- Tauri-based desktop GUI (experimental).
- CI/CD pipeline with Python 3.11/3.12/3.13 matrix.
- PyPI publishing workflow.

## [0.1.0b1] - 2026-05-06

### Added
- Initial internal release.
- Basic project structure and CLI skeleton.
