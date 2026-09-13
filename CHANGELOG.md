# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Desktop auto-updater.** The GUI checks the release endpoint once on
  startup and shows a non-blocking "Update available - Install" notice;
  nothing downloads without that click, and after installing it asks
  for a restart. Every download is verified against a minisign public
  key baked into the app (the private key lives only on the owner's
  machine), and the release pipeline signs each installer and publishes
  `latest.json`. Windows and Linux can update from the first tagged
  release that ships this; macOS stays blocked on code signing
  (Gatekeeper refuses an unsigned replacement).
- **Real-footage evaluation harness.** `scripts/score_real_footage.py`
  runs the real planner on a clip you supply and scores the flagged
  shot ranges against a hand-marked JSON of expected ranges
  (`docs/real-footage-eval.md` is the workflow): per-category
  time-coverage precision/recall/F1 with missed and extra ranges
  spelled out, a configurable boundary tolerance for shot snapping,
  and an append-only JSONL that records the clip as a SHA-256 plus
  resolution - never a path - so numbers accumulate locally and can be
  shared without leaking filenames (the same privacy shape as
  `bench --real`).
- **Plugin API for custom detectors.** Register a detector class through
  the `pureframe.plugins` entry-point group and it shows up in
  `pureframe plugins list`; opt in per run with a repeatable
  `--enable-plugin name` on `plan`/`process`. Plugin categories are box
  providers: they flag per-frame blur boxes on visual evidence alone
  (never gated on audio), can never downgrade a nudity verdict, densify
  and smooth through the same pipeline as nudity boxes, and their
  categories join the per-category threshold controls
  (`--thresholds file.json` keys them like `nudity`). Plans stay
  portable - boxes are data, so a plan renders on machines without the
  plugin. Ships with `pureframe-plugins-examples`, a motion-blob
  reference plugin, and `docs/plugins.md` (user page);
  `docs/plugin-api.md` is the developer reference.
- **Emoji censor style.** `--blur-mode emoji` draws one emoji over each
  box center, sized from the box; the default character is per category
  (kisses get a kiss mark) and `--emoji-char` overrides it. Without a
  system emoji font it falls back to a solid box, so a flagged region
  never renders as untouched pixels. `--blur-mode` now exists on
  `plan`/`process`/`apply` at all (the render style was previously
  settable only from code), and `apply` can override the plan's snapshot.
- **Before/after in the GUI.** The plan editor's shot view shows the
  original/censored PNG pair for the selected shot (as written by
  `pureframe preview --before-after`) side by side under the frame
  thumbnail, through a new path-safe `read_preview_pair` command;
  without a preview run it shows the command hint instead.
- **Bundled FFmpeg on every standalone.** The macOS tarball and Linux
  tarball now ship static `ffmpeg` + `ffprobe` next to the binary, like
  the Windows zip already did - the standalones no longer require a
  system ffmpeg. Each packaging job runs both bundled binaries before
  archiving.
- **Cross-platform emoji verification in CI.** The emoji style's font
  resolution and ink rendering are now asserted on every CI platform;
  the previous tests skipped font-less systems silently, so the macOS
  and Windows font paths were never proven. Each test job also uploads
  rendered probe frames (`scripts/emoji_probe.py`) as run artifacts, so
  a rendered frame can be eyeballed per platform. The solid-box
  fallback stays as the safety net on font-less machines.

### Changed
- Prerelease tags no longer publish to PyPI. The publish workflow reads
  the version from `pyproject.toml` and skips the publish job when it
  carries an `a`/`b`/`rc` suffix, so release-candidate tags can exercise
  the full tag pipeline (CI, installers, standalones, checksums) without
  shipping to PyPI; GitHub releases built from such tags are marked
  prerelease. Publishing itself now authenticates with the
  project-scoped `PYPI_TOKEN` secret instead of trusted publishing,
  whose publisher record was registered under a stale account name and
  could never match the renamed one.

### Fixed
- **The standalone builds crashed on real work - twice.** First: the
  PyInstaller build excluded matplotlib to save space, but
  `panns_inference` (the audio classifier's engine) imports
  `matplotlib.pyplot` at module level, so every standalone (Windows
  zip, macOS tarball, Linux tarball) died on any `process`/`plan` run
  with audio. Second, one layer deeper once that was fixed: NudeNet's
  detector resolves its `320n.onnx` model next to its own module file,
  and PyInstaller does not collect package data by default - so the
  model was never in the bundle and detection died with `NO_SUCHFILE`.
  The `--version` smoke in the packaging jobs never reaches either
  path. matplotlib, the lazy `panns_inference` import, and the nudenet
  data files are now bundled. Found by the v0.2.3-rc1/rc2 tag
  exercises, which for the first time ran a real `pureframe process`
  inside the released artifacts with no system ffmpeg on PATH. The
  packaging jobs now also run a tiny real `process` (bundled models,
  bundled ffmpeg) before archiving, so this class of failure fails the
  release build instead of shipping.
- **Bare `pureframe plugins` died with "Missing command."** Typer's
  subcommand error instead of the plugin listing; the bare form now
  prints the same table as `plugins list` (#99). The plugin corners
  the first pass left open - checkpoint resume with a plugin enabled,
  plugin + `--no-cache`, a plugin-flagged plan rendered without the
  plugin, and the real venv install path as a dedicated CI job - are
  pinned by tests in the same change (#99).
- **`--profile` rejected the documented lowercase form.** Three doc
  pages say `--profile cpu` / `low` / `medium` / `high`; the option now
  accepts any casing and still rejects unknown names with the same
  message shape (#99).
- **AVI files lied about their frame count, and pairwise detectors
  starved.** An MPEG-4 AVI carries a stream duration one tick longer
  than the frames it holds (61 declared for 60 packets on the
  regression fixture), and the plan trusted it twice: the keyframe
  sampler requested the phantom tail frame, extraction silently
  returned nothing, and a profile sampling two keyframes per shot
  handed a differencing detector a single frame. The motion-blob
  plugin scored the shot 0/60 censored on AVI while the identical
  content flagged 60/60 on WebM. `probe_video` now recounts packets
  (demux only) for containers known to overcount and `detect_shots`
  clamps shot boundaries to that count; found by the adversarial
  critic pass on fresh AVI input.

## [0.2.2] - 2026-09-11

### Added
- **Per-category threshold controls in the CLI.** `--threshold-nudity`,
  `--threshold-clip` and `--threshold-audio` (plus `--thresholds
  file.json` accepting `{"nudity": …, "clip": …, "audio": …}`) replace the
  strictness preset's base value for that one category; the others keep the
  preset, and the content-type/`--strict` multipliers still apply on top.
  `--threshold` keeps working as the nudity alias - but it now actually has
  an effect under any strictness (previously it was silently ignored unless
  `--strictness custom` was also passed).
- **Expected-time estimates.** `plan`/`process` print "Analysis estimate:
  ≈ …" right after the probe (frame count × the profile's per-frame cost,
  calibrated from the v0.2.1 bench medians in `pureframe/eta.py`), and
  `process`/`apply` print "Render estimate: ≈ …" once the plan knows how
  many frames are flagged (re-encode cost for flagged frames, stream-copy
  for the rest). The analysis progress bar keeps its live remaining-time
  column.
- **Inference cache keyed on the file's content.** Re-running `process`/`plan`
  on an unchanged file with the same settings skips model inference
  entirely - verdicts come from the checkpoint store, which now folds a
  streaming SHA-256 of the input into its key. Replacing the file at the
  same path (the old key couldn't see that) is a cache miss, as is any
  config change. `--no-cache` forces a from-scratch analysis for one run
  without touching what the cache holds.
- **Before/after preview.** `pureframe preview --before-after` renders one
  full-resolution PNG pair per flagged shot - the untouched frame and the
  same frame with that plan's censoring applied through the same overlay
  code the real renderer uses - embedded side-by-side in the HTML report,
  so blur placement can be verified per shot without rendering the video.
- **Nightly slow-suite CI job.** The scheduled job runs the real-render e2e
  guards and model classifier tests daily; regular CI keeps `-m "not slow"`,
  which is how the 0.2.0 render bugs shipped unnoticed.
- **Real-world benchmark mode.** `pureframe bench --real yourfile.mp4` times
  the full process flow on your own content per profile with a per-phase
  breakdown, appending privacy-safe JSON lines to a local JSONL (files are
  identified by SHA-256 and resolution, never a path) so numbers accumulate
  and can be shared without leaking filenames.
- **GUI timeline scrubbing.** The plan editor's timeline gains a seekable
  scrub bar: dragging updates the timecode immediately and fetches the frame
  at the position (debounced, latest-wins) through the existing thumbnail
  IPC; selecting a shot moves the bar to that shot's midpoint.
- **CUDA device selection (`--device`).** Multi-GPU machines could only ever
  use GPU 0. `--device 1` pins NudeNet (ONNX CUDA EP provider options), CLIP
  and PANNs to the chosen GPU, and auto-profiling reads that device's VRAM
  so a smaller second GPU picks a fitting profile. Excluded from the
  checkpoint hash - a cached verdict from GPU 0 is valid on GPU 1. Full
  multi-GPU sharding is analyzed and descoped in `docs/multi-gpu.md`.

### Fixed
- **Flagged plans failed to serialize on Python 3.13.** `sample_keyframes`
  returned numpy int64 indices that ended up as `Shot.frames` keys; pydantic's
  strict JSON serializer rejects numpy scalars, so `pureframe plan` crashed
  while saving any plan that actually flagged something (the unflagged test
  fixture never hit the path). The sampler now returns plain ints.
- **Flagged shots could lose their blur boxes.** The plan loop filtered
  densified detections at the raw CLI threshold (default 0.55) while the
  fusion flags on the *effective* threshold - so under `--strictness high`
  (nudity preset 0.35) a 0.45-score detection produced a `BLACK_BOX`
  verdict with no boxes, and nothing rendered for it. Densify now keeps
  everything the fusion could have flagged on.

- **Only the first shot of a video was ever analyzed (0.2.0–0.2.1).** The
  plan loop's prefetch worker (added with the pipelined extraction in 0.2.0)
  used a queue helper whose bare `return` read as "stop" to the shot loop,
  so extraction quit after shot 0 and the plan ended with one verdict - on a
  real movie, nothing past the first cut was ever inspected or censored.
  Every fixture in the suite happened to be single-shot, which is how it
  slipped through; a three-shot regression clip now pins one verdict per
  shot and boxes on the right one.
- **WebM and AVI inputs failed to render.** Every re-encode was hardcoded
  to H.264, which WebM refuses outright ("Only VP8 or VP9 or AV1 video …
  supported") and AVI rejects without an Annex-B bitstream filter - so
  `process` on either container died at the final mux, despite the README
  promising both. Censored segments now follow the source codec (VP8/VP9
  for WebM, MPEG-4 for AVI, HEVC stays HEVC), each with its encoder's own
  quality flags, and AVI/H.264 gets `h264_mp4toannexb`. Pinned by
  `tests/test_container_formats.py`: MKV/H.264, WebM/VP9, WebM/VP8,
  AVI/MPEG-4 and AVI/H.264, through both the smart segment path and the
  full re-encode path, checking codec, frame count and blur placement.
- **Smart renderer keyframe map was wrong for VP8/VP9.** The probe relied
  on the decoder honoring `-skip_frame nokey`; the VP8/VP9 decoders ignore
  it and reported every frame as a keyframe, so WebM copy cuts landed off
  sync points and dropped frames (148 of 150 on the regression clip). The
  map is now read from packet flags - demux only, no decoding - which is
  codec-independent and faster on long files.
- **Smart render crashed on Python 3.11.** The renderer receives fps as a
  `Fraction` from the plan metadata and formatted a derived duration with
  `:.1f`, which `Fraction` only supports from Python 3.12 - on 3.11 the
  smart path raised before reaching its fallback. fps is now normalized to
  a float at the renderer boundary.
- **The box smoother froze the last two frames of every track.** The median
  filter used `scipy.signal.medfilt`, which zero-pads the sequence edges -
  the median of the final frames was dragged toward 0, so a subject moving
  at the end of a shot had its blur box visibly lag behind (22 px behind on
  an 8 px/frame mover, frozen for the final 3 frames). The smoother now
  uses `scipy.ndimage.median_filter(mode="nearest")`, which replicates edge
  values. Box EMA and track hysteresis were evaluated against the fixed
  pipeline and rejected - on realistic sparse anchors the EMA's lag biases
  every interpolation (1.4–1.5× worse RMSE on movers), and the median
  filter already absorbs most dense-anchor jitter; the measurements and
  the remaining idea (Kalman/spline over anchors) are recorded in
  `pipeline/smooth.py` and `tests/test_smoothing_tails.py`.

### Changed
- HEVC sources now re-encode censored segments as HEVC instead of H.264.
  Mixing the two broke the concat step, so every HEVC file silently took the
  full re-encode path; the smart path now applies to HEVC too. `output_codec`
  keeps applying to H.264 sources exactly as before.

## [0.2.1] - 2026-09-08

Urgent follow-up to 0.2.0: **0.2.0's render path censored almost nothing.**
Two defects shipped together and masked each other - nudity verdicts carried
no boxes (so flagged shots were re-encoded without blur), and the smart
renderer's keyframe-unaware segment cuts duplicated content after the first
flagged segment. Upgrading from 0.2.0 is strongly recommended; from ≤0.1.0b16
it is the same upgrade path as 0.2.0 plus these fixes.

### Fixed
- **Nudity detections were never censored: every `NUDITY_EXPLICIT` verdict
  shipped with `boxes=None`.** The plan loop only attached tracked boxes for
  kiss-flagged shots (mouth boxes) and `FULL_FRAME_BLUR` verdicts; the
  primary category - nudity, a `BLACK_BOX` action that renders per-frame
  boxes - fell through both paths, so the renderer re-encoded flagged shots
  **without applying any blur**. Only the sexual-context full-frame branches
  and hand-edited plans ever censored anything. Nudity-flagged shots now run
  the same densify + IoU-tracking pass, attaching per-frame boxes to the
  plan (151 tracked frames on the e2e fixture, previously 0). The real-render
  e2e guards that cover this were failing on master and skipped by CI
  (`-m "not slow"`); both now pass.
- **Smart renderer duplicated content after the first flagged segment.**
  Clean segments were cut with input-side `-ss`/`-to` under `-c copy`, which
  snaps back to the previous keyframe in decode order - on a clip whose only
  keyframe is frame 0, the final clean segment re-copied the entire video
  (a 5 s clip rendered 9.1 s, and the README demo GIF visibly "played the
  original, unblurred" after the censored section). The renderer now probes
  the keyframe map, snaps segment boundaries outward to keyframes, and splits
  with the segment muxer (lossless, B-frame-safe); single-keyframe inputs
  correctly take the full-re-encode fallback. Output frame count now matches
  the input exactly - pinned by a real-clip regression test.
- GUI plan editor: `SEXUAL_CONTEXT_NO_NUDITY` timeline segments rendered red
  instead of orange (the category check matched "NUDITY" inside
  "NO_NUDITY" before the sexual-context check).

### Changed
- `packaging/sync_versions.py` now also syncs (and `--check` guards) the
  `pureframe-desktop` version entry in `gui/src-tauri/Cargo.lock`, which had
  silently stayed at 0.1.0-beta.15 through two releases.
- Demo GIF regenerated from the fixed renderer (0.94 MiB, exact 5 s loop) and
  the GUI plan-editor screenshot refreshed with the corrected timeline colors.
- Re-benchmarked after the renderer fixes: published medians reproduce within
  run-to-run noise (`pureframe bench --duration 30 --reps 3`, RTX 3060:
  CPU 3.0 s · LOW 15.4 s · MEDIUM 17.1 s · HIGH 24.9 s).

## [0.2.0] - 2026-09-08

First stable release. Headline: the speed offensive.

### Added
- **Speed offensive** - algorithmic + model-level optimizations targeting ~10–20 min for a 90-min movie on CPU-only hardware (see `docs/performance.md` for the full narrative):
  - `pureframe bench`: repeatable benchmark with a detection-exercising synthetic clip, per-phase timers, checkpoint isolation, JSON/Markdown reports.
  - Per-phase timers (`--verbose`) and the `PUREFRAME_TIMERS_FILE` machine hook.
  - Seek-based frame extraction (was: re-decode from frame 0 per shot); probed metadata reused across calls.
  - `fuse.context_audio_needed`: the audio classifier runs only when the CLIP scene signal makes its score relevant (verdict-identical skip, pinned by parity tests).
  - Scene-detection `frame_skip` per profile; kiss shots sampled at the densify stride.
  - Pipelined plan generation: keyframe extraction overlaps model inference on a bounded worker.
  - int8 dynamic quantization of NudeNet on CPU profiles (cached, `--no-quant` escape hatch, eval-parity gated) and a center-10s PANNs analysis window.
  - Encoder presets per profile; encoder/fps resolved once per smart render; one less full-frame copy per dirty frame.
  - `eval-parity` CI job: the real detector runs the synthetic corpus on every PR; detection-signature drift fails the build. See `eval-baseline.json`.

### Changed
- README publish pass: the demo GIF is now a cinematic synthetic scene with a
  tracked censor blur (the old one was a raw test pattern *and* placed the
  blur in the wrong corner - plan boxes must be authored in detection space);
  measured post-offensive bench numbers replaced the stale pre-offensive
  table; GUI screenshots added; stale version pins removed.

## [0.1.0b16] - 2026-09-06

The revival release. The code was sound, but the ecosystem moved underneath it
while the repo was dormant: transformers 5, OpenCV 5, ffmpeg 7, and a ruff
release all broke something. Every failure mode now ships with a regression
guard. Full suite (251 tests + GUI e2e) green on Ubuntu/macOS/Windows ×
Python 3.11–3.13. Also ships the unreleased work from the previous cycle
(smart blur render path, GUI security hardening, Playwright e2e harness).

### Added
- Live job status for the desktop GUI: the Tauri backend drains job stdout/stderr into a bounded log ring (previously the pipes were never read, which could block long renders mid-run) and exposes a `job_status` command reporting state, exit code, output path, and a 24-line log tail. Plan jobs report the derived `<input>.censorplan.json` path so the UI can open them directly.
- GUI: hardware profile, content type, and detection threshold are now sent with every job (the Settings page previously stored preferences it never used), validated server-side.
- GUI: dark theme, per-job progress bars parsed from CLI output, completion/failure toasts, and a "Review plan" action on finished plan jobs.
- Explicit ruff lint contract in `pyproject.toml` (E4/E7/E9/F/I/UP at target py311) so a ruff release can no longer silently redefine "passing" in CI.
- Regression tests: checkpoint trust semantics, NudeNet `xywh→xyxy` box contract, BLUR-mode e2e with Laplacian-variance assertions.
- Dynamic release badge and demo caption in README; demo GIF regenerated web-optimized (~1 MB).
- Playwright E2E smoke harness for the desktop GUI (`gui/e2e/`) with a `window.__TAURI_INTERNALS__` shim so the React tree boots outside the Tauri runtime; CI runs the suite on every push/PR via the `gui-e2e` job.
- `BlurMode` enum in `config.py` (BLUR / BOX / PIXELATE) with shared overlay callback in `pipeline/render/overlay.py` - render path now applies real localized Gaussian blur or pixelation instead of solid boxes.
- GPU-aware PANNs audio classifier with label-name lookup (falls back to known indices when `panns_inference.labels` unavailable).
- Per-category max-cosine CLIP scene scoring (categories no longer compete based on prompt count).
- Cross-platform CI matrix (`ubuntu-latest`, `macos-latest`, `windows-latest`).

### Fixed
- transformers 5.x compatibility: `CLIPModel.get_text_features/get_image_features` return a `BaseModelOutputWithPooling` (projected features in `pooler_output`) instead of a bare tensor. Both 4.x (>=4.30) and 5.x supported - un-breaks `SceneClassifier` init.
- OpenCV 5 compatibility: both `opencv-python` flavors pinned `<5` (the unpinned GUI-flavor dragged in by scenedetect clobbers the pinned headless one via the shared `cv2` namespace, and 5.x removed the Caffe DNN importer). `FaceDetector` degrades with a logged warning instead of crashing when the importer is absent.
- ffmpeg 7+ compatibility: `-vsync` was removed; frame selection now uses `-fps_mode passthrough` with a legacy fallback. Failed decodes raise with the real ffmpeg stderr instead of silently returning no frames.
- Checkpoint trust: DONE jobs no longer skip processing when the output file is missing/empty or keyed to a different output path; renders that produce nothing are never recorded DONE.
- Frame-extraction deadlock: ffmpeg stderr is drained in a background thread (long `select` expressions filled the OS pipe buffer and blocked stdout reads).
- Mux failures surface the ffmpeg stderr tail; concat demuxer files escape single quotes and refuse paths outside the render tmpdir.
- `select_hw_encoder` resilient to non-standard `ffmpeg -encoders` output (no longer raises `IndexError`).
- Smart render: `_extract_and_render_segment` passes `ss`/`to` to `write_video_with_overlay`; frame indices translated via the segment's `frame_offset` so overlay key lookups remain correct after the FFmpeg slice.
- `gui/package.json`: corrected `lucide-react ^1.14.0` → `^0.460.0`; removed bogus `radix-ui ^1.4.3` and `shadcn ^4.7.0` runtime dependencies.

### Changed
- 111 mechanical lint fixes (import ordering, PEP 604/585 annotations, `datetime.timezone.utc` → `UTC`).
- Scripts: inline argv lists with `shell=False`, `Path.write_text`, import placement.
- `config_hash` incorporates render settings (`blur_mode`, `blur_kernel`, `blur_sigma`, `pixelate_blocks`, `output_codec`, `output_crf`, `clip_threshold`, `audio_threshold`, `box_color`) so checkpoints invalidate correctly on render-related config changes.
- Checkpoint store derives `completed_shots` via `COUNT(*) FROM shot_verdicts` (eliminates inflation on resume).
- `batch.py` clones the base config via `model_copy(update=..., deep=True)` so all CLI flags propagate to per-file runs.
- `cli.execute_render` catches `Exception` (not `BaseException`) - no more accidental `KeyboardInterrupt` swallowing.

### Security
- Tauri desktop CSP locked down: explicit `default-src 'self'`, `script-src 'self'`, `object-src 'none'`, `frame-ancestors 'none'`.
- Tauri Rust commands canonicalize every user-supplied path through `validated_path` with an extension allow-list (blocks path traversal and arbitrary read/write).
- Plan size cap (16 MiB) and JSON validation before `save_plan` writes to disk.
- `cancel_job` calls `child.wait()` after `kill()` to reap zombies.
- GUI replaces browser `prompt()` with the native Tauri dialog plugin (real file picker, no spoof-able input).
- Mode string in `start_job` validated against `{"plan","apply","process"}` allow-list before being passed to the subprocess.
- ffmpeg concat files escape single quotes and refuse paths outside the render tmpdir.

### Documentation
- `ROADMAP.md` synced to reality; `BENCHMARKS.md` expanded with methodology notes and a contributor capture checklist.

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
