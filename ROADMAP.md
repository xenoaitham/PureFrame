# Roadmap

Last reconciled 2026-09-10, against v0.2.1 plus the two fixes that followed
it (#66, #67). Every tick below carries its evidence - a PR, a workflow, a
file - so nobody has to guess again.

Version-number buckets stopped matching reality (desktop installers, a
"v0.3.0" item, shipped before half of the "v0.2.0" list), so this is now one
board: **Now** is being worked in order, **Next** is queued and scoped,
**Later** is wanted but not scoped, and **Blocked on me** needs a purchase or
a decision that isn't the code's to make.

## Now

_Nothing in flight - the next work is queued under Next._

## Next

- **Before/after in the GUI.** The CLI side shipped (`pureframe preview
  --before-after` writes full-resolution PNG pairs per flagged shot, embedded
  in the HTML report); the GUI plan editor still needs to load and display
  them via `load_plan`-style IPC.
- **Plugin API for custom detectors.** Design note before code: a detector
  is `detect_batch(frames) -> [Detection]` + a label→category map + a
  threshold, registered via entry points; `fuse()` must treat non-nudity
  categories as box providers (the box plumbing exists since #63). Ship with
  one example plugin and a docs page. → **Design note written:
  `docs/plugin-api.md`** - code lands in the four slices it lays out, after
  a read-through from me.
- **Emoji / sticker overlay** as a fourth censor style (blur, solid box and
  pixelate exist - `BlurMode` in `config.py`).
- **Bundled FFmpeg for the macOS and Linux standalones.** The Windows zip
  already carries `ffmpeg.exe` + `ffprobe.exe`; the others still need a
  system ffmpeg.

## Later

- **Multi-GPU sharding.** The descope shipped: `--device cuda:N` pins all ML
  models to a chosen GPU with per-device VRAM profiling (`docs/multi-gpu.md`).
  Shot-level sharding stays here - the plan loop and checkpoint store would
  need surgery, and detection is the smallest phase on the reference box
  (Amdahl); revisit for 4K-first or CPU-dominated deployments.
- **Evaluation on real footage.** The synthetic corpus and the eval-parity
  gate exist; a metrics suite on real (user-supplied) footage does not.
- **AV1 and other codecs re-encoded in their own codec.** Today they fall
  back to H.264, which WebM rejects (see `docs/KNOWN_LIMITATIONS.md`).
- **Model-based box smoothing (Kalman or spline over anchors).** Box EMA and
  track hysteresis were evaluated and rejected - the EMA's lag biases every
  interpolation on sparse anchors (1.4–1.5× worse RMSE on movers) and the
  median filter already absorbs most dense-anchor jitter; measurements in
  `pipeline/smooth.py` and `tests/test_smoothing_tails.py`. The one fix that
  survived evaluation - the zero-padded median filter freezing track tails -
  shipped.
- Subtitle-aware detection (don't blur text overlays), smart audio ducking,
  web-based review UI, mobile companion app, community-contributed censor
  plans.
- Comparison page vs VidAngel/ClearPlay and a demo video - see Blocked on me
  for the parts that aren't engineering.

## Blocked on me

- **macOS code-signing + notarization** and the **Tauri auto-updater with
  signed manifests.** Both need a paid Apple Developer certificate ($99/yr)
  and updater signing keys. Exact steps and the GitHub secrets list live in
  `docs/release-signing.md`; the cert purchase is the only missing piece.
- **Windows Authenticode signing.** Same doc, separate (paid) certificate.
  Until then Windows shows the SmartScreen warning on the installer.
- **Demo video and the comparison page.** Asset and messaging decisions;
  drafts can be prepared, the framing is my call.

## Shipped

### Foundations (v0.1.x)

- [x] CLI plan/apply/process workflow
- [x] NudeNet nudity detection
- [x] CLIP scene classification
- [x] PANNs audio moaning detection
- [x] Multi-level hardware profiles (CPU/Low/Medium/High)
- [x] Checkpoint/resume system
- [x] Batch processing
- [x] PyPI release
- [x] CI/CD pipeline (Python 3.11/3.12/3.13, Linux/macOS/Windows)
- [x] Content-type profiles (live-action, animation, anime, low-light)
- [x] Strictness levels (low, medium, high, custom)
- [x] Safe preview mode (HTML contact sheet)
- [x] Jobs cleanup --all/--failed
- [x] Coverage badge - 83% via Codecov
- [x] Real localized Gaussian blur / pixelate / solid box (`BlurMode`)
- [x] Hardened Tauri backend (path canonicalization, CSP lockdown, plan size cap)

### September 2026 speed offensive (v0.2.0)

- [x] Seek-based frame extraction (no per-shot from-0 decode)
- [x] NudeNet session stays resident (no per-frame ONNX re-init)
- [x] Lazy audio classification (runs only when its score can matter)
- [x] Scene-detection frame_skip profiles
- [x] Pipelined plan generation (extraction ∥ inference) - the worker
      quit after shot 0 until #66; see the incidents note below
- [x] int8 CPU quantization of NudeNet (eval-parity gated, `--no-quant` escape)
- [x] PANNs analysis window cap (center 10 s)
- [x] Encoder presets + per-render encoder/fps caching + copy elimination
- [x] `pureframe bench` repeatable benchmark with per-phase timers
- [x] `eval-parity` CI gate (detection-signature drift fails the build)
- [x] Fresh benchmark tables - closed as done: `BENCHMARKS.md` and
      `docs/performance.md` carry v0.2.1 numbers on the reference machine
      (CPU 3.0 s / LOW 15.1 s / MEDIUM 16.2 s / HIGH 23.7 s; a fresh
      session run reproduced them within noise). The before/after narrative
      lives in `docs/performance.md`; re-running pre-offensive code for a
      side-by-side table isn't worth the time.

### Rendering and formats

- [x] Smart render (re-encode only affected segments) - v0.2.0; keyframe-
      safe since v0.2.1 (#63)
- [x] More container formats: MKV, WebM, AVI - verified and fixed in #67.
      WebM and AVI actually failed at the final mux until then. Pinned by
      `tests/test_container_formats.py`: MKV/H.264, WebM/VP9, WebM/VP8,
      AVI/MPEG-4, AVI/H.264, through both the smart-segment and the
      full-re-encode paths

### Detection controls

- [x] Per-category thresholds in the CLI - `--threshold-nudity/-clip/-audio`
      and `--thresholds file.json`, replacing one preset value at a time;
      `--threshold` now works under any strictness. Also fixed while wiring
      it: densify filtered boxes at the raw default instead of the effective
      threshold, so preset-flagged shots could lose their blur boxes
- [x] Expected-time estimator - `pureframe/eta.py`, calibrated from the
      v0.2.1 bench medians; analysis ETA printed after the probe, render
      ETA once the plan knows the flagged-frame count
- [x] Temporal smoothing - the median filter's zero-padded edges froze the
      last two frames of every track (blur visibly lagged movers at shot
      end); now edge-replicating. Box EMA/hysteresis evaluated and rejected,
      measurements recorded (`tests/test_smoothing_tails.py`)
- [x] Cached inference per video - checkpoint keys fold a streaming SHA-256
      of the input (`Config.content_fingerprint`), so unchanged file +
      config re-runs skip model inference, and replaced files are misses
      (the old path-only key couldn't see that). `--no-cache` escape with a
      per-invocation salt; pinned by `tests/test_inference_cache.py`
- [x] Before/after preview (CLI) - `pureframe preview --before-after`
      renders a full-resolution original/censored PNG pair per flagged shot
      through the real overlay code and embeds them in the HTML report
- [x] Nightly slow-suite CI job - `.github/workflows/nightly.yml`; also
      fixed on the way: flagged plans failed to serialize on py3.13 (numpy
      int64 keys from the keyframe sampler)
- [x] Real-world benchmark - `pureframe bench --real <file>` times a
      user-supplied file per profile/phase into an append-only JSONL;
      records carry only a SHA-256 + metadata (no paths), no copyrighted
      sample ships, workflow documented in `docs/performance.md`

### Continuous integration

- [x] Nightly slow-suite job - `.github/workflows/nightly.yml` runs the full
      slow suite (real-render e2e guards + model classifier tests) daily and
      on demand; the gap where those guards never ran in CI is how the
      v0.2.0 render bugs shipped

### Plan editor (GUI)

- [x] Timeline scrubbing - seekable scrub bar under the shot strip; dragging
      fetches the frame at the position via the existing `extract_thumbnail`
      IPC (debounced, latest-wins), selecting a shot moves the bar to its
      midpoint. Iterated in a plain browser via the e2e shim; Playwright
      covers the interaction (#78)

### Desktop packaging (every release since v0.2.0 - `release.yml`)

- [x] Windows installer (`.exe`, `.msi`) and PyInstaller zip
- [x] macOS `.app` / `.dmg` (arm64 and x64) and PyInstaller tarball
- [x] Linux AppImage, `.deb`, `.rpm` and PyInstaller tarball
- [x] Bundled FFmpeg - Windows zip only (`ffmpeg.exe` + `ffprobe.exe`);
      macOS/Linux standalones use the system ffmpeg → tracked under Next
- [x] Bundled models - NudeNet ships inside the standalones (it lives in the
      `nudenet` wheel); CLIP and PANNs download on first run
- [x] First-run onboarding wizard - the GUI's onboarding page (`gui/src/App.tsx`)
- [x] `SHA256SUMS.txt` attached to every release (`release.yml`). Code
      signing is a separate, paid item - see Blocked on me
- [x] 13 assets on v0.2.1: `gh release view v0.2.1 --json assets`

### Incidents that shaped this board

- **0.2.0 censored almost nothing** (#63): nudity verdicts carried no boxes,
  and the smart renderer's keyframe-unaware cuts duplicated content. Both
  had real-render e2e tests that were failing on master - in the slow suite
  CI never runs. Hence "Nightly slow-suite CI job" under Now.
- **0.2.0–0.2.1 analyzed only the first shot** (#66): the plan loop's
  prefetch worker quit after shot 0. Every fixture in the suite was
  single-shot. A three-shot regression clip now pins one verdict per shot.
