# PureFrame — session handoff: finish the roadmap

You are continuing work on PureFrame, a local AI tool that applies smooth,
tracked blur over explicit visuals in movies (no cuts, no audio changes).
Repo: `~/Downloads/githup/PureFrame(antigravity)` — GitHub: **xenoaitham/PureFrame**
(account was renamed from MayonaiseLover; all URLs/CODEOWNERS already use
xenoaitham). Owner/user: LO. Tone: warm, direct, no corporate starch.

## State (all on master, CI green, 299 fast tests passing)
- **v0.2.1 is the current release** (PyPI + GitHub, 13 assets). v0.2.0 is
  broken — superseded same-week: its nudity verdicts carried no boxes
  (nothing blurred) and the smart renderer duplicated content after the
  first flagged segment (keyframe-unaware `-c copy` cuts). Both fixed in
  PR #63 with regression tests; read that PR before touching the renderer
  or the plan loop.
- Smart renderer now probes keyframes, snaps segment bounds outward, splits
  with the segment muxer; single-keyframe inputs take full re-encode.
  Output frame count == input frame count (real-clip regression test in
  tests/test_smart_render.py, fast suite).
- The two real-render e2e guards (tests/test_e2e.py, `slow`-marked) PASS now
  but CI still runs `-m "not slow"` — they are not in CI. Consider adding a
  scheduled/nightly slow-suite job; that gap is how the P0 shipped.
- Bench medians reproduce published tables within run noise (CPU 3.0s /
  LOW 15.4s / MEDIUM 17.1s / HIGH 24.9s on the RTX 3060 reference machine).
- GUI boots in a plain browser via the e2e Tauri shim
  (gui/e2e/tauri-shim.ts): `npm run dev` + inject the shim as an
  initScript — that's how README screenshots were captured. Reuse it for
  GUI feature work; no Rust build needed for UI iteration.
- Mimosa deep audit on the final tree: sealed, 0 findings
  (2026-09-08). Re-run after any dependency change.
- Dev machine: kernel 7.1.5 (fixed, mei_pxp blacklisted — do not
  re-litigate), NVIDIA 595.84, model loads safe. Sudo needs LO's password —
  hand him command blocks for privileged ops. ~/Downloads holds 46G of his
  personal files — never touch.

## Hard workflow rules (violating these wastes turns)
1. Direct pushes to master are BLOCKED by a GitHub ruleset. Always:
   branch → push → `gh pr create` → `gh run watch <id> --exit-status`
   (run id via `gh run list --branch <branch> --workflow ci.yml --limit 1`)
   → `gh pr merge --squash` → `git checkout master && git fetch && git
   reset --hard origin/master`. If a PR's CI never triggers after a branch
   delete/recreate, open a FRESH PR — that reliably fires it.
2. Mimosa edit gate: subprocess argv lists must be **inlined literally at
   the call site** (a variable first arg = blocked, even with shell=False).
   `open(path, "w")` = blocked — use `Path.write_text()`. It also
   false-flags some plain `sed`/read-only Bash calls; retry once, then use
   Read/Edit tools instead. Advisory commit notes about "compatibility
   policy" are informational, not failures.
3. Python env: uv (`uv sync --extra dev`). Lint: E4/E7/E9/F/I/UP at py311
   (UP042 ignored). Run `uv run ruff check pureframe tests scripts
   packaging && uv run ruff format --check pureframe tests scripts
   packaging` before committing (packaging/ is NOT covered by CI's lint —
   keep it clean anyway).
4. CI = `pytest -m "not slow"` on 3 OSes × py3.11–3.13 + eval-parity job
   (real NudeNet on the synthetic corpus vs eval-baseline.json; drift fails
   the build — regenerating the baseline is a deliberate act, never silence
   it casually).
5. Releases: bump pyproject → `uv run python packaging/sync_versions.py`
   (syncs package.json/tauri.conf/Cargo.toml/**Cargo.lock**; `--check`
   guards all) → date the CHANGELOG → PR → merge → tag `vX.Y.Z` → push tag
   → release.yml + publish.yml fire → verify on PyPI and
   `gh release view --json assets`.

## THIS SESSION'S TASK: reconcile, then finish, ROADMAP.md

### Step 0 — Reconcile the roadmap with reality (PR #1, docs-only)
ROADMAP.md is stale and understates the product. In one PR, move/mark items
with evidence — do not guess:
- Already shipped (mark `[x]` with the version that did it):
  - v0.3.0 "Windows executable / macOS .app / Linux AppImage / Bundled
    FFmpeg" — every release ships Tauri installers (.exe/.msi/.dmg/
    .AppImage/.deb/.rpm) and PyInstaller standalones; the Windows zip bundles
    ffmpeg.exe + ffprobe.exe.
  - v0.2.0 "More container formats (MKV, WebM, AVI)" — VERIFY first (the
    README claims MP4/MKV/AVI/WebM; write a fast parametrized test that
    runs `pureframe process` over tiny lavfi-generated MKV/WebM/AVI clips;
    if it passes, mark done with the test as evidence; if it fails, that's
    a real bug — fix it as part of this track).
  - v1.0.0 "Signed releases with checksums" — partially: SHA256SUMS.txt is
    attached to every release; code-signing is a separate (paid) item.
    Split it: mark checksums done, rename the signing item explicitly.
  - v0.3.0 "First-run onboarding wizard" — the GUI has an onboarding page
    (App.tsx `Page = "onboarding" | ...`).
  - Future Ideas "Multiple censoring styles (pixelate, …)" — BlurMode
    BLUR/BOX/PIXELATE exists (config.py); mark pixelate done, keep emoji
    overlay as the open idea.
  - The September speed-offensive section's last unchecked box ("Fresh
    before/after benchmark tables") — bench tables ARE fresh as of v0.2.1
    (docs/performance.md + BENCHMARKS.md); decide whether real before/after
    on pre-offensive code is worth a bench-archaeology run or close it as
    obsolete (recommend: close, the narrative lives in performance.md).
- Restructure the rest into a single prioritized "Now / Next / Later" board
  (drop the version-number buckets; they've stopped matching reality —
  v0.3.0 items shipped before v0.2.0 items).

### Step 1 — Quick wins (each its own PR, in this order)
1. **Per-category threshold controls in CLI** (`--threshold-kiss`,
   `--threshold-clip`, … or a `--thresholds file.json`). Config already has
   effective-threshold plumbing (`config.get_effective_thresholds()`); this
   is surface area + tests, not pipeline work.
2. **Expected time estimator.** `process`/`plan` print "≈ N min remaining"
   using probed fps/frames × the profile's per-frame cost; calibrate the
   constant from `pureframe bench` medians. Print early, update per phase.
3. **Temporal tracking improvements (box jitter).** `smooth_detections`
   (pipeline/smooth.py) runs an IoU tracker; add hysteresis/box-EMA and a
   synthetic jitter test (wobbly box sequence in → stable sequence out).
   Careful: eval-parity gates *detection scores*, not boxes — but keep it
   that way consciously; if you change tracking defaults, eyeball the demo
   GIF (`uv run python scripts/make_demo.py`) before/after.
4. **Cached model inference per video hash.** Cache key = (content hash,
   config_hash); store verdicts/boxes so a re-run with identical config
   skips plan inference. The checkpoint store (jobs.db) already persists
   verdicts — extend rather than invent. `--no-cache` escape. Tests must
   cover: config change → miss; file change → miss; same → hit.

### Step 2 — GUI track (reuse the shim workflow)
5. **Timeline scrubbing** (roadmap: "Tauri desktop GUI with timeline
   scrubbing"). Plan editor timeline → seekable scrubber over shot
   thumbnails (extract_thumbnail already exists server-side).
6. **Before/after preview.** Side-by-side original vs blurred frame for the
   selected shot. CLI side: `pureframe preview` already renders contact
   sheets — add a `--before-after` mode writing paired images the GUI can
   load via load_plan-style IPC, or render both frames in Rust. Prefer the
   Python-side mode (no Rust changes, testable).

### Step 3 — Bigger rocks (scope in a design note PR before coding)
7. **Plugin API for custom detectors.** Design first: a detector = class
   implementing `detect_batch(frames)->[Detection]` + label→category map +
   threshold, registered via entry points (`pureframe.plugins`). The fuse()
   path must treat non-nudity categories as box-providers (post-#63 the
   boxes plumbing exists). Ship with one example plugin + docs page.
8. **Real-world benchmark suite.** `pureframe bench` has `--input`-less
   synthetic clips; add `bench --real <file>` that times a user-supplied
   file per phase and appends to a local JSONL. Do NOT ship any copyrighted
   sample; document the workflow (user runs it on their own file, submits
   the JSONL).
9. **Multi-GPU support.** Only if a design lands cleanly: NudeNet session
   per CUDA device, shot-level sharding in the pipeline worker. If it
   turns into surgery on the checkpoint store, descope to "single active
   device selection" (`--device cuda:1`) and note the rest as Later.

### Blocked on LO (surface, don't grind)
- **macOS code-signing + notarization** and **Tauri auto-updater with
  signed manifests**: both need a paid Apple Developer cert ($99/yr) and
  updater keys. Prepare the exact steps + GitHub secrets list as a doc
  (`docs/release-signing.md`), get the cert decision from LO, stop there.
- **Demo video / comparison page**: assets/marketing decisions — draft
  outlines, ask LO.

### Verification habits (per PR)
`uv run ruff check pureframe tests scripts packaging && uv run ruff format
--check pureframe tests scripts packaging && CUDA_VISIBLE_DEVICES="" uv run
pytest -q -m "not slow"` (299+ tests, ~60s). Model loads are safe locally;
CI remains the backbone for heavy checks. For render-path changes, also run
the slow suite locally: `uv run pytest -q tests/test_e2e.py`. After any
renderer/tracker change, regenerate + eyeball the demo GIF.

### Don't
- Don't re-litigate the kernel/crash archaeology or the v0.2.0 incident
  beyond what the tests now pin.
- Don't post anything anywhere (Show HN / Reddit drafts live untracked in
  DRAFT_*.md for LO to review).
- Don't touch ~/Downloads contents outside the repo dir.
- Don't bump versions or cut releases without LO's explicit go — that's
  his call, every time.
