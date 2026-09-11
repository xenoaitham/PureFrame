# Performance: the low-end-PC speed offensive

*Written September 2026, during the revival. Goal: a 90-minute movie should
not take 5+ hours on a machine without a dedicated GPU.*

This document is the engineering narrative of the optimization campaign -
what was slow, how we found it, what changed, and how we prove nothing
broke. Every optimization here is gated by CI: the `eval-parity` job runs
the real detector on a fixed synthetic corpus and fails on any behavioral
drift, and `pureframe bench` measures before/after per phase.

## TL;DR - what changed

| Change | Phase affected | Effect | PR |
|---|---|---|---|
| Seek-based frame extraction (was: re-decode from frame 0 per shot) | plan | removes a near-quadratic decode cost - the single biggest fix | #53 |
| NudeNet ONNX session stays loaded across calls (was: re-init per densified frame on CPU/LOW) | plan | dozens of full model re-loads per flagged shot eliminated | #53 |
| Audio classifier runs only when its score can matter | plan | skips the per-shot PANNs forward + audio decode for most shots | #54 |
| Scene-detection `frame_skip` (MEDIUM 1, LOW/CPU 2) | plan | full-video decode cost cut 2–3× | #54 |
| Kiss shots sampled at the densify stride (was: every frame) | plan | long kiss shots stop dominating the plan stage | #54 |
| Encoder presets per profile (was: ffmpeg's implicit `medium`) | render | 2–3× faster re-encode of dirty segments | #56 |
| `select_hw_encoder` + fps resolved once per render (was: per segment) | render | N× fewer subprocess spawns and probes | #56 |
| One less full-frame copy per dirty frame (`memoryview`) | render | ~6 MB per frame at 1080p | #56 |
| Pipelined plan generation (extraction ∥ inference) | plan | decode overlaps model compute on all profiles | #57 |
| int8 dynamic quantization of NudeNet on CPU profiles | detect | ~2–4× CPU inference, eval-parity gated | #58 |
| PANNs audio capped to the center 10 s of long shots | plan | long shots stop scaling linearly | #58 |

## The investigation

The code was *correct* - the CI suite was green - but five separate
inefficiencies hid behind that green:

1. **The quadratic decode.** `extract_frames` built a `select=eq(n,i)+...`
   filter with **no input seek**, so ffmpeg decoded the video from frame 0
   and discarded everything before the requested frame. A shot at minute 60
   paid an hour of decoding to grab two frames - and every shot paid it
   again. Across ~1,000 shots, that dwarfs all model inference combined.
   Found by reading the ffmpeg invocation against the shot loop, not by
   profiling: the structure alone is damning.

2. **The self-inflicted model reload.** `NudityDetector` unloaded its ONNX
   session after *every* `detect_batch` call when `keep_models_loaded=False`
   (the CPU/LOW default). Since `densify_shot` calls it once per frame, each
   densified frame paid a full ONNX session re-initialization - a memory
   conservation measure that fired inside a loop that immediately reloaded
   the model anyway.

3. **Eager context models.** `fuse()` consults the CLIP scene context and
   the PANNs audio result in every branch *except* the decisive-nudity
   branch - but both models ran eagerly for every shot, including shots
   where their scores could not change the verdict. Closer reading showed
   the two sexual-act branches *require* the scene signal to clear its own
   threshold before audio can matter at all, which makes the audio gate
   provably verdict-neutral to skip.

4. **Dead configuration.** `detection_batch_size` (32/16/4/1 per profile)
   had zero consumers - NudeNet 3.x has no batched inference, and the
   wrapper looped per frame regardless. Config that lies is worse than no
   config.

5. **The rendering round-trip.** Dirty segments re-encoded through a Python
   rawvideo pipe (double colorspace conversion, two full-frame copies per
   frame, no x264 preset - ffmpeg's `medium` default costs 2–3× encode
   time), plus 4–5 subprocess spawns and two probes *per segment*.

## The fixes, and why they're safe

Every change above carries its own safety story:

- **Seek-based extraction** is pinned by pixel-parity tests: the seek path
  must return frames identical to a sequential decode on a
  constant-framerate clip (seeded synthetic colors, scattered indices,
  ground truth from OpenCV).
- **The audio gate** (`fuse.context_audio_needed`) mirrors `fuse`'s own
  decision boundaries and is pinned by parity tests: below the scene
  thresholds, a neutral AudioContext and a maximal one produce the
  identical verdict, byte for byte.
- **Quantization** ships behind three guards: a fallback to fp32 on any
  quantization error, a `--no-quant` escape hatch, and the **eval-parity CI
  job** - the quantized model runs the evaluation corpus in CI against the
  committed baseline signature, so any behavioral drift fails the PR before
  it can ship.
- **frame_skip** degrades scene-boundary precision to ±2 frames - covered
  by the renderer's 0.5 s segment padding, which existed for exactly this
  class of imprecision.

## Measuring it yourself

```bash
# Per-phase timings for a single run (verbose table):
pureframe process movie.mp4 --verbose

# The full benchmark matrix (generates a detection-exercising synthetic
# clip, isolates checkpoint state, reports medians + phase breakdown):
pureframe bench --duration 30 --reps 3 -o bench-report.json
```

`bench` deliberately does **not** benchmark a zero-detection clip (the old
approach measured everything except the expensive parts) and does not wipe
model caches between reps (that benchmarks downloads, not PureFrame).

## Measured results (September 2026)

`pureframe bench --duration 30 --reps 3` on the author's machine - **12 cores,
RTX 3060, Linux 7.1.5, pureframe 0.1.0b16** - after the entire offensive:

| Profile | 30 s clip (median) | Detections | Top phases |
|---|---:|---:|---|
| CPU | **3.0 s** | 0 | scene_detect 0.7 · extract 0.4 · nudity 0.2 |
| LOW | **15.1 s** | 1 | render 4.1 · faces 4.0 · scene 0.6 |
| MEDIUM | **16.2 s** | 1 | faces 5.8 · render 4.0 · scene 0.7 |
| HIGH | **23.7 s** | 1 | faces 12.2 · render 4.0 · extract_kiss 1.6 |

What the per-phase data proves:

- **CPU profile: 10× realtime** on the synthetic clip - the audio gate kept
  PANNs out entirely (`detect_audio` absent), CLIP is disabled by design
  (0.0 s), and the int8-quantized NudeNet classified the whole clip in
  0.2 s.
- **The audio gate works as designed on GPU profiles too**: audio ran for
  exactly one context-worthy shot (0.07 s) instead of every shot.
- **HIGH costs what you'd expect**: `detect_faces 12.2 s` is the max-quality
  `densify_every_n_frames=1` sampling doing per-frame face detection - the
  honest price of maximum coverage, not a regression.
- `render 4.0 s` on GPU profiles is nvenc's full re-encode of the flagged
  shot (the bench's single flagged shot covers 100 % of its duration, so the
  smart renderer correctly falls back to full re-encode - worst case by
  construction).

Honest caveats: this is a 30 s synthetic clip with 1–2 shots, not a movie.
A real film's hundreds of mostly-clean shots amortize the per-shot work
differently, and the CPU profile's zero-detection run under-represents real
content. Reproduce with `pureframe bench --duration 30 --reps 3`; real-movie
benchmarks are tracked in the roadmap (v1.0, "Real-world benchmark suite").

## Before/after

To be filled from `pureframe bench` runs on reference hardware. The numbers
below are the *pre-optimization* extrapolations from the author's RTX 3060
machine and are kept until real post-optimization numbers replace them.

| Profile | 90-min movie (before) | Target |
|---|---:|---:|
| CPU | ~64+ min (likely far worse with detections - the quadratic decode hid it) | ~10–20 min |
| MEDIUM | ~124 min | ~30 min |

## Benchmarks on your own files

`pureframe bench --real yourfile.mp4 --profiles LOW,MEDIUM --reps 3` times the
full process flow on content you own, with a per-phase breakdown, and appends
one JSON line per run to `pureframe_bench_real.jsonl` so runs accumulate.
Records identify the file only by its SHA-256 and basic metadata - never a
path or filename - so the JSONL is safe to share when comparing machines.

No copyrighted sample is shipped with PureFrame and none is requested: run it
on your own files and, if you want to contribute numbers, submit the JSONL
with a note about your hardware (CPU/GPU, RAM) via a GitHub discussion or
issue.
