# Performance Benchmarks

## September 2026 - Speed offensive validation (`pureframe bench`)

Measured with `pureframe bench --duration 30 --reps 3` (medians) on the
author's machine, **after** the low-end-PC speed offensive - see
[docs/performance.md](docs/performance.md) for what changed:

| | |
|---|---|
| **OS** | Pop!_OS 24.04, kernel **7.1.5-76070105** |
| **CPU** | Intel Core i5-10400F (12 threads) |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| **pureframe** | 0.1.0b16 + speed offensive |

| Profile | 30s 720p bench clip (median) | Detections | Top phases |
|---|---:|---:|---|
| CPU | **3.0s** | 0 | scene_detect 0.7s · extract 0.4s · nudity 0.2s |
| LOW | **15.1s** | 1 | render 4.1s · detect_faces 4.0s · scene 0.6s |
| MEDIUM | **16.2s** | 1 | detect_faces 5.8s · render 4.0s · scene 0.7s |
| HIGH | **23.7s** | 1 | detect_faces 12.2s · render 4.0s · extract_kiss 1.6s |

Notes: the bench clip is synthetic with one flagged shot; per-phase medians
come from the built-in `PhaseTimers`. The audio classifier ran for exactly
one context-worthy shot on GPU profiles (the lazy audio gate); CLIP is
disabled on the CPU profile by design. Reproduce with
`pureframe bench --duration 30 --reps 3 -o report.json`.

## Legacy full-scale measurements (pre-offensive, for reference)

> Measurements below are **real, on the machine indicated**. Other tiers are estimated from community contributions. To add your hardware, run `scripts/run_benchmarks.sh` and open a PR.

> **Note on these numbers:** The benchmark clip is a 30s synthetic 1080p video (solid colours, test patterns) with **zero detections**. Real movie content with actual detections will take longer - blurring adds per-frame overhead, and complex scenes trigger more model inference. Use these numbers as a floor, not a ceiling.

## Measured: Author's Machine

| | |
|---|---|
| **OS** | Linux (Pop!_OS 24.04, kernel 6.18.7) |
| **CPU** | Intel Core i5-10400F @ 2.90GHz (6 cores) |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| **Python** | 3.13.12 |
| **PyTorch** | 2.11.0+cu130 |
| **CUDA** | 13.0 |

| Profile | 30s 1080p clip | Extrapolated 90-min movie | Detection FPS |
|---|---|---|---|
| HIGH (12 GB VRAM) | **25.96s** | ~78 min | ~28 fps |
| MEDIUM (6–11 GB VRAM) | **41.23s** | ~124 min | ~18 fps |
| LOW (3–5 GB VRAM) | **27.83s** | ~83 min | ~26 fps |
| CPU (no GPU) | **21.37s** | ~64 min* | ~34 fps* |

*CPU appears faster on synthetic content because CUDA initialization overhead is amortised differently and no inference is triggered on zero-detection clips. Expect CPU to be **3–10× slower** on real movie content.*

## Estimated: Other Hardware

Community contributions welcome. Run `scripts/run_benchmarks.sh` and open a PR to add your hardware below.

| Hardware | Profile | Estimated 90-min movie |
|---|---|---|
| RTX 4090 (24 GB) | HIGH | ~30 min (estimated, not measured) |
| RTX 4070 (12 GB) | HIGH | ~50 min (estimated, not measured) |
| GTX 1650 (4 GB) | LOW | ~120 min (estimated, not measured) |
| Apple M2 Pro | MEDIUM | ~90 min (estimated, not measured) |
| Apple M4 Max | HIGH | ~35 min (estimated, not measured) |

## Profile Knobs

### HIGH (12+ GB VRAM)
- Detection resolution: 1080px
- Batch size: 32 frames
- FP16 inference: enabled
- Models kept in VRAM between stages
- Samples 5 keyframes per shot + densifies every frame

### MEDIUM (6–11 GB VRAM)
- Detection resolution: 720px
- Batch size: 16 frames
- FP16 inference: enabled
- Models kept in VRAM between stages
- Samples 3 keyframes per shot, densifies every 2nd frame

### LOW (3–5 GB VRAM)
- Detection resolution: 540px
- Batch size: 4 frames
- FP16 inference: enabled
- Models unloaded between stages (reduces VRAM pressure)
- Samples 3 keyframes per shot, densifies every 3rd frame

### CPU (no GPU)
- Detection resolution: 480px
- Batch size: 1 frame
- FP16 disabled (CPU doesn't benefit)
- Models unloaded between stages
- Samples 2 keyframes per shot, densifies every 5th frame

## How to Contribute Benchmarks

1. Clone the repo and install: `pip install -e .`
2. Generate the benchmark clip and run timings:
   ```bash
   bash scripts/run_benchmarks.sh
   ```
3. Capture the following per profile (HIGH / MEDIUM / LOW / CPU):
   - Wall-clock seconds on the 30s 1080p synthetic clip
   - Detection FPS reported by `pureframe process --profile <P>`
   - OS, CPU model, GPU model + VRAM, Python, PyTorch, CUDA versions
4. Open a PR replacing the matching row in the "Estimated" table with a "measured" row, formatted like the **Measured** section at the top.

### Methodology notes

- The synthetic clip is intentionally detection-light. **Do not** interpret the extrapolated 90-min movie figure as a guarantee - real content typically lands 2-10× slower due to per-frame inference on dense scenes plus localized blur encode cost.
- HIGH profile runs FP16 on CUDA; CPU profile runs FP32 on a single thread per model.
- Smart segment renderer is enabled by default - figures already reflect stream-copy of clean segments.
- Audio classifier runs on GPU when available; PANNs adds ~3-5% to wall-clock on 90-min content.
