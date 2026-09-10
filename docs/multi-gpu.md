# Multi-GPU support — design note

Status: **descope shipped, full sharding stays Later.** PureFrame now honors
`--device <N>` (0-based CUDA index) on `process`/`plan`; multi-GPU
*sharding* — one model per GPU, shots distributed across them — is analyzed
below and deliberately not built.

## What ships: single active device selection (`--device`)

Multi-GPU machines (one big GPU + one small, or two mid-range cards) could
only ever use GPU 0: ONNX Runtime defaults to `device_id 0`, torch put CLIP
and PANNs on `cuda:0`, and profile auto-detection read VRAM from device 0.
`--device 1` now pins everything ML-related to the second GPU:

- `detect_profile(device)` profiles free VRAM on the chosen device, so a
  small second GPU correctly picks a lighter profile instead of inheriting
  device 0's;
- the NudeNet session runs the CUDA EP with `{"device_id": N}` provider
  options (`onnx_providers_for` in `hardware.py`);
- CLIP and PANNs move to `cuda:<N>` (`torch_device_str`), falling back to
  CPU cleanly when the index doesn't exist.

`device` is excluded from the checkpoint config hash on purpose: it is a
performance knob, not a detection decision — a cached verdict computed on
GPU 0 is valid on GPU 1. (Float results can differ in the last ulp between
devices; both are "correct" and eval-parity keeps its CPU-pinned baseline.)

## What stays Later: true multi-GPU sharding

The original idea — one NudeNet session per CUDA device, shots sharded
across them — measured against the current pipeline:

- **The plan loop is a single consumer.** Shots stream through one
  prefetch thread into one inference loop. Sharding needs a worker pool
  with one detector instance per device and result reassembly by shot
  index — a rewrite of the pipeline's heart, not a parameter.
- **The checkpoint store becomes the bottleneck.** Verdict writes happen
  from the consumer thread today; N writers mean either a single-writer
  queue (serialization point — the sharding gains leak away on shot-heavy
  content) or multi-writer SQLite with busy-handling (surgery on the store
  the checkpoint trust semantics depend on).
- **The gains are narrower than they look.** Nudity detection is 0.1–0.5 s
  of a ~16–24 s profile run on the reference machine (`bench-report.json`);
  scene detection, extraction, kiss densify (face detector: CPU-only) and
  the render dominate. Sharding the smallest phase across two GPUs buys
  single-digit percent end-to-end. Amdahl wins.
- **Batch-size interaction.** MEDIUM/HIGH already run detection_batch_size
  16/32; one modern GPU saturates those batches on 720p content. Two
  devices halve per-device batch residency but add inter-device
  synchronization.

When it's worth revisiting: 4K-first pipelines (detection resolution
1080+), CPU-profile deployments where nudity is the dominant phase, or a
move to batch-across-shots inference. The `--device` plumbing (pinned
providers, torch strings, per-device VRAM profiling) is the foundation
that work would build on.

## If LO ever wants sharding

The least-surgery shape would be process-level: N worker processes, each
pinned to one device via the existing `--device`, each handling a
contiguous shot range with its own checkpoint job segment, results merged
by the existing plan loader (verdicts are already per-shot rows). That
reuses the checkpoint store untouched — but it's a different orchestration
layer (segment scheduling, failure recovery per worker), estimated as its
own project, not a PR.
