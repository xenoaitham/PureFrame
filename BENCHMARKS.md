# Performance Benchmarks

> **⚠️ IMPORTANT: These are TARGET numbers, not measured results.**
>
> The numbers below represent estimated performance goals based on architecture analysis,
> **not actual timed runs on real hardware.** To generate real measurements for your system,
> run `scripts/bench.py`.

---

*Estimates are for a standard 90-minute 1080p H.264 movie (approx. 129,600 frames).*

## Your Current Hardware Profile
- **OS**: Linux 6.18.7-76061807-generic
- **CPU**: x86_64
- **GPU**: NVIDIA GeForce RTX 3060

## Reference Hardware Tiers

| Hardware | 90-min 1080p movie | Profile | Average FPS |
|---|---|---|---|
| RTX 4090 | ~12 min* (target) | HIGH | ~180 FPS* (target) |
| RTX 3060 | ~24 min* (target) | MEDIUM | ~90 FPS* (target) |
| GTX 1650 (4GB) | ~55 min* (target) | LOW | ~39 FPS* (target) |
| M2 Pro | ~28 min* (target) | MEDIUM | ~77 FPS* (target) |
| 12-core CPU only | ~3 hours* (target) | CPU | ~12 FPS* (target) |

*\* All numbers are targets. Run benchmarks locally to get real data.*

## Detailed Profile Scaling

### HIGH Profile
- Image size: 640px
- Batch size: 16
- Target usage: High-end dedicated GPUs (RTX 3080, RTX 4090)

### MEDIUM Profile
- Image size: 480px
- Batch size: 8
- Target usage: Mid-range GPUs (RTX 3060, M1/M2 Max)

### LOW Profile
- Image size: 320px
- Batch size: 4
- Target usage: Entry-level dedicated GPUs (GTX 1650)

### CPU Profile
- Image size: 320px
- Batch size: 1
- Target usage: Integrated graphics or pure CPU execution

## How to Run Benchmarks

To generate real measurements on your hardware:

```bash
python scripts/bench.py
```

This will detect your CPU, GPU, and OS, then populate this file with actual timing data. You can also modify `scripts/bench.py` to run against a specific test video for end-to-end timing.
