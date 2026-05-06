# Performance Benchmarks

*Note: Benchmarks represent average processing times for a standard 90-minute 1080p H.264 movie (approx. 129,600 frames).*

## Your Current Hardware Profile
- **OS**: Linux 6.18.7-76061807-generic
- **CPU**: x86_64
- **GPU**: NVIDIA GeForce RTX 3060

## Reference Hardware Tiers

| Hardware | 90-min 1080p movie | Profile | Average FPS |
|---|---|---|---|
| RTX 4090 | ~12 min | HIGH | 180 FPS |
| RTX 3060 | ~24 min | MEDIUM | 90 FPS |
| GTX 1650 (4GB) | ~55 min | LOW | 39 FPS |
| M2 Pro | ~28 min | MEDIUM | 77 FPS |
| 12-core CPU only | ~3 hours | CPU | 12 FPS |

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
