"""Expected-time estimates for `process` / `plan`.

Constants are calibrated from the published v0.2.1 bench run
(`BENCHMARKS.md`, 30 s @ 30 fps = 900 frames on the RTX 3060 reference
machine, `bench-report.json`):

- analysis per frame = (total − render) / 900 per profile — the whole plan
  stage minus the render phase;
- full re-encode per frame ≈ 4.1 s / 900 — the bench clip is a single
  flagged shot, so its render phase re-encodes every frame (nvenc on the
  reference GPU);
- stream copy per frame ≈ 0.07 s / 900 — the CPU-profile bench clip flagged
  nothing, so its render phase is a pure copy (dominated by ffmpeg startup).

These are order-of-magnitude calibrations, not guarantees: real content
varies with shot count, kiss densify, and whether the smart path can copy.
Numbers are prefixed with "≈" everywhere they surface.
"""

from pureframe.hardware import HardwareProfile

# (total − render) / 900 from the v0.2.1 bench medians, seconds per frame.
ANALYSIS_SPF = {
    HardwareProfile.CPU: 0.0032,
    HardwareProfile.LOW: 0.0122,
    HardwareProfile.MEDIUM: 0.0135,
    HardwareProfile.HIGH: 0.0219,
}

# Full re-encode, seconds per frame. nvenc number from the bench render
# phase; libx264 (CPU profile) calibrated to ~50 fps 720p veryfast — the
# bench has no CPU-profile re-encode data point.
REENCODE_SPF = {
    HardwareProfile.CPU: 0.020,
    HardwareProfile.LOW: 0.0046,
    HardwareProfile.MEDIUM: 0.0046,
    HardwareProfile.HIGH: 0.0046,
}

# Stream copy, seconds per frame (CPU-profile bench render phase: pure copy).
COPY_SPF = 0.0001


def _profile(profile: HardwareProfile | None) -> HardwareProfile:
    return profile if profile is not None else HardwareProfile.CPU


def estimate_analysis_seconds(
    profile: HardwareProfile | None, total_frames: int
) -> float:
    """Plan-stage estimate: probe + shots + detection + densify + fuse."""
    return ANALYSIS_SPF[_profile(profile)] * max(total_frames, 0)


def estimate_render_seconds(
    profile: HardwareProfile | None, total_frames: int, flagged_frames: int
) -> float:
    """Render estimate: flagged frames re-encode, the rest stream-copy."""
    flagged = min(max(flagged_frames, 0), max(total_frames, 0))
    clean = max(total_frames, 0) - flagged
    p = _profile(profile)
    return REENCODE_SPF[p] * flagged + COPY_SPF * clean


def format_duration(seconds: float) -> str:
    """Human "≈" duration: seconds under a minute, then minutes, then hours."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"≈ {seconds:.0f} s"
    if seconds < 3600:
        return f"≈ {seconds / 60:.1f} min"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"≈ {hours} h {minutes:02d} min"
