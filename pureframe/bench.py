"""`pureframe bench` — repeatable performance benchmark.

Generates a synthetic clip that actually exercises the pipeline (moving
skin-tone regions trigger detections; a silent audio track exercises the
audio classifier), runs the `process` flow once per profile/rep against an
isolated checkpoint directory, and writes a JSON + Markdown report with
per-phase timings.

The inner runs invoke the Typer app in-process (CliRunner) rather than
spawning a subprocess: no binary discovery, deterministic capture, and the
only environment switch needed is `PUREFRAME_DATA_DIR`, which
`cli.get_store()` honors for checkpoint isolation. Model caches are shared
across runs on purpose — wiping them would benchmark downloads, not
PureFrame.
"""

import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from typer.testing import CliRunner

PHASES = (
    "probe",
    "scene_detect",
    "extract",
    "extract_kiss",
    "detect_nudity",
    "detect_clip",
    "detect_audio",
    "detect_faces",
    "densify",
    "fuse",
    "render",
)

BENCH_PROFILES = ("CPU", "LOW", "MEDIUM", "HIGH")


def generate_bench_clip(
    path: Path, duration: float = 30.0, width: int = 1280, height: int = 720
) -> Path:
    """Render the synthetic benchmark clip with ffmpeg.

    Two skin-tone boxes sweep the frame so NudeNet has something to fire on
    (the previous bench clip was flat grey with zero detections, so its
    numbers never exercised detect/densify/blur). A silent mono track keeps
    the audio classifier in the measured path.
    """
    box1 = (
        f"drawbox=x='(iw-300)/2+{width // 4}*sin(2*PI*t/10)':"
        f"y='(ih-400)/2+{height // 5}*cos(2*PI*t/7)':"
        "w=300:h=400:color=0xE0AC69@1:t=fill"
    )
    box2 = (
        f"drawbox=x='(iw-200)/2-{width // 5}*sin(2*PI*t/13+1)':"
        f"y='(ih-260)/2+{height // 6}*sin(2*PI*t/5)':"
        "w=200:h=260:color=0xD8A074@1:t=fill"
    )
    vf = f"{box1},{box2}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=duration={duration}:size={width}x{height}:rate=30",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=32000:cl=mono",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-shortest",
            # Output-side hard stop: with two lavfi inputs, newer ffmpeg
            # builds let -shortest overshoot badly (a 2 s request muxed
            # ~8 s of anullsrc on a macOS runner), so pin the mux duration.
            "-t",
            f"{duration}",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )
    return path


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        return out.stdout.splitlines()[0]
    except Exception:
        return "unknown"


def _package_version() -> str:
    try:
        return version("pureframe")
    except PackageNotFoundError:
        return "unknown"


def capture_environment() -> dict:
    return {
        "pureframe": _package_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ffmpeg": _ffmpeg_version(),
    }


def _parse_timers_payload(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PUREFRAME_TIMERS "):
            return json.loads(line[len("PUREFRAME_TIMERS ") :])
    return None


def _parse_flagged(payload: dict | None, stdout: str) -> int:
    if payload and "flagged_shots" in payload:
        return int(payload["flagged_shots"])
    m = re.search(r"Flagged (\d+) shots", stdout)
    return int(m.group(1)) if m else 0


def run_benchmark(
    profiles: list[str],
    reps: int = 1,
    duration: float = 30.0,
    width: int = 1280,
    height: int = 720,
    keep_clip: Path | None = None,
) -> dict:
    """Run the benchmark matrix and return the full report as a dict."""
    from pureframe.cli import app

    runner = CliRunner()

    workdir = Path(tempfile.mkdtemp(prefix="pureframe_bench_"))
    clip = keep_clip if keep_clip else workdir / "bench_clip.mp4"
    if not clip.exists():
        print(f"Generating benchmark clip ({duration:.0f}s, {width}x{height})...")
        generate_bench_clip(clip, duration, width, height)

    report: dict = {
        "environment": capture_environment(),
        "clip": str(clip),
        "profiles": {},
    }
    # The CLI emits its payload to BOTH the timers file and a stdout line;
    # enabling the stdout channel here gives bench a fallback if the file
    # mechanism ever hiccups.
    old_print_timers = os.environ.get("PUREFRAME_PRINT_TIMERS")
    os.environ["PUREFRAME_PRINT_TIMERS"] = "1"

    try:
        for profile in profiles:
            totals: list[float] = []
            phase_sums: dict[str, list[float]] = {}
            flagged_counts: list[int] = []
            reps_detail: list[dict] = []

            for rep in range(reps):
                out_path = workdir / f"out_{profile}_{rep}.mp4"
                timers_file = workdir / f"timers_{profile}_{rep}.json"
                with tempfile.TemporaryDirectory(prefix="pureframe_bench_db_") as td:
                    old_data_dir = os.environ.get("PUREFRAME_DATA_DIR")
                    os.environ["PUREFRAME_DATA_DIR"] = td
                    os.environ["PUREFRAME_TIMERS_FILE"] = str(timers_file)
                    started = time.perf_counter()
                    try:
                        result = runner.invoke(
                            app,
                            [
                                "process",
                                str(clip),
                                "--output",
                                str(out_path),
                                "--profile",
                                profile,
                                "--force",
                            ],
                        )
                    finally:
                        elapsed = time.perf_counter() - started
                        _restore_env("PUREFRAME_DATA_DIR", old_data_dir)
                        _restore_env("PUREFRAME_TIMERS_FILE", None)

                if result.exit_code != 0:
                    raise SystemExit(
                        f"bench run failed for profile {profile} (rep {rep}): "
                        f"{result.exception!r}\n"
                        f"CLI stdout (tail): {result.stdout[-2000:]}\n"
                        f"CLI stderr (tail): {getattr(result, 'stderr', '')[-1000:]}"
                    )

                # The timers file holds the bare JSON payload; the stdout
                # fallback line carries a "PUREFRAME_TIMERS " prefix, so the
                # two channels need different parsers.
                if timers_file.exists():
                    payload = json.loads(timers_file.read_text(encoding="utf-8"))
                else:
                    payload = _parse_timers_payload(result.stdout)
                if payload is None:
                    raise SystemExit(
                        f"no timer payload emitted for profile {profile} (rep {rep})\n"
                        f"CLI stdout (tail): {result.stdout[-2000:]}\n"
                        f"CLI stderr (tail): {getattr(result, 'stderr', '')[-1000:]}"
                    )
                timers = payload.get("phases", {})
                flagged = _parse_flagged(payload, result.stdout)

                totals.append(elapsed)
                flagged_counts.append(flagged)
                for name, d in timers.items():
                    phase_sums.setdefault(name, []).append(d["seconds"])
                reps_detail.append(
                    {
                        "rep": rep,
                        "total_seconds": round(elapsed, 2),
                        "timers": timers,
                    }
                )
                print(
                    f"[{profile}] rep {rep + 1}/{reps}: {elapsed:.1f}s, "
                    f"{flagged} shots flagged"
                )

            report["profiles"][profile] = {
                "reps": reps_detail,
                "total_seconds_median": round(statistics.median(totals), 2),
                "flagged_median": int(statistics.median(flagged_counts)),
                "phase_seconds_median": {
                    name: round(statistics.median(vals), 3)
                    for name, vals in phase_sums.items()
                },
            }

        return report
    finally:
        _restore_env("PUREFRAME_PRINT_TIMERS", old_print_timers)
        if not keep_clip:
            shutil.rmtree(workdir, ignore_errors=True)


def _restore_env(key: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old_value


def run_benchmark_real(
    video: Path,
    profiles: list[str],
    reps: int = 1,
    jsonl_path: Path | None = None,
) -> list[dict]:
    """Benchmark a user-supplied video, appending one JSON line per run.

    Same flow as the synthetic benchmark (CliRunner `process`, per-phase
    timers) but on real content, and results stream to a local JSONL so
    runs accumulate. Records are privacy-safe by construction: the file is
    identified by its SHA-256 and basic metadata — the path never leaves
    the machine, so a shared JSONL carries no filenames.
    """
    from pureframe.checkpoint import content_fingerprint
    from pureframe.cli import app
    from pureframe.pipeline.probe import probe_video

    runner = CliRunner()
    meta = probe_video(video)
    file_id = {
        "sha256": content_fingerprint(video),
        "duration_seconds": round(meta.duration_seconds, 2),
        "width": meta.width,
        "height": meta.height,
        "fps": float(meta.fps),
        "container": meta.container,
        "video_codec": meta.video_codec,
    }

    workdir = Path(tempfile.mkdtemp(prefix="pureframe_bench_real_"))
    sink = jsonl_path if jsonl_path else Path.cwd() / "pureframe_bench_real.jsonl"

    old_print_timers = os.environ.get("PUREFRAME_PRINT_TIMERS")
    os.environ["PUREFRAME_PRINT_TIMERS"] = "1"

    records: list[dict] = []
    try:
        for profile in profiles:
            for rep in range(reps):
                out_path = workdir / f"out_{profile}_{rep}.mp4"
                with tempfile.TemporaryDirectory(prefix="pureframe_bench_db_") as td:
                    old_data_dir = os.environ.get("PUREFRAME_DATA_DIR")
                    os.environ["PUREFRAME_DATA_DIR"] = td
                    os.environ.pop("PUREFRAME_TIMERS_FILE", None)
                    started = time.perf_counter()
                    try:
                        result = runner.invoke(
                            app,
                            [
                                "process",
                                str(video),
                                "--output",
                                str(out_path),
                                "--profile",
                                profile,
                                "--force",
                            ],
                        )
                    finally:
                        elapsed = time.perf_counter() - started
                        _restore_env("PUREFRAME_DATA_DIR", old_data_dir)
                        os.environ.pop("PUREFRAME_TIMERS_FILE", None)

                if result.exit_code != 0:
                    raise SystemExit(
                        f"bench run failed for profile {profile} (rep {rep}): "
                        f"{result.exception!r}\n"
                        f"CLI stdout (tail): {result.stdout[-2000:]}"
                    )

                payload = _parse_timers_payload(result.stdout)
                if payload is None:
                    raise SystemExit(
                        f"no timer payload emitted for profile {profile} (rep {rep})"
                    )

                record = {
                    "kind": "real-bench",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "environment": capture_environment(),
                    "file": file_id,
                    "profile": profile,
                    "rep": rep,
                    "total_seconds": round(elapsed, 2),
                    "flagged_shots": _parse_flagged(payload, result.stdout),
                    "phase_seconds": {
                        name: round(d["seconds"], 3)
                        for name, d in payload.get("phases", {}).items()
                    },
                }
                records.append(record)
                with open(sink, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"[{profile}] rep {rep + 1}/{reps}: {elapsed:.1f}s, "
                    f"{record['flagged_shots']} shots flagged → {sink}"
                )
    finally:
        _restore_env("PUREFRAME_PRINT_TIMERS", old_print_timers)
        shutil.rmtree(workdir, ignore_errors=True)

    return records


def summarize_real_records(records: list[dict]) -> str:
    """Markdown summary (per profile medians) from real-bench JSONL records."""
    if not records:
        return "No runs recorded."
    by_profile: dict[str, list[dict]] = {}
    for r in records:
        by_profile.setdefault(r["profile"], []).append(r)

    f = records[0]["file"]
    lines = [
        f"Real benchmark — {f['width']}x{f['height']} {f['duration_seconds']:.0f}s "
        f"{f['video_codec']}/{f['container']} (sha256 {f['sha256'][:12]}…)",
        "",
        "| Profile | Runs | Total (median) | Flagged | Top phases |",
        "|---|---|---:|---:|---|",
    ]
    for profile, runs in by_profile.items():
        totals = sorted(r["total_seconds"] for r in runs)
        median = (
            totals[len(totals) // 2]
            if len(totals) % 2
            else ((totals[len(totals) // 2 - 1] + totals[len(totals) // 2]) / 2)
        )
        flagged = statistics.median([r["flagged_shots"] for r in runs])
        phase_sums: dict[str, list[float]] = {}
        for r in runs:
            for name, s in r["phase_seconds"].items():
                phase_sums.setdefault(name, []).append(s)
        top = sorted(
            ((name, statistics.median(vals)) for name, vals in phase_sums.items()),
            key=lambda kv: -kv[1],
        )[:3]
        top_str = ", ".join(f"{n} {s:.1f}s" for n, s in top)
        lines.append(
            f"| {profile} | {len(runs)} | {median:.1f}s | {flagged:.0f} | {top_str} |"
        )
    return "\n".join(lines)


def report_to_markdown(report: dict) -> str:
    """Markdown table rows (for BENCHMARKS.md) from a report dict."""
    env = report["environment"]
    lines = [
        f"Benchmarked with pureframe {env['pureframe']} | {env['platform']} | "
        f"{env['cpu_count']} cores | {env['ffmpeg']}",
        "",
        "| Clip | Profile | Total (median) | Detections | Top phases |",
        "|---|---|---:|---:|---|",
    ]
    for profile, data in report["profiles"].items():
        top = sorted(data["phase_seconds_median"].items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"{n} {s:.1f}s" for n, s in top)
        lines.append(
            f"| {Path(report['clip']).name} | {profile} | "
            f"{data['total_seconds_median']:.1f}s | {data['flagged_median']} | {top_str} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print("Run via: pureframe bench", file=sys.stderr)
