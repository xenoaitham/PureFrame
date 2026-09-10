import json
import os
import subprocess

# When running as a PyInstaller-frozen executable, prepend the executable's
# directory to PATH so a co-bundled ffmpeg/ffprobe is discovered without the
# user installing it system-wide. Safe no-op for normal pip installs.
import sys as _sys
import tempfile
import threading
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from queue import Full, Queue
from uuid import uuid4

import platformdirs
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from pureframe.checkpoint import CheckpointStore
from pureframe.config import Config, ContentType, Strictness, load_thresholds_file
from pureframe.eta import (
    estimate_analysis_seconds,
    estimate_render_seconds,
    format_duration,
)
from pureframe.hardware import HardwareProfile, detect_profile, get_settings
from pureframe.pipeline.densify import densify_shot
from pureframe.pipeline.detect.audio import AudioClassifier, AudioContext
from pureframe.pipeline.detect.face import FaceDetector
from pureframe.pipeline.detect.nudity import NudityDetector
from pureframe.pipeline.detect.scene_clip import SceneClassifier
from pureframe.pipeline.fuse import context_audio_needed, fuse
from pureframe.pipeline.probe import probe_video
from pureframe.pipeline.render.apply import apply_censoring
from pureframe.pipeline.render.plan import CensorPlan
from pureframe.pipeline.sample import extract_frames, sample_keyframes
from pureframe.pipeline.shots import Action, Category, ShotVerdict, detect_shots
from pureframe.pipeline.smooth import smooth_detections
from pureframe.utils.ffmpeg import PureFrameError
from pureframe.utils.logging import setup_logging
from pureframe.utils.timing import PhaseTimers

if getattr(_sys, "frozen", False):
    _exe_dir = os.path.dirname(_sys.executable)
    os.environ["PATH"] = _exe_dir + os.pathsep + os.environ.get("PATH", "")

app = typer.Typer(help="PureFrame CLI")
jobs_app = typer.Typer(help="Manage jobs and checkpoints")
app.add_typer(jobs_app, name="jobs")

console = Console()


def version_callback(value: bool) -> None:
    if value:
        try:
            package_version = version("pureframe")
        except PackageNotFoundError:
            package_version = "unknown"
        typer.echo(f"PureFrame {package_version}")
        raise typer.Exit()


@app.callback()
def main(
    version_option: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show PureFrame version and exit.",
    ),
) -> None:
    pass


def get_store() -> CheckpointStore:
    # PUREFRAME_DATA_DIR isolates checkpoint state (used by `pureframe bench`
    # and tests); default to the standard per-user data directory.
    data_dir = os.environ.get("PUREFRAME_DATA_DIR") or platformdirs.user_data_dir(
        "PureFrame"
    )
    db_path = Path(data_dir) / "jobs.db"
    return CheckpointStore(db_path)


def _threshold_overrides(
    thresholds_file: Path | None,
    nudity: float | None,
    clip: float | None,
    audio: float | None,
) -> dict[str, float]:
    """Merge a ``--thresholds`` file with the per-category flags; flags win."""
    overrides = load_thresholds_file(thresholds_file) if thresholds_file else {}
    for category, value in (("nudity", nudity), ("clip", clip), ("audio", audio)):
        if value is not None:
            overrides[category] = value
    return overrides


def _build_config(**kwargs) -> Config:
    """``Config.from_cli`` with validation errors reported as CLI errors."""
    try:
        return Config.from_cli(**kwargs)
    except ValueError as e:
        console.print(f"[red]Invalid configuration:[/red] {e}")
        raise typer.Exit(2) from e


def _extraction_worker(
    shots,
    completed_indices,
    config,
    settings,
    meta,
    timers,
    out_queue: Queue,
    stop_event: threading.Event,
) -> None:
    """Prefetch keyframe extraction while the main thread runs inference.

    ffmpeg decode is a subprocess and ONNX/torch release the GIL during
    compute, so overlapping them shortens the plan stage on every profile.
    FIFO order keeps results deterministic; errors are forwarded in-band.
    """

    def _put(item) -> bool:
        """Queue *item*; False only if the consumer asked us to stop.

        The caller uses the result to bail out of the shot loop, so a bare
        ``return`` here (v0.2.0–v0.2.1) read as "stop" after the very first
        shot — every later shot went unanalyzed and uncensored.
        """
        while not stop_event.is_set():
            try:
                out_queue.put(item, timeout=0.5)
                return True
            except Full:
                continue
        return False

    try:
        for shot in shots:
            if shot.index in completed_indices:
                if not _put((shot, [], {})):
                    return
                continue
            kf_indices = sample_keyframes(shot, settings.sample_keyframes_per_shot)
            with timers.phase("extract"):
                frames = extract_frames(
                    config.input_path,
                    kf_indices,
                    settings.detection_resolution,
                    meta=meta,
                )
            if not _put((shot, kf_indices, frames)):
                return
    except Exception as e:
        _put(e)
    finally:
        _put(None)


def generate_plan(config: Config, timers: PhaseTimers | None = None) -> CensorPlan:
    timers = timers or PhaseTimers()
    store = get_store()
    job = store.find_or_create_job(config.input_path, config.output_path, config)

    settings = get_settings(config.profile)

    try:
        package_version = version("pureframe")
    except PackageNotFoundError:
        package_version = "unknown"
    console.print(f"[bold blue]PureFrame[/bold blue] v{package_version}")
    console.print(f"Job ID: {job.id}")
    console.print(f"Profile: [bold]{settings.profile.value}[/bold]")
    console.print(f"Input: {config.input_path}")

    if job.status == "DONE" or job.status == "RENDERING":
        verdicts = store.load_verdicts(job.id)
        shots = detect_shots(config.input_path, frame_skip=settings.scene_frame_skip)
        meta = probe_video(config.input_path)
    else:
        store.update_status(job.id, "DETECTING")

        with console.status("[bold green]Probing video..."), timers.phase("probe"):
            meta = probe_video(config.input_path)

        if meta.total_frames > 0:
            console.print(
                f"Analysis estimate: {format_duration(estimate_analysis_seconds(settings.profile, meta.total_frames))} "
                f"({meta.total_frames} frames, live remaining time in the bar below)"
            )

        with (
            console.status("[bold green]Detecting shots..."),
            timers.phase("scene_detect"),
        ):
            shots = detect_shots(
                config.input_path, frame_skip=settings.scene_frame_skip
            )

        store.update_status(job.id, "DETECTING", total_shots=len(shots))
        console.print(f"Detected {len(shots)} shots.")

        detector = NudityDetector(settings, quantize=config.quantize_cpu)
        scene_classifier = SceneClassifier(settings)
        if config.no_clip:
            scene_classifier.enabled = False

        # Defer constructing the audio model when audio is disabled or the
        # input has no audio streams. The PANNs ctor downloads a ~300MB
        # checkpoint via wget on first use which can hang in CI.
        audio_enabled = not (config.no_audio or len(meta.audio_streams) == 0)
        audio_classifier = AudioClassifier(settings, enabled=audio_enabled)

        face_detector = FaceDetector()

        existing_verdicts = store.load_verdicts(job.id)
        completed_indices = {v.shot_index for v in existing_verdicts}

        # Densify must keep every detection fuse() could have flagged on.
        # Filtering at the raw CLI value instead (the default 0.55) dropped
        # the boxes of shots flagged by a lower preset — e.g. a 0.40 score
        # under --strictness high — leaving BLACK_BOX verdicts with no boxes.
        eff_nudity, _, _ = config.get_effective_thresholds()
        densify_threshold = eff_nudity * (0.85 if config.strict else 1.0)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing shots...",
                    total=len(shots),
                    completed=len(completed_indices),
                )

                # Prefetch the next shot's keyframe extraction while the
                # current one is being classified.
                prefetch: Queue = Queue(maxsize=2)
                stop_event = threading.Event()
                worker = threading.Thread(
                    target=_extraction_worker,
                    args=(
                        shots,
                        completed_indices,
                        config,
                        settings,
                        meta,
                        timers,
                        prefetch,
                        stop_event,
                    ),
                    name="pureframe-extract",
                    daemon=True,
                )
                worker.start()

                try:
                    while True:
                        item = prefetch.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        shot, kf_indices, frames_bgr = item

                        if shot.index in completed_indices:
                            progress.advance(task)
                            continue

                        progress.update(
                            task,
                            description=f"Analyzing [shot {shot.index + 1}/{len(shots)}]",
                        )

                        frames_list = [
                            frames_bgr[i] for i in kf_indices if i in frames_bgr
                        ]
                        if not frames_list:
                            verdict = ShotVerdict(
                                shot_index=shot.index,
                                action=Action.NONE,
                                category=Category.SAFE,
                                confidence=1.0,
                                reasoning="No frames",
                            )
                            store.save_verdict(job.id, verdict)
                            progress.advance(task)
                            continue

                        with timers.phase("detect_nudity"):
                            batch_dets = detector.detect_batch(frames_list)

                        mid_idx = len(frames_list) // 2
                        mid_frame = frames_list[mid_idx]
                        with timers.phase("detect_clip"):
                            scene_ctx = scene_classifier.classify_shot(mid_frame)

                        start_sec = shot.start_frame / meta.fps
                        end_sec = shot.end_frame / meta.fps
                        if audio_classifier.enabled and context_audio_needed(
                            scene_ctx, config, config.strict
                        ):
                            with timers.phase("detect_audio"):
                                audio_ctx = audio_classifier.classify_segment(
                                    config.input_path, start_sec, end_sec
                                )
                        else:
                            # The audio score cannot change the verdict when the
                            # CLIP scene signal is below its thresholds — skip the
                            # PANNs run entirely and fuse with a neutral context.
                            audio_ctx = AudioContext(
                                moaning_score=0.0,
                                sexual_audio_score=0.0,
                                music_score=0.0,
                                speech_score=0.0,
                            )

                        with timers.phase("fuse"):
                            verdict = fuse(
                                shot,
                                batch_dets,
                                scene_ctx,
                                audio_ctx,
                                config,
                                strict_mode=config.strict,
                            )

                        if config.strict and verdict.category == Category.KISS_LIGHT:
                            verdict.action = Action.BLACK_BOX

                        if verdict.action != Action.NONE:
                            is_kiss_black_box = (
                                verdict.action == Action.BLACK_BOX
                                and verdict.category
                                in (
                                    Category.KISS_INTENSE,
                                    Category.KISS_LIGHT,
                                )
                            )
                            if is_kiss_black_box:
                                # Sample the shot at the profile's densify
                                # stride instead of decoding every frame; the
                                # IoU tracker in smooth_detections bridges the
                                # gaps so the blur stays continuous.
                                stride = max(1, settings.densify_every_n_frames)
                                all_frames = list(
                                    range(shot.start_frame, shot.end_frame, stride)
                                )
                                if all_frames and all_frames[-1] != shot.end_frame - 1:
                                    all_frames.append(shot.end_frame - 1)
                                with timers.phase("extract_kiss"):
                                    all_bgr = extract_frames(
                                        config.input_path,
                                        all_frames,
                                        settings.detection_resolution,
                                        meta=meta,
                                    )

                                dense_faces = {}
                                with timers.phase("detect_faces"):
                                    for f_idx, f_bgr in all_bgr.items():
                                        mouths = face_detector.detect_mouths(f_bgr)
                                        from pureframe.pipeline.detect.nudity import (
                                            Detection,
                                        )

                                        dense_faces[f_idx] = [
                                            Detection(label="MOUTH", score=1.0, box=m)
                                            for m in mouths
                                        ]

                                smooth_mouths = smooth_detections(
                                    dense_faces, shot, config.box_padding_pct
                                )
                                from pureframe.pipeline.shots import Box

                                verdict.boxes = [
                                    Box(
                                        x1=b[0],
                                        y1=b[1],
                                        x2=b[2],
                                        y2=b[3],
                                        frame_idx=f,
                                    )
                                    for f, boxes in smooth_mouths.items()
                                    for b in boxes
                                ]
                            else:
                                # Every other flagged verdict lands here —
                                # nudity BLACK_BOX included. Without this the
                                # verdict carries boxes=None and the renderer
                                # re-encodes the shot without censoring it
                                # (nudity is the primary category, so this
                                # path is the product's main blur source).
                                # FULL_FRAME_BLUR ignores boxes at render
                                # time, but the plan stays reviewable.
                                from pureframe.pipeline.shots import FrameResult

                                for idx, dets in zip(kf_indices, batch_dets):
                                    shot.frames[idx] = FrameResult(
                                        frame_idx=idx, detections=dets
                                    )

                                with timers.phase("densify"):
                                    dense_dets = densify_shot(
                                        shot,
                                        config.input_path,
                                        detector,
                                        settings,
                                        densify_threshold,
                                        meta=meta,
                                    )
                                smooth_boxes = smooth_detections(
                                    dense_dets, shot, config.box_padding_pct
                                )
                                from pureframe.pipeline.shots import Box

                                verdict.boxes = [
                                    Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3], frame_idx=f)
                                    for f, boxes in smooth_boxes.items()
                                    for b in boxes
                                ]

                        store.save_verdict(job.id, verdict)
                        progress.advance(task)
                finally:
                    stop_event.set()
                    # Unblock a worker waiting on a full queue, then drain.
                    while not prefetch.empty():
                        try:
                            prefetch.get_nowait()
                        except Exception:
                            break

        except KeyboardInterrupt:
            store.update_status(job.id, "FAILED", error="Interrupted by user")
            raise
        except Exception as e:
            store.update_status(job.id, "FAILED", error=str(e))
            raise

        if not settings.keep_models_loaded:
            detector.unload()
            scene_classifier.unload()
            audio_classifier.unload()
            del detector
            del scene_classifier
            del audio_classifier
            del face_detector

        store.update_status(job.id, "RENDERING")

        verdicts = store.load_verdicts(job.id)

    # Compute totals
    verdicts.sort(key=lambda x: x.shot_index)
    total_censored = 0
    total_blur = 0
    shot_map = {s.index: s for s in shots}
    for v in verdicts:
        shot = shot_map.get(v.shot_index)
        if shot and v.action != Action.NONE:
            frames = shot.end_frame - shot.start_frame
            total_censored += frames
            if v.action == Action.FULL_FRAME_BLUR:
                total_blur += frames

    try:
        package_version = version("pureframe")
    except PackageNotFoundError:
        package_version = "unknown"

    plan = CensorPlan(
        pureframe_version=package_version,
        plan_version=1,
        input_metadata=meta,
        config_snapshot=config.model_dump(),
        shots=shots,
        verdicts=verdicts,
        total_censored_frames=total_censored,
        total_blur_frames=total_blur,
        generated_at=datetime.now(UTC),
    )
    return plan


def execute_render(
    plan: CensorPlan,
    config: Config,
    smart: bool = True,
    timers: PhaseTimers | None = None,
):
    timers = timers or PhaseTimers()
    store = get_store()
    job = store.find_or_create_job(config.input_path, config.output_path, config)

    frame_actions = plan.build_frame_actions()

    if plan.input_metadata.total_frames > 0:
        console.print(
            f"Render estimate: {format_duration(estimate_render_seconds(config.profile, plan.input_metadata.total_frames, plan.total_censored_frames))} "
            f"({plan.total_censored_frames} censored of {plan.input_metadata.total_frames} frames)"
        )

    try:
        with console.status(
            "[bold green]Rendering final video... (this may take a while)"
        ):
            with timers.phase("render"):
                if smart:
                    from pureframe.pipeline.render.smart import apply_censoring_smart

                    apply_censoring_smart(
                        config.input_path,
                        config.output_path,
                        frame_actions,
                        config,
                        get_settings(config.profile),
                        plan.input_metadata.total_frames,
                        plan.input_metadata.fps,
                        input_codec=plan.input_metadata.video_codec,
                    )
                else:
                    apply_censoring(
                        config.input_path,
                        config.output_path,
                        frame_actions,
                        config,
                        get_settings(config.profile),
                        input_codec=plan.input_metadata.video_codec,
                    )

        # A render that silently produced nothing must never be recorded as
        # DONE — that is exactly how stale checkpoints went on to skip every
        # future attempt ("already DONE. Skipping.") while no output existed.
        out_file = Path(config.output_path) if config.output_path else None
        if out_file is None or not out_file.exists() or out_file.stat().st_size == 0:
            raise PureFrameError(
                f"Render finished but the output file is missing or empty: {out_file}"
            )

        store.update_status(job.id, "DONE")
        console.print(
            f"\n[bold green]Success![/bold green] Output saved to {config.output_path}"
        )
        console.print(f"Total censored frames: {len(frame_actions)}")
    except KeyboardInterrupt:
        store.update_status(job.id, "FAILED", error="Interrupted during rendering")
        raise
    except Exception as e:
        # Catching ``Exception`` (not ``BaseException``) so that SystemExit
        # and KeyboardInterrupt propagate cleanly without being logged as
        # render failures.
        import traceback

        traceback.print_exc()
        store.update_status(job.id, "FAILED", error=str(e))
        raise


def process_file(config: Config):
    store = get_store()
    job = store.find_or_create_job(config.input_path, config.output_path, config)

    skip = job.status == "DONE" and not config.force
    if skip and config.output_path is not None:
        out_abs = Path(config.output_path).absolute()
        # A DONE checkpoint only proves the job recorded back then. It does
        # not apply when the caller now targets a different output path (the
        # store is keyed on input + config hash), or when the previously
        # rendered file has since been deleted or truncated. In those cases
        # we must redo the work instead of silently producing nothing.
        if job.output_path != str(out_abs):
            skip = False
        elif not out_abs.exists() or out_abs.stat().st_size == 0:
            skip = False

    if skip:
        console.print(
            f"[green]Job {job.id} for {config.input_path.name} is already DONE. Skipping.[/green]"
        )
        return

    timers = PhaseTimers()
    plan = generate_plan(config, timers)

    console.print(
        f"Flagged {sum(1 for v in plan.verdicts if v.action != Action.NONE)} shots for censoring."
    )
    execute_render(plan, config, timers=timers)

    if config.log_level == "DEBUG":
        console.print(timers.summary())
    if os.environ.get("PUREFRAME_PRINT_TIMERS") == "1" or os.environ.get(
        "PUREFRAME_TIMERS_FILE"
    ):
        # Machine-readable hook for `pureframe bench`.
        import json

        payload = json.dumps(
            {
                "phases": timers.as_dict(),
                "flagged_shots": sum(
                    1 for v in plan.verdicts if v.action != Action.NONE
                ),
            }
        )
        timers_file = os.environ.get("PUREFRAME_TIMERS_FILE")
        if timers_file:
            Path(timers_file).write_text(payload, encoding="utf-8")
        else:
            print(f"PUREFRAME_TIMERS {payload}", flush=True)


@app.command("plan")
def plan_cmd(
    input: Path = typer.Argument(..., exists=True, help="Path to input video file"),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output censorplan JSON (defaults to input.censorplan.json)",
    ),
    profile: HardwareProfile = typer.Option(
        None, "--profile", help="Hardware profile override"
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Nudity detection threshold (alias for --threshold-nudity)",
    ),
    threshold_nudity: float | None = typer.Option(
        None,
        "--threshold-nudity",
        min=0.0,
        max=1.0,
        help="Nudity threshold; replaces the strictness preset's value",
    ),
    threshold_clip: float | None = typer.Option(
        None,
        "--threshold-clip",
        min=0.0,
        max=1.0,
        help="CLIP scene threshold; replaces the strictness preset's value",
    ),
    threshold_audio: float | None = typer.Option(
        None,
        "--threshold-audio",
        min=0.0,
        max=1.0,
        help="Audio threshold; replaces the strictness preset's value",
    ),
    thresholds_file: Path | None = typer.Option(
        None,
        "--thresholds",
        exists=True,
        dir_okay=False,
        help='JSON file with any of "nudity", "clip", "audio"; flags win over it',
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Lowers thresholds 15% across the board"
    ),
    no_clip: bool = typer.Option(
        False, "--no-clip", help="Disables CLIP scene classifier"
    ),
    no_audio: bool = typer.Option(
        False, "--no-audio", help="Disables audio classifier"
    ),
    content_type: ContentType = typer.Option(
        ContentType.LIVE_ACTION, "--content-type", help="Content type preset"
    ),
    strictness: Strictness = typer.Option(
        Strictness.MEDIUM,
        "--strictness",
        help="Strictness level: low, medium, high, custom",
    ),
    no_quant: bool = typer.Option(
        False,
        "--no-quant",
        help="Disable int8 CPU model quantization (GPU profiles unaffected)",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Ignore cached plans/verdicts for this run and re-analyze from scratch",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Run detection only, save a censor plan JSON. Do not render."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    if profile is None:
        profile = detect_profile()

    if output is None:
        output = input.with_name(f"{input.name}.censorplan.json")

    try:
        overrides = _threshold_overrides(
            thresholds_file,
            threshold_nudity if threshold_nudity is not None else threshold,
            threshold_clip,
            threshold_audio,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--thresholds") from e

    config = _build_config(
        input_path=input,
        output_path=input.with_name(
            f"{input.stem}.pureframe{input.suffix}"
        ),  # Dummy output path for DB tracking
        profile=profile,
        threshold_overrides=overrides,
        strict=strict,
        no_clip=no_clip,
        no_audio=no_audio,
        content_type=content_type,
        strictness=strictness,
        quantize_cpu=not no_quant,
        no_cache=no_cache,
        cache_salt=uuid4().hex if no_cache else "",
        log_level="DEBUG" if verbose else "INFO",
    )

    timers = PhaseTimers()
    plan = generate_plan(config, timers)
    plan.serialize(output)

    if verbose:
        console.print(timers.summary())
    if os.environ.get("PUREFRAME_PRINT_TIMERS") == "1":
        import json

        print(f"PUREFRAME_TIMERS {json.dumps(timers.as_dict())}", flush=True)
    console.print(f"[green]Plan saved to {output}[/green]")


@app.command("apply")
def apply_cmd(
    input: Path = typer.Argument(..., exists=True, help="Path to input video file"),
    plan_path: Path = typer.Argument(..., exists=True, help="Path to censorplan JSON"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Path to output video file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Take a previously generated plan and render the censored video."""
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    plan = CensorPlan.load(plan_path)

    # We reconstruct config from the snapshot
    config_dict = plan.config_snapshot.copy()
    config_dict["input_path"] = input
    if output:
        config_dict["output_path"] = output
    else:
        config_dict["output_path"] = input.with_name(
            f"{input.stem}.pureframe{input.suffix}"
        )

    config = Config(**config_dict)

    console.print("[bold blue]PureFrame Apply[/bold blue]")
    console.print(f"Applying plan: {plan_path.name}")
    try:
        execute_render(plan, config)
    except Exception as e:
        import traceback

        console.print(f"[red]APPLY ERROR:[/red] {e}")
        console.print(traceback.format_exc())
        raise


@app.command("plan-edit")
def plan_edit_cmd(
    plan_path: Path = typer.Argument(
        ..., exists=True, help="Path to censorplan JSON to edit"
    ),
):
    """Open plan in $EDITOR and validate after closing."""
    editor = os.environ.get("EDITOR", "nano")
    import shlex

    while True:
        # On Windows we must keep backslashes literal — shlex POSIX mode would
        # treat them as escape characters and mangle paths like
        # ``C:\Python311\python.exe``.
        editor_cmd = shlex.split(editor, posix=(os.name != "nt"))
        editor_cmd.append(str(plan_path))
        subprocess.call(editor_cmd)

        try:
            CensorPlan.load(plan_path)
            console.print("[green]Plan valid and saved.[/green]")
            break
        except Exception as e:
            console.print(f"[red]Invalid plan JSON:[/red] {e}")
            retry = typer.confirm("Re-edit to fix?")
            if not retry:
                console.print(
                    "[yellow]Exiting without fixing plan. It might be broken.[/yellow]"
                )
                break


@app.command("plan-whitelist")
def plan_whitelist_cmd(
    plan_path: Path = typer.Argument(..., exists=True, help="Path to censorplan JSON"),
    shot_index: int = typer.Argument(..., help="Shot index to whitelist"),
):
    """Set a shot's action to NONE in the plan."""
    plan = CensorPlan.load(plan_path)
    found = False
    for v in plan.verdicts:
        if v.shot_index == shot_index:
            v.action = Action.NONE
            found = True
            break

    if found:
        plan.serialize(plan_path)
        console.print(f"[green]Shot {shot_index} whitelisted successfully.[/green]")
    else:
        console.print(f"[red]Shot {shot_index} not found in plan verdicts.[/red]")


@app.command("process")
def process_cmd(
    input: Path = typer.Argument(
        ..., exists=True, help="Path to input video file or folder"
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output video file (ignored if input is folder)",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process folder recursively"
    ),
    parallel: int = typer.Option(
        1, "--parallel", "-p", help="Number of parallel workers for folders"
    ),
    profile: HardwareProfile = typer.Option(
        None, "--profile", help="Hardware profile override"
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Nudity detection threshold (alias for --threshold-nudity)",
    ),
    threshold_nudity: float | None = typer.Option(
        None,
        "--threshold-nudity",
        min=0.0,
        max=1.0,
        help="Nudity threshold; replaces the strictness preset's value",
    ),
    threshold_clip: float | None = typer.Option(
        None,
        "--threshold-clip",
        min=0.0,
        max=1.0,
        help="CLIP scene threshold; replaces the strictness preset's value",
    ),
    threshold_audio: float | None = typer.Option(
        None,
        "--threshold-audio",
        min=0.0,
        max=1.0,
        help="Audio threshold; replaces the strictness preset's value",
    ),
    thresholds_file: Path | None = typer.Option(
        None,
        "--thresholds",
        exists=True,
        dir_okay=False,
        help='JSON file with any of "nudity", "clip", "audio"; flags win over it',
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Lowers thresholds 15% across the board"
    ),
    no_clip: bool = typer.Option(
        False, "--no-clip", help="Disables CLIP scene classifier"
    ),
    no_audio: bool = typer.Option(
        False, "--no-audio", help="Disables audio classifier"
    ),
    content_type: ContentType = typer.Option(
        ContentType.LIVE_ACTION, "--content-type", help="Content type preset"
    ),
    strictness: Strictness = typer.Option(
        Strictness.MEDIUM, "--strictness", help="Strictness level"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force reprocess even if job already done"
    ),
    no_quant: bool = typer.Option(
        False,
        "--no-quant",
        help="Disable int8 CPU model quantization (GPU profiles unaffected)",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Ignore cached plans/verdicts for this run and re-analyze from scratch",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Process a video file or folder to censor explicit content.
    """
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    if profile is None:
        profile = detect_profile()

    try:
        overrides = _threshold_overrides(
            thresholds_file,
            threshold_nudity if threshold_nudity is not None else threshold,
            threshold_clip,
            threshold_audio,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--thresholds") from e

    cache_salt = uuid4().hex if no_cache else ""

    if input.is_dir():
        from pureframe.batch import process_folder

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            dummy_file = Path(f.name)
            base_config = _build_config(
                input_path=dummy_file,
                profile=profile,
                threshold_overrides=overrides,
                strict=strict,
                no_clip=no_clip,
                no_audio=no_audio,
                content_type=content_type,
                strictness=strictness,
                force=force,
                quantize_cpu=not no_quant,
                no_cache=no_cache,
                cache_salt=cache_salt,
                log_level="DEBUG" if verbose else "INFO",
            )
        process_folder(input, recursive, parallel, base_config)
    else:
        config = _build_config(
            input_path=input,
            output_path=output,
            profile=profile,
            threshold_overrides=overrides,
            strict=strict,
            no_clip=no_clip,
            no_audio=no_audio,
            content_type=content_type,
            strictness=strictness,
            force=force,
            quantize_cpu=not no_quant,
            no_cache=no_cache,
            cache_salt=cache_salt,
            log_level="DEBUG" if verbose else "INFO",
        )
        process_file(config)


@jobs_app.command("list")
def jobs_list():
    """List all unfinished jobs."""
    store = get_store()
    unfinished = store.list_unfinished()
    if not unfinished:
        console.print("No unfinished jobs.")
        return

    table = Table(title="Unfinished Jobs")
    table.add_column("ID")
    table.add_column("Input")
    table.add_column("Status")
    table.add_column("Completed / Total")
    table.add_column("Started At")

    for j in unfinished:
        p = Path(j.input_path).name
        table.add_row(
            str(j.id),
            p,
            j.status,
            f"{j.completed_shots} / {j.total_shots or '?'}",
            str(j.started_at),
        )
    console.print(table)


@jobs_app.command("resume")
def jobs_resume(job_id: int = typer.Argument(..., help="Job ID to resume")):
    """Resume a specific job by ID."""
    store = get_store()
    unfinished = store.list_unfinished()
    job = next((j for j in unfinished if j.id == job_id), None)

    if not job:
        console.print(f"[red]Job {job_id} not found or already DONE.[/red]")
        return

    if not job.config_json:
        console.print(
            f"[red]Job {job_id} does not have a saved configuration to resume from.[/red]"
        )
        return

    console.print(f"Resuming job {job_id} on {job.input_path}...")
    cfg = Config.model_validate_json(job.config_json)
    process_file(cfg)


@jobs_app.command("cleanup")
def jobs_cleanup(
    all_jobs: bool = typer.Option(
        False, "--all", help="Remove all jobs including pending"
    ),
    failed: bool = typer.Option(False, "--failed", help="Remove only failed jobs"),
):
    """Delete completed or failed job records."""
    store = get_store()
    with store.conn:
        cursor = store.conn.cursor()
        if all_jobs:
            cursor.execute("DELETE FROM shot_verdicts")
            cursor.execute("DELETE FROM jobs")
        elif failed:
            cursor.execute(
                "DELETE FROM shot_verdicts WHERE job_id IN (SELECT id FROM jobs WHERE status = 'FAILED')"
            )
            cursor.execute("DELETE FROM jobs WHERE status = 'FAILED'")
        else:
            cursor.execute(
                "DELETE FROM shot_verdicts WHERE job_id IN (SELECT id FROM jobs WHERE status = 'DONE' AND finished_at < datetime('now', '-30 days'))"
            )
            cursor.execute(
                "DELETE FROM jobs WHERE status = 'DONE' AND finished_at < datetime('now', '-30 days')"
            )
        deleted = cursor.rowcount
    console.print(f"Cleaned up {deleted} job records.")


@app.command("preview")
def preview_cmd(
    plan_path: Path = typer.Argument(..., exists=True, help="Path to censorplan JSON"),
    output: Path = typer.Option(None, "--output", "-o", help="Output HTML report path"),
    blur: bool = typer.Option(
        True, "--blur/--no-blur", help="Apply blur to flagged regions in thumbnails"
    ),
):
    """Export flagged frame thumbnails as an HTML contact sheet for safe review."""
    plan = CensorPlan.load(plan_path)

    if output is None:
        output = plan_path.with_suffix(".preview.html")

    flagged = [v for v in plan.verdicts if v.action != Action.NONE]

    if not flagged:
        console.print(
            "[green]No flagged shots in this plan. Nothing to preview.[/green]"
        )
        return

    # Infer video path from plan
    video_path_str = plan.config_snapshot.get("input_path", "")
    video_path = Path(video_path_str)
    video_name = video_path.name if video_path_str else "unknown"

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>PureFrame Preview Report</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,sans-serif;background:#0f0f0f;color:#e0e0e0;padding:2rem;}",
        "h1{color:#60a5fa;} h2{color:#f59e0b;border-bottom:1px solid #333;padding-bottom:0.5rem;}",
        ".shot{background:#1a1a2e;border-radius:8px;padding:1rem;margin:1rem 0;}",
        ".meta{color:#888;font-size:0.85rem;}",
        ".action-box{display:inline-block;padding:4px 12px;border-radius:4px;font-weight:bold;}",
        ".BLACK_BOX{background:#dc2626;color:#fff;}",
        ".FULL_FRAME_BLUR{background:#f59e0b;color:#000;}",
        "</style></head><body>",
        "<h1>PureFrame Preview Report</h1>",
        f"<p class='meta'>Plan: {plan_path.name} | Video: {video_name} | "
        f"Flagged: {len(flagged)}/{len(plan.verdicts)} shots | "
        f"Generated: {plan.generated_at}</p>",
    ]

    for v in flagged:
        shot = next((s for s in plan.shots if s.index == v.shot_index), None)
        if not shot:
            continue

        time_str = f"{shot.start_time:.1f}s – {shot.end_time:.1f}s"
        html_parts.append("<div class='shot'>")
        html_parts.append(f"<h2>Shot #{v.shot_index}</h2>")
        html_parts.append(
            f"<p>Time: {time_str} | Frames: {shot.start_frame}–{shot.end_frame}</p>"
        )
        html_parts.append(
            f"<p>Category: <strong>{v.category}</strong> | Confidence: {v.confidence:.1%}</p>"
        )
        html_parts.append(
            f"<p>Action: <span class='action-box {v.action}'>{v.action}</span></p>"
        )
        html_parts.append(f"<p class='meta'>Reasoning: {v.reasoning}</p>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    output.write_text("\n".join(html_parts), encoding="utf-8")
    console.print(f"[green]Preview report saved to {output}[/green]")
    console.print(
        f"Flagged {len(flagged)} shots across {plan.input_metadata.duration_seconds:.0f}s of video."
    )


@app.command()
def evaluate(
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Path to save evaluation report JSON"
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Detection confidence threshold"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the PureFrame evaluation benchmark.

    Tests detection accuracy against 50 synthetic scenarios across 8 content genres.
    Computes precision, recall, F1, and false positive rate with per-genre breakdown.
    """
    setup_logging(log_level="DEBUG" if verbose else "INFO")

    from pureframe.eval import run_synthetic_benchmark

    console.print("[bold]PureFrame Evaluation Benchmark[/bold]")
    console.print(f"Running 50 synthetic scenarios at threshold={threshold}...")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Running benchmark...", total=None)
        report = run_synthetic_benchmark(threshold=threshold)

    # Display results
    table = Table(title="Aggregate Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Precision", f"{report.precision:.1%}")
    table.add_row("Recall", f"{report.recall:.1%}")
    table.add_row("F1 Score", f"{report.f1_score:.1%}")
    table.add_row("False Positive Rate", f"{report.false_positive_rate:.1%}")
    table.add_row("Accuracy", f"{report.accuracy:.1%}")
    table.add_row("Total Scenes", str(report.total_scenes))
    console.print(table)
    console.print()

    # Per-genre table
    genre_table = Table(title="Per-Genre Breakdown")
    genre_table.add_column("Genre", style="cyan")
    genre_table.add_column("Total", style="white")
    genre_table.add_column("Precision", style="green")
    genre_table.add_column("Recall", style="green")
    genre_table.add_column("F1", style="green")
    for genre, metrics in sorted(report.genre_metrics.items()):
        genre_table.add_row(
            genre,
            str(metrics["total"]),
            f"{metrics['precision']:.1%}",
            f"{metrics['recall']:.1%}",
            f"{metrics['f1']:.1%}",
        )
    console.print(genre_table)
    console.print()

    # Threshold sweep
    thresh_table = Table(title="Threshold Sensitivity Analysis")
    thresh_table.add_column("Threshold", style="cyan")
    thresh_table.add_column("Precision", style="green")
    thresh_table.add_column("Recall", style="green")
    thresh_table.add_column("F1", style="green")
    thresh_table.add_column("FPR", style="red")
    for t in report.threshold_analysis:
        thresh_table.add_row(
            str(t["threshold"]),
            f"{t['precision']:.1%}",
            f"{t['recall']:.1%}",
            f"{t['f1']:.1%}",
            f"{t['fpr']:.1%}",
        )
    console.print(thresh_table)

    if output:
        report.save(output)
        console.print(f"\n[green]Report saved to {output}[/green]")
    else:
        default_path = Path("evaluation_report.json")
        report.save(default_path)
        console.print(f"\n[green]Report saved to {default_path}[/green]")


@app.command("bench")
def bench_cmd(
    duration: float = typer.Option(
        30.0, "--duration", help="Synthetic clip length in seconds"
    ),
    width: int = typer.Option(1280, "--width", help="Clip width"),
    height: int = typer.Option(720, "--height", help="Clip height"),
    profiles: str = typer.Option(
        "CPU,LOW,MEDIUM,HIGH", "--profiles", help="Comma-separated hardware profiles"
    ),
    reps: int = typer.Option(
        1, "--reps", min=1, help="Runs per profile (median reported)"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Write the JSON report to this path"
    ),
    keep_clip: bool = typer.Option(
        False,
        "--keep-clip",
        help="Save the generated clip next to the report for reuse",
    ),
):
    """Repeatable performance benchmark across hardware profiles.

    Generates a synthetic clip with moving skin-tone regions (so detection,
    densify and blur actually run), then times the full `process` flow per
    profile with per-phase breakdowns. Checkpoint state is isolated; model
    caches are shared between runs.
    """
    from pureframe.bench import BENCH_PROFILES, report_to_markdown, run_benchmark

    profile_list = [p.strip().upper() for p in profiles.split(",") if p.strip()]
    invalid = [p for p in profile_list if p not in BENCH_PROFILES]
    if invalid:
        console.print(
            f"[red]Unknown profile(s): {', '.join(invalid)}. "
            f"Valid: {', '.join(BENCH_PROFILES)}[/red]"
        )
        raise typer.Exit(1)

    keep_path = None
    if keep_clip:
        # Reused across invocations; run_benchmark generates it if missing.
        keep_path = Path.cwd() / f"pureframe_bench_clip_{width}x{height}.mp4"

    report = run_benchmark(
        profiles=profile_list,
        reps=reps,
        duration=duration,
        width=width,
        height=height,
        keep_clip=keep_path,
    )

    console.print()
    console.print(report_to_markdown(report))

    if output:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"\n[green]JSON report saved to {output}[/green]")


if __name__ == "__main__":
    app()
