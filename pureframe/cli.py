import typer
from importlib.metadata import PackageNotFoundError, version
from typing import Optional
import platformdirs
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from pureframe.config import Config
from pureframe.hardware import HardwareProfile, detect_profile, get_settings
from pureframe.utils.logging import setup_logging
from pureframe.pipeline.probe import probe_video
from pureframe.pipeline.shots import detect_shots, Action, Category, ShotVerdict
from pureframe.pipeline.sample import sample_keyframes, extract_frames
from pureframe.pipeline.detect.nudity import NudityDetector
from pureframe.pipeline.densify import densify_shot
from pureframe.pipeline.smooth import smooth_detections
from pureframe.pipeline.render.apply import apply_censoring
from pureframe.pipeline.detect.scene_clip import SceneClassifier
from pureframe.pipeline.detect.audio import AudioClassifier
from pureframe.pipeline.detect.face import FaceDetector
from pureframe.pipeline.fuse import fuse
from pureframe.checkpoint import CheckpointStore
from pureframe.pipeline.render.plan import CensorPlan

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
    version_option: Optional[bool] = typer.Option(
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
    db_path = Path(platformdirs.user_data_dir("PureFrame")) / "jobs.db"
    return CheckpointStore(db_path)


def generate_plan(config: Config) -> CensorPlan:
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
        shots = detect_shots(config.input_path)
        meta = probe_video(config.input_path)
    else:
        store.update_status(job.id, "DETECTING")

        with console.status("[bold green]Probing video..."):
            meta = probe_video(config.input_path)

        with console.status("[bold green]Detecting shots..."):
            shots = detect_shots(config.input_path)

        store.update_status(job.id, "DETECTING", total_shots=len(shots))
        console.print(f"Detected {len(shots)} shots.")

        detector = NudityDetector(settings)
        scene_classifier = SceneClassifier(settings)
        if config.no_clip:
            scene_classifier.enabled = False

        audio_classifier = AudioClassifier(settings)
        if config.no_audio or len(meta.audio_streams) == 0:
            audio_classifier.enabled = False

        face_detector = FaceDetector()

        existing_verdicts = store.load_verdicts(job.id)
        completed_indices = {v.shot_index for v in existing_verdicts}

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

                for shot in shots:
                    if shot.index in completed_indices:
                        progress.advance(task)
                        continue

                    progress.update(
                        task,
                        description=f"Analyzing [shot {shot.index + 1}/{len(shots)}]",
                    )
                    kf_indices = sample_keyframes(
                        shot, settings.sample_keyframes_per_shot
                    )
                    frames_bgr = extract_frames(
                        config.input_path, kf_indices, settings.detection_resolution
                    )

                    frames_list = [frames_bgr[i] for i in kf_indices if i in frames_bgr]
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

                    batch_dets = detector.detect_batch(frames_list)

                    mid_idx = len(frames_list) // 2
                    mid_frame = frames_list[mid_idx]
                    scene_ctx = scene_classifier.classify_shot(mid_frame)

                    start_sec = shot.start_frame / meta.fps
                    end_sec = shot.end_frame / meta.fps
                    audio_ctx = audio_classifier.classify_segment(
                        config.input_path, start_sec, end_sec
                    )

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
                        if verdict.action == Action.BLACK_BOX:
                            if verdict.category in (
                                Category.KISS_INTENSE,
                                Category.KISS_LIGHT,
                            ):
                                all_frames = list(
                                    range(shot.start_frame, shot.end_frame)
                                )
                                all_bgr = extract_frames(
                                    config.input_path,
                                    all_frames,
                                    settings.detection_resolution,
                                )

                                dense_faces = {}
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
                                    Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3], frame_idx=f)
                                    for f, boxes in smooth_mouths.items()
                                    for b in boxes
                                ]
                            else:
                                from pureframe.pipeline.shots import FrameResult

                                for idx, dets in zip(kf_indices, batch_dets):
                                    shot.frames[idx] = FrameResult(
                                        frame_idx=idx, detections=dets
                                    )

                                dense_dets = densify_shot(
                                    shot,
                                    config.input_path,
                                    detector,
                                    settings,
                                    config.nudity_threshold,
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
        generated_at=datetime.now(timezone.utc),
    )
    return plan


def execute_render(plan: CensorPlan, config: Config):
    store = get_store()
    job = store.find_or_create_job(config.input_path, config.output_path, config)

    frame_actions = plan.build_frame_actions()

    try:
        with console.status(
            "[bold green]Rendering final video... (this may take a while)"
        ):
            apply_censoring(
                config.input_path,
                config.output_path,
                frame_actions,
                config,
                get_settings(config.profile),
            )

        store.update_status(job.id, "DONE")
        console.print(
            f"\n[bold green]Success![/bold green] Output saved to {config.output_path}"
        )
        console.print(f"Total censored frames: {len(frame_actions)}")
    except KeyboardInterrupt:
        store.update_status(job.id, "FAILED", error="Interrupted during rendering")
        raise
    except BaseException as e:
        import traceback

        traceback.print_exc()
        store.update_status(job.id, "FAILED", error=str(e))
        raise


def process_file(config: Config):
    store = get_store()
    job = store.find_or_create_job(config.input_path, config.output_path, config)
    if job.status == "DONE":
        console.print(
            f"[green]Job {job.id} for {config.input_path.name} is already DONE. Skipping.[/green]"
        )
        return

    plan = generate_plan(config)

    console.print(
        f"Flagged {sum(1 for v in plan.verdicts if v.action != Action.NONE)} shots for censoring."
    )
    execute_render(plan, config)


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
    threshold: float = typer.Option(
        0.55, "--threshold", help="Nudity detection threshold"
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

    config = Config.from_cli(
        input_path=input,
        output_path=input.with_name(
            f"{input.stem}.pureframe{input.suffix}"
        ),  # Dummy output path for DB tracking
        profile=profile,
        nudity_threshold=threshold,
        strict=strict,
        no_clip=no_clip,
        no_audio=no_audio,
        log_level="DEBUG" if verbose else "INFO",
    )

    plan = generate_plan(config)
    plan.serialize(output)
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
        editor_cmd = shlex.split(editor)
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
    threshold: float = typer.Option(
        0.55, "--threshold", help="Nudity detection threshold"
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

    if input.is_dir():
        from pureframe.batch import process_folder

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            dummy_file = Path(f.name)
            base_config = Config.from_cli(
                input_path=dummy_file,
                profile=profile,
                nudity_threshold=threshold,
                strict=strict,
                no_clip=no_clip,
                no_audio=no_audio,
                log_level="DEBUG" if verbose else "INFO",
            )
        process_folder(input, recursive, parallel, base_config)
    else:
        config = Config.from_cli(
            input_path=input,
            output_path=output,
            profile=profile,
            nudity_threshold=threshold,
            strict=strict,
            no_clip=no_clip,
            no_audio=no_audio,
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
def jobs_cleanup():
    """Delete jobs marked DONE more than 30 days ago."""
    store = get_store()
    with store.conn:
        cursor = store.conn.cursor()
        cursor.execute(
            "DELETE FROM shot_verdicts WHERE job_id IN (SELECT id FROM jobs WHERE status = 'DONE' AND finished_at < datetime('now', '-30 days'))"
        )
        cursor.execute(
            "DELETE FROM jobs WHERE status = 'DONE' AND finished_at < datetime('now', '-30 days')"
        )
        deleted = cursor.rowcount
    console.print(f"Cleaned up {deleted} old jobs.")


if __name__ == "__main__":
    app()
