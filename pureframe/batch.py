import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rich
from pydantic import BaseModel
from rich.live import Live
from rich.table import Table

from pureframe.config import Config


def _batch_worker(cfg: Config):
    """Module-level worker so ProcessPoolExecutor can pickle it."""
    from pureframe.cli import process_file

    try:
        process_file(cfg)
        return cfg.input_path.name, "DONE", None
    except Exception as e:
        return cfg.input_path.name, "FAILED", str(e)


class BatchReport(BaseModel):
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def process_folder(
    folder: Path, recursive: bool, parallel: int, base_config: Config
) -> BatchReport:
    """
    Process a folder of video files using a ProcessPoolExecutor.
    """
    from pureframe.cli import get_store

    store = get_store()

    extensions = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v", ".ts", ".wmv"}

    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file()]
    else:
        files = [p for p in folder.glob("*") if p.is_file()]

    videos = []
    for f in files:
        if f.suffix.lower() in extensions and ".pureframe" not in f.suffixes:
            videos.append(f)

    report = BatchReport()

    # Resolve profile once here — don't pass None into subprocesses
    from pureframe.hardware import detect_profile

    resolved_profile = base_config.profile or detect_profile()

    jobs_to_run = []

    for v in videos:
        out_path = v.with_name(f"{v.stem}.pureframe{v.suffix}")

        # Clone config for this file via ``model_copy`` so we inherit every
        # field (including content_type, strictness, blur_mode, etc.). The
        # previous implementation listed fields by hand and silently dropped
        # ``content_type`` and ``strictness``, causing batch runs to ignore
        # those settings entirely. The content fingerprint is recomputed —
        # the base config carries the dummy placeholder file's hash, and a
        # stale one would key every re-run of this file to a fresh job.
        from pureframe.checkpoint import content_fingerprint

        cfg = base_config.model_copy(
            update={
                "input_path": v,
                "output_path": out_path,
                "profile": resolved_profile,
                "content_fingerprint": content_fingerprint(v),
            },
            deep=True,
        )

        job = store.find_or_create_job(v, out_path, cfg)
        if job.status == "DONE":
            report.skipped += 1
            continue

        jobs_to_run.append(cfg)

    if not jobs_to_run:
        rich.print(
            "[bold yellow]No files to process or all are already DONE.[/bold yellow]"
        )
        return report

    table = Table(title="Batch Processing Progress")
    table.add_column("File")
    table.add_column("Status")

    status_map = {cfg.input_path.name: "PENDING" for cfg in jobs_to_run}

    def render_table():
        t = Table(title="Batch Processing Progress")
        t.add_column("File")
        t.add_column("Status")
        for k, v in status_map.items():
            t.add_row(k, v)
        return t

    # Use 'spawn' start method — CUDA cannot be re-initialized in forked subprocesses
    mp_ctx = multiprocessing.get_context("spawn")
    with Live(render_table(), refresh_per_second=2) as live:
        with ProcessPoolExecutor(max_workers=parallel, mp_context=mp_ctx) as executor:
            futures = {executor.submit(_batch_worker, cfg): cfg for cfg in jobs_to_run}
            for k in status_map:
                status_map[k] = "PROCESSING"
            live.update(render_table())

            for future in as_completed(futures):
                name, st, err = future.result()
                status_map[name] = st if not err else f"FAILED: {err[:60]}"
                if st == "DONE":
                    report.processed += 1
                else:
                    report.failed += 1
                live.update(render_table())

    rich.print(
        f"Batch completed: [green]{report.processed} processed[/green], [yellow]{report.skipped} skipped[/yellow], [red]{report.failed} failed[/red]."
    )
    return report
