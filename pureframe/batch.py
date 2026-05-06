import platformdirs
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import rich
from rich.live import Live
from rich.table import Table
from typing import Optional
from pydantic import BaseModel

from pureframe.config import Config
from pureframe.hardware import get_settings

class BatchReport(BaseModel):
    processed: int = 0
    skipped: int = 0
    failed: int = 0

def process_folder(
    folder: Path, 
    recursive: bool, 
    parallel: int, 
    base_config: Config
) -> BatchReport:
    """
    Process a folder of video files using a ProcessPoolExecutor.
    """
    from pureframe.cli import process_file, get_store
    
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
    
    jobs_to_run = []
    
    for v in videos:
        out_path = v.with_name(f"{v.stem}.pureframe{v.suffix}")
        
        # Clone config for this file
        cfg = Config(
            input_path=v,
            output_path=out_path,
            profile=base_config.profile,
            nudity_threshold=base_config.nudity_threshold,
            box_padding_pct=base_config.box_padding_pct,
            box_color=base_config.box_color,
            output_codec=base_config.output_codec,
            output_crf=base_config.output_crf,
            log_level=base_config.log_level,
            strict=base_config.strict,
            no_clip=base_config.no_clip,
            no_audio=base_config.no_audio
        )
        
        job = store.find_or_create_job(v, out_path, cfg)
        if job.status == "DONE":
            report.skipped += 1
            continue
            
        jobs_to_run.append(cfg)
        
    if not jobs_to_run:
        rich.print("[bold yellow]No files to process or all are already DONE.[/bold yellow]")
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

    # Helper function for the worker pool
    def worker(cfg: Config):
        try:
            process_file(cfg)
            return cfg.input_path.name, "DONE", None
        except Exception as e:
            return cfg.input_path.name, "FAILED", str(e)

    with Live(render_table(), refresh_per_second=2) as live:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(worker, cfg): cfg for cfg in jobs_to_run}
            for k in status_map:
                status_map[k] = "PROCESSING"
            live.update(render_table())
            
            for future in as_completed(futures):
                name, st, err = future.result()
                status_map[name] = st
                if st == "DONE":
                    report.processed += 1
                else:
                    report.failed += 1
                live.update(render_table())
                
    rich.print(f"Batch completed: [green]{report.processed} processed[/green], [yellow]{report.skipped} skipped[/yellow], [red]{report.failed} failed[/red].")
    return report
