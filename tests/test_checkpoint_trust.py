"""Regression tests for checkpoint trust semantics.

Root cause being guarded against: jobs were recorded DONE even when rendering
failed or produced nothing, after which every retry printed
"already DONE. Skipping." and no output ever appeared.
"""

from pathlib import Path

import pytest

import pureframe.cli as cli
from pureframe.config import Config


@pytest.fixture
def store(tmp_path):
    """cli.get_store is monkeypatched by conftest's autouse fixture to a fresh
    tmp DB per test - always go through it so we seed the same store that
    process_file consults."""
    return cli.get_store()


def _make_config(input_path: Path, output_path: Path) -> Config:
    input_path.write_bytes(b"fake")
    return Config.from_cli(input_path=input_path, output_path=output_path)


def test_done_job_with_existing_output_skips(store, tmp_path, monkeypatch):
    src = tmp_path / "movie.mp4"
    out = tmp_path / "out.mp4"
    cfg = _make_config(src, out)

    job = store.find_or_create_job(cfg.input_path, cfg.output_path, cfg)
    out.write_bytes(b"real video bytes")
    store.update_status(job.id, "DONE")

    calls = []
    monkeypatch.setattr(
        cli, "generate_plan", lambda c, timers=None: calls.append("plan")
    )
    monkeypatch.setattr(
        cli,
        "execute_render",
        lambda p, c, smart=True, timers=None: calls.append("render"),
    )

    cli.process_file(cfg)
    assert calls == [], "a DONE job with valid output must not reprocess"


def test_done_job_with_deleted_output_reruns(store, tmp_path, monkeypatch):
    src = tmp_path / "movie.mp4"
    out = tmp_path / "out.mp4"
    cfg = _make_config(src, out)

    job = store.find_or_create_job(cfg.input_path, cfg.output_path, cfg)
    store.update_status(job.id, "DONE")
    # Output was never written (or has been deleted) - the checkpoint lies.

    rendered = []
    monkeypatch.setattr(
        cli,
        "generate_plan",
        lambda c, timers=None: type("PlanStub", (), {"verdicts": []})(),
    )
    monkeypatch.setattr(
        cli,
        "execute_render",
        lambda p, c, smart=True, timers=None: rendered.append(c.output_path),
    )

    cli.process_file(cfg)
    assert rendered == [cfg.output_path], (
        "DONE with missing output must reprocess, never skip"
    )


def test_done_job_with_empty_output_reruns(store, tmp_path, monkeypatch):
    src = tmp_path / "movie.mp4"
    out = tmp_path / "out.mp4"
    cfg = _make_config(src, out)

    job = store.find_or_create_job(cfg.input_path, cfg.output_path, cfg)
    store.update_status(job.id, "DONE")
    out.write_bytes(b"")  # truncated/corrupt render artifact

    rendered = []
    monkeypatch.setattr(
        cli,
        "generate_plan",
        lambda c, timers=None: type("PlanStub", (), {"verdicts": []})(),
    )
    monkeypatch.setattr(
        cli,
        "execute_render",
        lambda p, c, smart=True, timers=None: rendered.append(c.output_path),
    )

    cli.process_file(cfg)
    assert rendered == [cfg.output_path]


def test_done_job_with_different_output_target_reruns(store, tmp_path, monkeypatch):
    """The store keys on input + config hash; a new --output target must not
    inherit the old job's DONE status."""
    src = tmp_path / "movie.mp4"
    out_a = tmp_path / "a.mp4"
    out_b = tmp_path / "b.mp4"
    cfg_a = _make_config(src, out_a)
    cfg_b = _make_config(src, out_b)

    job = store.find_or_create_job(cfg_a.input_path, cfg_a.output_path, cfg_a)
    out_a.write_bytes(b"video")
    store.update_status(job.id, "DONE")

    rendered = []
    monkeypatch.setattr(
        cli,
        "generate_plan",
        lambda c, timers=None: type("PlanStub", (), {"verdicts": []})(),
    )
    monkeypatch.setattr(
        cli,
        "execute_render",
        lambda p, c, smart=True, timers=None: rendered.append(c.output_path),
    )

    cli.process_file(cfg_b)
    assert rendered == [cfg_b.output_path], (
        "same input but different requested output must render to it"
    )


def test_force_flag_reprocesses_done_job(store, tmp_path, monkeypatch):
    src = tmp_path / "movie.mp4"
    out = tmp_path / "out.mp4"
    cfg = _make_config(src, out)

    job = store.find_or_create_job(cfg.input_path, cfg.output_path, cfg)
    out.write_bytes(b"video")
    store.update_status(job.id, "DONE")

    rendered = []
    monkeypatch.setattr(
        cli,
        "generate_plan",
        lambda c, timers=None: type("PlanStub", (), {"verdicts": []})(),
    )
    monkeypatch.setattr(
        cli,
        "execute_render",
        lambda p, c, smart=True, timers=None: rendered.append(c.output_path),
    )

    forced = cfg.model_copy(update={"force": True})
    cli.process_file(forced)
    assert rendered == [out], "--force must bypass the DONE skip"


def test_failed_job_is_never_skipped(store, tmp_path, monkeypatch):
    src = tmp_path / "movie.mp4"
    out = tmp_path / "out.mp4"
    cfg = _make_config(src, out)

    job = store.find_or_create_job(cfg.input_path, cfg.output_path, cfg)
    store.update_status(job.id, "FAILED", error="boom")

    rendered = []
    monkeypatch.setattr(
        cli,
        "generate_plan",
        lambda c, timers=None: type("PlanStub", (), {"verdicts": []})(),
    )
    monkeypatch.setattr(
        cli,
        "execute_render",
        lambda p, c, smart=True, timers=None: rendered.append(c.output_path),
    )

    cli.process_file(cfg)
    assert rendered == [cfg.output_path]
