"""Tests for the preview command and CLI new features."""

import json
from pathlib import Path

from typer.testing import CliRunner

from pureframe.cli import app
from pureframe.pipeline.render.plan import CensorPlan
from pureframe.pipeline.shots import Action

runner = CliRunner()


def test_process_with_content_type(tmp_path: Path, synthetic_video: Path):
    """Test that --content-type flag is accepted."""
    plan_json = tmp_path / "plan_ct.json"

    result = runner.invoke(
        app,
        [
            "plan",
            str(synthetic_video),
            "--output",
            str(plan_json),
            "--no-clip",
            "--no-audio",
            "--content-type",
            "animation",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert plan_json.exists()

    plan = CensorPlan.load(plan_json)
    assert plan.config_snapshot.get("content_type") == "animation"


def test_process_with_strictness(tmp_path: Path, synthetic_video: Path):
    """Test that --strictness flag is accepted."""
    plan_json = tmp_path / "plan_strict.json"

    result = runner.invoke(
        app,
        [
            "plan",
            str(synthetic_video),
            "--output",
            str(plan_json),
            "--no-clip",
            "--no-audio",
            "--strictness",
            "high",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert plan_json.exists()

    plan = CensorPlan.load(plan_json)
    assert plan.config_snapshot.get("strictness") == "high"


def test_preview_no_flagged_shots(tmp_path: Path, synthetic_video: Path):
    """Preview on a plan with no flagged shots should report nothing to preview."""
    plan_json = tmp_path / "plan_preview.json"

    # Generate plan
    runner.invoke(
        app,
        [
            "plan",
            str(synthetic_video),
            "--output",
            str(plan_json),
            "--no-clip",
            "--no-audio",
            "--strictness",
            "low",  # Low strictness = fewer detections
        ],
        catch_exceptions=False,
    )

    assert plan_json.exists()

    # Whitelist all shots
    plan = CensorPlan.load(plan_json)
    for v in plan.verdicts:
        v.action = Action.NONE
    plan.serialize(plan_json)

    # Preview should report no flagged shots
    preview_html = tmp_path / "preview.html"
    result = runner.invoke(
        app,
        ["preview", str(plan_json), "--output", str(preview_html)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "No flagged shots" in result.stdout


def test_preview_with_flagged_shots(tmp_path: Path, synthetic_video: Path):
    """Preview on a plan with flagged shots should generate HTML report."""
    plan_json = tmp_path / "plan_preview2.json"

    # Generate plan
    runner.invoke(
        app,
        [
            "plan",
            str(synthetic_video),
            "--output",
            str(plan_json),
            "--no-clip",
            "--no-audio",
        ],
        catch_exceptions=False,
    )

    assert plan_json.exists()

    # Infer video path from plan config_snapshot
    plan = CensorPlan.load(plan_json)
    from pureframe.pipeline.shots import Category

    plan.verdicts[0].action = Action.BLACK_BOX
    plan.verdicts[0].category = Category.NUDITY_EXPLICIT
    plan.verdicts[0].confidence = 0.95
    plan.verdicts[0].reasoning = "Test flagged"
    plan.serialize(plan_json)

    # Preview
    preview_html = tmp_path / "preview.html"
    result = runner.invoke(
        app,
        ["preview", str(plan_json), "--output", str(preview_html)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert preview_html.exists()
    assert "Flagged 1 shots" in result.stdout

    # Verify HTML content
    html = preview_html.read_text()
    assert "PureFrame Preview Report" in html
    assert "Shot #0" in html
    assert "BLACK_BOX" in html


def test_jobs_cleanup_all(tmp_path: Path):
    """Test jobs cleanup --all flag."""
    result = runner.invoke(
        app,
        ["jobs", "cleanup", "--all"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "Cleaned up" in result.stdout


def test_jobs_cleanup_failed(tmp_path: Path):
    """Test jobs cleanup --failed flag."""
    result = runner.invoke(
        app,
        ["jobs", "cleanup", "--failed"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "Cleaned up" in result.stdout


def test_golden_plan_structure(tmp_path: Path, synthetic_video: Path):
    """Golden-file test: verify censor plan JSON structure matches expected schema."""
    plan_json = tmp_path / "golden.json"

    result = runner.invoke(
        app,
        [
            "plan",
            str(synthetic_video),
            "--output",
            str(plan_json),
            "--no-clip",
            "--no-audio",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    with open(plan_json) as f:
        data = json.load(f)

    # Verify top-level schema
    assert "pureframe_version" in data
    assert "plan_version" in data
    assert data["plan_version"] == 1
    assert "input_metadata" in data
    assert "config_snapshot" in data
    assert "shots" in data
    assert "verdicts" in data
    assert "generated_at" in data

    # Verify input_metadata
    meta = data["input_metadata"]
    assert "duration_seconds" in meta
    assert "fps" in meta
    assert "width" in meta
    assert "height" in meta
    assert meta["duration_seconds"] > 0
    assert meta["fps"]  # Can be a string like "24/1" or a number

    # Verify shots structure
    assert len(data["shots"]) > 0
    for shot in data["shots"]:
        assert "index" in shot
        assert "start_frame" in shot
        assert "end_frame" in shot
        assert "start_time" in shot
        assert "end_time" in shot
        assert shot["end_frame"] > shot["start_frame"]

    # Verify verdicts structure
    assert len(data["verdicts"]) == len(data["shots"])
    for v in data["verdicts"]:
        assert "shot_index" in v
        assert "action" in v
        assert "category" in v
        assert "confidence" in v
        assert "reasoning" in v
        assert v["action"] in ["NONE", "BLACK_BOX", "FULL_FRAME_BLUR"]
        assert 0.0 <= v["confidence"] <= 1.0

    # Config snapshot should contain our flags
    cs = data["config_snapshot"]
    assert cs["no_clip"] is True
    assert cs["no_audio"] is True
