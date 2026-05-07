from pathlib import Path
from typer.testing import CliRunner

from pureframe.cli import app
from pureframe.pipeline.render.plan import CensorPlan
from pureframe.pipeline.shots import Action

runner = CliRunner()


def test_plan_and_apply_commands(tmp_path: Path, synthetic_video: Path):
    output_video = tmp_path / "final.mp4"
    plan_json = tmp_path / "plan.json"

    # 1. Run plan command
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
    )

    assert result.exit_code == 0, f"STDOUT:\n{result.stdout}\nEXCEPTION:\n{result.exception}"
    assert plan_json.exists()

    # 2. Check plan contents
    plan = CensorPlan.load(plan_json)
    assert len(plan.shots) > 0
    assert plan.input_metadata.duration_seconds > 0
    assert "no_clip" in plan.config_snapshot
    assert plan.config_snapshot["no_clip"] is True

    # Let's say we want to whitelist the first shot that has an action
    for v in plan.verdicts:
        if v.action != Action.NONE:
            v.action = Action.NONE

    # Resave plan
    plan.serialize(plan_json)

    # 3. Run apply command
    result_apply = runner.invoke(
        app,
        ["apply", str(synthetic_video), str(plan_json), "--output", str(output_video)],
    )

    assert result_apply.exit_code == 0, f"STDOUT:\n{result_apply.stdout}\nEXCEPTION:\n{result_apply.exception}"
    assert output_video.exists()
    assert output_video.stat().st_size > 0


def test_plan_edit_and_whitelist(tmp_path: Path, synthetic_video: Path, monkeypatch):
    plan_json = tmp_path / "plan2.json"

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
    )

    # Get a shot that was flagged
    plan = CensorPlan.load(plan_json)

    target_shot_idx = -1
    for v in plan.verdicts:
        if v.action != Action.NONE:
            target_shot_idx = v.shot_index
            break

    if target_shot_idx == -1:
        # If no shot was flagged, just pick shot 0
        target_shot_idx = 0
        plan.verdicts[0].action = Action.BLACK_BOX
        plan.serialize(plan_json)

    # Whitelist command
    res = runner.invoke(app, ["plan-whitelist", str(plan_json), str(target_shot_idx)])

    assert res.exit_code == 0

    plan2 = CensorPlan.load(plan_json)
    for v in plan2.verdicts:
        if v.shot_index == target_shot_idx:
            assert v.action == Action.NONE

    # Mock plan-edit
    # We can't actually launch nano, so we mock os.environ["EDITOR"] to something harmless like "cat" or a script
    import sys

    editor_script = tmp_path / "editor.py"
    editor_script.write_text("import sys\nwith open(sys.argv[1], 'r') as f:\n  pass\n")

    monkeypatch.setenv("EDITOR", f"{sys.executable} {editor_script}")

    res_edit = runner.invoke(app, ["plan-edit", str(plan_json)])

    assert res_edit.exit_code == 0
    assert "Plan valid and saved" in res_edit.stdout
