"""Extended CLI integration tests for coverage improvement."""

import json

from typer.testing import CliRunner

from pureframe.cli import app
from pureframe.pipeline.render.plan import CensorPlan
from pureframe.pipeline.shots import Action

runner = CliRunner()


class TestCLIHelpAndVersion:
    def test_help_text(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "PureFrame CLI" in result.stdout

    def test_plan_help(self):
        result = runner.invoke(
            app, ["plan", "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"}
        )
        assert result.exit_code == 0
        # Strip ANSI escape codes - Rich/Typer help rendering varies by terminal width.
        import re

        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        assert "--output" in clean or "-o" in clean

    def test_apply_help(self):
        result = runner.invoke(app, ["apply", "--help"])
        assert result.exit_code == 0

    def test_process_help(self):
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0

    def test_jobs_help(self):
        result = runner.invoke(app, ["jobs", "--help"])
        assert result.exit_code == 0

    def test_preview_help(self):
        result = runner.invoke(app, ["preview", "--help"])
        assert result.exit_code == 0


class TestCLIPlanOptions:
    def test_plan_with_content_type(self, tmp_path, synthetic_video):
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
                "anime",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert plan_json.exists()

        with open(plan_json) as f:
            data = json.load(f)
        assert data["config_snapshot"]["content_type"] == "anime"

    def test_plan_with_strictness(self, tmp_path, synthetic_video):
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

        with open(plan_json) as f:
            data = json.load(f)
        assert data["config_snapshot"]["strictness"] == "high"

    def test_plan_with_both(self, tmp_path, synthetic_video):
        plan_json = tmp_path / "plan_both.json"
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
                "low-light",
                "--strictness",
                "low",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        with open(plan_json) as f:
            data = json.load(f)
        assert data["config_snapshot"]["content_type"] == "low-light"
        assert data["config_snapshot"]["strictness"] == "low"


class TestCLIJobsSubcommands:
    def test_jobs_list(self):
        result = runner.invoke(app, ["jobs", "list"])
        # Should succeed even if no jobs exist
        assert result.exit_code == 0

    def test_jobs_cleanup_no_args(self):
        result = runner.invoke(app, ["jobs", "cleanup"])
        assert result.exit_code == 0


class TestCLIPlanWhitelist:
    def test_plan_whitelist_command(self, tmp_path, synthetic_video):
        # First generate a plan
        plan_json = tmp_path / "whitelist_plan.json"
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

        # Get shot count
        plan = CensorPlan.load(plan_json)
        if len(plan.verdicts) > 0:
            # Whitelist shot 0
            result = runner.invoke(
                app,
                ["plan-whitelist", str(plan_json), "0"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0

            # Verify it was whitelisted
            plan_after = CensorPlan.load(plan_json)
            assert plan_after.verdicts[0].action == Action.NONE


class TestCLIPlanEdit:
    def test_plan_edit_command(self, tmp_path, synthetic_video, monkeypatch):
        plan_json = tmp_path / "edit_plan.json"
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

        # Use 'true' as editor (no-op - doesn't modify file)
        monkeypatch.setenv("EDITOR", "true")
        if CensorPlan.load(plan_json).verdicts:
            result = runner.invoke(
                app,
                ["plan-edit", str(plan_json)],
                catch_exceptions=False,
            )
            assert result.exit_code == 0


class TestCLIPreview:
    def test_preview_generates_html(self, tmp_path, synthetic_video):
        plan_json = tmp_path / "preview_plan.json"
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

        html_out = tmp_path / "preview.html"
        result = runner.invoke(
            app,
            ["preview", str(plan_json), "--output", str(html_out)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Output message should indicate preview was generated or no flagged shots
        assert (
            "preview" in result.stdout.lower()
            or "no flagged" in result.stdout.lower()
            or html_out.exists()
        )


class TestCLIInvalidInputs:
    def test_plan_nonexistent_file(self, tmp_path):
        result = runner.invoke(
            app,
            ["plan", str(tmp_path / "nonexistent.mp4")],
        )
        assert result.exit_code != 0

    def test_apply_nonexistent_plan(self, tmp_path):
        result = runner.invoke(
            app,
            ["apply", str(tmp_path / "nonexistent.json")],
        )
        assert result.exit_code != 0
