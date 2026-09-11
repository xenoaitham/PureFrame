"""Additional CLI coverage tests - version, edge cases, error paths."""

from typer.testing import CliRunner

from pureframe.cli import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "PureFrame" in result.stdout

    def test_short_version_flag(self):
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "PureFrame" in result.stdout


class TestPlanDefaults:
    def test_plan_auto_output(self, tmp_path, synthetic_video):
        """Plan with no --output should auto-generate filename."""
        import shutil

        test_video = tmp_path / "test_video.mp4"
        shutil.copy(synthetic_video, test_video)

        result = runner.invoke(
            app,
            [
                "plan",
                str(test_video),
                "--no-clip",
                "--no-audio",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        expected_plan = test_video.with_name("test_video.mp4.censorplan.json")
        assert expected_plan.exists()

    def test_plan_verbose(self, tmp_path, synthetic_video):
        plan_json = tmp_path / "verbose_plan.json"
        result = runner.invoke(
            app,
            [
                "plan",
                str(synthetic_video),
                "--output",
                str(plan_json),
                "--no-clip",
                "--no-audio",
                "--verbose",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0


class TestApplyEdgeCases:
    def test_apply_invalid_json(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not json")
        result = runner.invoke(app, ["apply", str(bad_json)])
        assert result.exit_code != 0


class TestProcessDefaults:
    def test_process_nonexistent(self, tmp_path):
        result = runner.invoke(
            app,
            ["process", str(tmp_path / "nonexistent.mp4")],
        )
        assert result.exit_code != 0


class TestJobsSubcommands:
    def test_jobs_list_table(self):
        result = runner.invoke(app, ["jobs", "list"])
        assert result.exit_code == 0

    def test_jobs_cleanup_all(self):
        result = runner.invoke(app, ["jobs", "cleanup", "--all"])
        assert result.exit_code == 0

    def test_jobs_cleanup_failed(self):
        result = runner.invoke(app, ["jobs", "cleanup", "--failed"])
        assert result.exit_code == 0
