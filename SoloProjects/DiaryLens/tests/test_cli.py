import subprocess
import sys
from pathlib import Path

import pytest

from diarylens import __version__
from diarylens import config
from diarylens.cli import main
from diarylens.config import ensure_project_dirs
from diarylens.week_runner import WeekRunnerError


def test_version():
    assert __version__ == "0.1.0"


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "diarylens" in result.stdout
    assert "Weekly diary analysis pipeline" in result.stdout


def test_extract_pdf_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "extract-pdf", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "extract-pdf" in result.stdout
    assert "--week-id" in result.stdout


def test_clean_text_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "clean-text", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "clean-text" in result.stdout
    assert "--week-id" in result.stdout


def test_split_days_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "split-days", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "split-days" in result.stdout
    assert "--week-id" in result.stdout


def test_generate_report_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "generate-report", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "generate-report" in result.stdout
    assert "--week-id" in result.stdout


def test_aggregate_week_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "aggregate-week", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aggregate-week" in result.stdout
    assert "--week-id" in result.stdout


def test_extract_daily_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "extract-daily", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "extract-daily" in result.stdout
    assert "--week-id" in result.stdout
    assert "--verify" in result.stdout


def test_test_llm_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "test-llm", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "test-llm" in result.stdout
    assert "--model" in result.stdout


def test_run_week_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "run-week", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "run-week" in result.stdout
    assert "--week-id" in result.stdout
    assert "--pdf" in result.stdout
    assert "--no-verify" in result.stdout
    assert "--force" in result.stdout


def test_run_week_cli_no_verify_and_force(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_week_pipeline(
        week_id: str,
        pdf_path,
        *,
        verify_daily: bool = True,
        force: bool = False,
    ):
        captured["week_id"] = week_id
        captured["pdf_path"] = pdf_path
        captured["verify_daily"] = verify_daily
        captured["force"] = force
        return Path("data/reports/2026-W22_weekly_report.md")

    monkeypatch.setattr("diarylens.cli.run_week_pipeline", fake_run_week_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diarylens",
            "run-week",
            "--week-id",
            "2026-W22",
            "--pdf",
            "data/raw/weekly/2026-W22.pdf",
            "--no-verify",
            "--force",
        ],
    )

    main()

    assert captured["week_id"] == "2026-W22"
    assert captured["verify_daily"] is False
    assert captured["force"] is True


def test_run_week_cli_report_exists_error(monkeypatch, capsys):
    def fake_run_week_pipeline(*args, **kwargs):
        raise WeekRunnerError(
            "report already exists: data/reports/2026-W22_weekly_report.md\n"
            "Use --force to overwrite."
        )

    monkeypatch.setattr("diarylens.cli.run_week_pipeline", fake_run_week_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diarylens",
            "run-week",
            "--week-id",
            "2026-W22",
            "--pdf",
            "data/raw/weekly/2026-W22.pdf",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "report already exists" in captured.err
    assert "Use --force to overwrite." in captured.err



def test_telegram_bot_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "telegram-bot", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "telegram-bot" in result.stdout


def test_telegram_bot_cli_calls_runner(monkeypatch):
    called = {"value": False}

    def fake_run_telegram_bot():
        called["value"] = True

    monkeypatch.setattr("diarylens.cli.run_telegram_bot", fake_run_telegram_bot)
    monkeypatch.setattr(sys, "argv", ["diarylens", "telegram-bot"])

    main()

    assert called["value"] is True


def test_ensure_project_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "diarylens.config.PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "diarylens.config.DATA_DIR",
        tmp_path / "data",
    )
    monkeypatch.setattr(
        "diarylens.config.RAW_WEEKLY_DIR",
        tmp_path / "data" / "raw" / "weekly",
    )
    monkeypatch.setattr(
        "diarylens.config.RAW_ARCHIVE_DIR",
        tmp_path / "data" / "raw" / "archive",
    )
    monkeypatch.setattr(
        "diarylens.config.INTERIM_RAW_TEXT_DIR",
        tmp_path / "data" / "interim" / "raw_text",
    )
    monkeypatch.setattr(
        "diarylens.config.INTERIM_CLEAN_TEXT_DIR",
        tmp_path / "data" / "interim" / "clean_text",
    )
    monkeypatch.setattr(
        "diarylens.config.DAYS_MD_DIR",
        tmp_path / "data" / "processed" / "days_md",
    )
    monkeypatch.setattr(
        "diarylens.config.DAY_MANIFESTS_DIR",
        tmp_path / "data" / "processed" / "day_manifests",
    )
    monkeypatch.setattr(
        "diarylens.config.DAYS_JSON_DIR",
        tmp_path / "data" / "processed" / "days_json",
    )
    monkeypatch.setattr(
        "diarylens.config.DAYS_JSON_FAILED_DIR",
        tmp_path / "data" / "processed" / "days_json_failed",
    )
    monkeypatch.setattr(
        "diarylens.config.WEEKS_JSON_DIR",
        tmp_path / "data" / "processed" / "weeks_json",
    )
    monkeypatch.setattr(
        "diarylens.config.REPORTS_DIR",
        tmp_path / "data" / "reports",
    )
    monkeypatch.setattr(
        "diarylens.config.DATA_CONTEXT_DIR",
        tmp_path / "data" / "context",
    )
    monkeypatch.setattr(
        "diarylens.config.PROMPTS_DIR",
        tmp_path / "prompts",
    )
    monkeypatch.setattr(
        "diarylens.config.PROJECT_DIRS",
        (
            tmp_path / "data" / "raw" / "weekly",
            tmp_path / "data" / "raw" / "archive",
            tmp_path / "data" / "interim" / "raw_text",
            tmp_path / "data" / "interim" / "clean_text",
            tmp_path / "data" / "processed" / "days_md",
            tmp_path / "data" / "processed" / "day_manifests",
            tmp_path / "data" / "processed" / "days_json",
            tmp_path / "data" / "processed" / "days_json_failed",
            tmp_path / "data" / "processed" / "weeks_json",
            tmp_path / "data" / "reports",
            tmp_path / "data" / "context",
            tmp_path / "prompts",
        ),
    )

    ensure_project_dirs()

    for path in config.PROJECT_DIRS:
        assert path.is_dir()
