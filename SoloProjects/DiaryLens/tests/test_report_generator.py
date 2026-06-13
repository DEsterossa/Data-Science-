import json
from pathlib import Path
from unittest.mock import patch

import pytest

from diarylens.schemas import WeeklyAggregation, save_json
from diarylens.report_generator import (
    ReportGenerationError,
    clean_llm_markdown_report,
    generate_report,
    load_optional_context_file,
    render_weekly_report_prompt,
    report_output_path,
    weekly_json_relative_path,
)

MOCK_WEEKLY_JSON = {
    "type": "week",
    "week_id": "2026-W22",
    "start_date": "2026-05-25",
    "end_date": "2026-05-31",
    "days_included": ["2026-05-26"],
    "missing_days": [],
    "week_essence": [
        {
            "summary": "Неделя содержала учебные и бытовые задачи.",
            "evidence": [
                {
                    "date": "2026-05-26",
                    "source_day_md": "data/processed/days_md/2026-05-26.md",
                    "source_daily_json": "data/processed/days_json/2026-05-26.json",
                    "source_field": "study_signals",
                    "quote": "поделал лабы по мбп",
                    "note": "учёба",
                }
            ],
        }
    ],
    "main_events": [],
    "main_wins": [],
    "main_tensions": [],
    "emotional_background": [],
    "body_energy": [],
    "study_and_projects": [],
    "social_context": [],
    "actual_focus": [],
    "repeated_topics": [],
    "important_contradictions": [],
    "open_loops": [],
    "risks_next_week": [],
    "next_week_focus_candidates": [],
    "what_not_to_do": [],
    "short_summary": "Короткое резюме недели.",
}

MOCK_REPORT = """# Weekly Review — 2026-W22

## 1. Суть недели

- Неделя содержала учебные и бытовые задачи.

## 2. Что реально происходило

Были учебные задачи и бытовые события.

## 3. Wins

- Был выполнен учебный эпизод.

## 4. Главные напряжения

- Были отдельные напряжения.

## 5. Эмоциональный фон и энергия

Кратко.

## 6. Фактический фокус недели

Кратко.

## 7. Повторяющиеся темы и противоречия

Кратко.

## 8. Open loops

- Нет явных open loops.

## 9. Разумный следующий шаг

- Закрыть один конкретный учебный хвост.

## 10. Что не делать

- Не раздувать список задач.

## 11. Evidence notes

- 2026-05-26 — "поделал лабы по мбп"
"""


def _setup_project(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    weeks_json_dir = tmp_path / "data" / "processed" / "weeks_json"
    reports_dir = tmp_path / "data" / "reports"
    context_dir = tmp_path / "data" / "context"
    prompts_dir = tmp_path / "prompts"
    for directory in (weeks_json_dir, reports_dir, context_dir, prompts_dir):
        directory.mkdir(parents=True)
    return tmp_path, weeks_json_dir, reports_dir, context_dir, prompts_dir


def _save_weekly_json(weeks_json_dir: Path) -> None:
    save_json(
        WeeklyAggregation.model_validate(MOCK_WEEKLY_JSON),
        weeks_json_dir / "2026-W22.json",
    )


def test_load_optional_context_file_returns_empty_when_missing(tmp_path):
    assert load_optional_context_file(tmp_path / "missing.md") == ""


def test_load_optional_context_file_reads_existing_file(tmp_path):
    path = tmp_path / "goals.md"
    path.write_text("Цель: учёба", encoding="utf-8")
    assert load_optional_context_file(path) == "Цель: учёба"


def test_render_weekly_report_prompt_replaces_placeholders():
    rendered = render_weekly_report_prompt(
        "week={week_id}\nstart={start_date}\nend={end_date}\n"
        "goals={goals_context}\nuser={user_context}\n{weekly_json}",
        week_id="2026-W22",
        start_date="2026-05-25",
        end_date="2026-05-31",
        weekly_json='{"week_id": "2026-W22"}',
        goals_context="Мои цели",
        user_context="Обо мне",
    )

    assert "week=2026-W22" in rendered
    assert "start=2026-05-25" in rendered
    assert "end=2026-05-31" in rendered
    assert "goals=Мои цели" in rendered
    assert "user=Обо мне" in rendered
    assert '{"week_id": "2026-W22"}' in rendered


def test_clean_llm_markdown_report_strips_fences_and_prefix():
    wrapped = f"```markdown\n{MOCK_REPORT}\n```"
    cleaned = clean_llm_markdown_report(wrapped, "2026-W22")
    assert cleaned.startswith("# Weekly Review — 2026-W22")
    assert "```" not in cleaned


def test_generate_report_raises_when_weekly_json_missing(tmp_path):
    _, weeks_json_dir, reports_dir, context_dir, prompts_dir = _setup_project(tmp_path)
    (prompts_dir / "weekly_report.md").write_text("prompt", encoding="utf-8")

    with pytest.raises(ReportGenerationError, match="weekly JSON not found"):
        generate_report(
            "2026-W22",
            weeks_json_dir=weeks_json_dir,
            reports_dir=reports_dir,
            prompt_path=prompts_dir / "weekly_report.md",
            goals_context_path=context_dir / "goals.md",
            user_context_path=context_dir / "context_about_me.md",
        )


@patch("diarylens.report_generator.generate_text")
def test_generate_report_saves_markdown(mock_generate_text, tmp_path, monkeypatch):
    _, weeks_json_dir, reports_dir, context_dir, prompts_dir = _setup_project(tmp_path)
    monkeypatch.setenv("GIGACHAT_MODEL_REPORT", "GigaChat-2-Pro")

    _save_weekly_json(weeks_json_dir)
    (prompts_dir / "weekly_report.md").write_text(
        "week={week_id}\nstart={start_date}\nend={end_date}\n"
        "goals={goals_context}\nuser={user_context}\n{weekly_json}",
        encoding="utf-8",
    )
    (context_dir / "goals.md").write_text("Цели недели", encoding="utf-8")
    (context_dir / "context_about_me.md").write_text("Контекст пользователя", encoding="utf-8")
    mock_generate_text.return_value = MOCK_REPORT

    result = generate_report(
        "2026-W22",
        weeks_json_dir=weeks_json_dir,
        reports_dir=reports_dir,
        prompt_path=prompts_dir / "weekly_report.md",
        goals_context_path=context_dir / "goals.md",
        user_context_path=context_dir / "context_about_me.md",
    )

    assert result.output_path == report_output_path("2026-W22", reports_dir)
    assert result.output_path.exists()
    saved = result.output_path.read_text(encoding="utf-8")
    assert saved.startswith("# Weekly Review — 2026-W22")
    assert "поделал лабы по мбп" in saved

    mock_generate_text.assert_called_once()
    assert mock_generate_text.call_args.kwargs["model"] == "GigaChat-2-Pro"

    prompt = mock_generate_text.call_args.args[0]
    assert "week=2026-W22" in prompt
    assert "start=2026-05-25" in prompt
    assert "end=2026-05-31" in prompt
    assert "goals=Цели недели" in prompt
    assert "user=Контекст пользователя" in prompt
    assert '"week_id": "2026-W22"' in prompt


@patch("diarylens.report_generator.generate_text")
def test_generate_report_uses_weekly_model_when_report_env_missing(
    mock_generate_text, tmp_path, monkeypatch
):
    _, weeks_json_dir, reports_dir, context_dir, prompts_dir = _setup_project(tmp_path)
    monkeypatch.delenv("GIGACHAT_MODEL_REPORT", raising=False)
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "GigaChat-2")

    _save_weekly_json(weeks_json_dir)
    (prompts_dir / "weekly_report.md").write_text("prompt", encoding="utf-8")
    mock_generate_text.return_value = MOCK_REPORT

    generate_report(
        "2026-W22",
        weeks_json_dir=weeks_json_dir,
        reports_dir=reports_dir,
        prompt_path=prompts_dir / "weekly_report.md",
        goals_context_path=context_dir / "goals.md",
        user_context_path=context_dir / "context_about_me.md",
    )

    assert mock_generate_text.call_args.kwargs["model"] == "GigaChat-2"


@patch("diarylens.cli.generate_report")
def test_generate_report_cli_success(mock_generate, capsys):
    from diarylens.cli import _run_generate_report
    from diarylens.report_generator import GenerateReportResult

    mock_generate.return_value = GenerateReportResult(
        week_id="2026-W22",
        input_path=Path("data/processed/weeks_json/2026-W22.json"),
        output_path=Path("data/reports/2026-W22_weekly_report.md"),
    )

    _run_generate_report("2026-W22")
    captured = capsys.readouterr()

    assert "Weekly report generated." in captured.out
    assert "Week ID: 2026-W22" in captured.out
    assert weekly_json_relative_path("2026-W22") in captured.out
    assert "data/reports/2026-W22_weekly_report.md" in captured.out


@patch("diarylens.cli.generate_report")
def test_generate_report_cli_does_not_leak_auth_key(mock_generate, monkeypatch, capsys):
    from diarylens.cli import _run_generate_report
    from diarylens.report_generator import GenerateReportResult

    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "super-secret-auth-key")
    mock_generate.return_value = GenerateReportResult(
        week_id="2026-W22",
        input_path=Path("data/processed/weeks_json/2026-W22.json"),
        output_path=Path("data/reports/2026-W22_weekly_report.md"),
    )

    _run_generate_report("2026-W22")
    captured = capsys.readouterr()

    assert "super-secret-auth-key" not in captured.out
    assert "super-secret-auth-key" not in captured.err
