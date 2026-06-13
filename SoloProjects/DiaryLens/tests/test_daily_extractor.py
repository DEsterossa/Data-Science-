import json
from pathlib import Path
from unittest.mock import patch

import pytest

from diarylens.daily_extractor import (
    DailyExtractionError,
    build_empty_daily_extraction,
    day_json_output_path,
    extract_daily_for_week,
    extract_day_with_fallback,
    failed_day_output_path,
    format_extract_daily_summary,
    is_refusal_response,
    load_prompt_template,
    parse_llm_json_response,
    render_prompt,
    render_verification_prompt,
    validate_daily_extraction,
    verify_day_extraction,
)
from diarylens.schemas import DailyExtraction, load_json, save_json


SAMPLE_MANIFEST = {
    "week_id": "2026-W22",
    "source_clean_path": "data/interim/clean_text/2026-W22_clean.md",
    "days_count": 2,
    "days": [
        {
            "date": "2026-05-25",
            "original_header": "25 мая 2026 г. Пн",
            "weekday": "Пн",
            "path": "data/processed/days_md/2026-05-25.md",
            "source_order": 2,
        },
        {
            "date": "2026-05-26",
            "original_header": "26 мая 2026 г. Вт",
            "weekday": "Вт",
            "path": "data/processed/days_md/2026-05-26.md",
            "source_order": 1,
        },
    ],
}

SAMPLE_DAY_MD = """# Diary day: 2026-05-26

Original header: 26 мая 2026 г. Вт
Week ID: 2026-W22

Сделал лабы и погулял.
"""

REFUSAL_TEXT = "Генеративные языковые модели не обладают собственным мнением"

SAMPLE_LLM_JSON = """{
  "date": "2026-05-26",
  "week_id": "2026-W22",
  "source_day_md": "data/processed/days_md/2026-05-26.md",
  "important_moments": [
    {
      "quote": "посидел, сделал лабы по мбп",
      "note": "учёба"
    }
  ],
  "wins": [],
  "tensions": [],
  "emotions": [],
  "body_energy_signals": [
    {
      "quote": "съел удон в обед",
      "note": "еда"
    }
  ],
  "study_signals": [
    {
      "quote": "сделал лабы по мбп",
      "note": "лабораторные работы"
    }
  ],
  "ml_ds_signals": [],
  "social_signals": [],
  "decisions": [],
  "open_questions": [],
  "key_quotes": [],
  "short_summary": "День включал лабораторные работы, еду и обычные бытовые дела."
}"""


def _valid_payload(date: str) -> dict:
    return {
        "date": date,
        "week_id": "2026-W22",
        "source_day_md": f"data/processed/days_md/{date}.md",
        "important_moments": [{"quote": f"Момент {date}", "note": "событие"}],
        "wins": [],
        "tensions": [],
        "emotions": [],
        "body_energy_signals": [],
        "study_signals": [],
        "ml_ds_signals": [],
        "social_signals": [],
        "decisions": [],
        "open_questions": [],
        "key_quotes": [],
        "short_summary": f"День {date}.",
    }


def _setup_project(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    project_root = tmp_path / "project"
    manifest_dir = project_root / "data" / "processed" / "day_manifests"
    days_md_dir = project_root / "data" / "processed" / "days_md"
    days_json_dir = project_root / "data" / "processed" / "days_json"
    days_json_failed_dir = project_root / "data" / "processed" / "days_json_failed"
    prompts_dir = project_root / "prompts"
    for directory in (
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ):
        directory.mkdir(parents=True)
    return (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    )


def test_render_prompt_replaces_placeholders(tmp_path):
    _, _, _, _, _, prompts_dir = _setup_project(tmp_path)
    prompt_path = prompts_dir / "daily_extraction.md"
    prompt_path.write_text(
        "date={date}\nweek={week_id}\nsource={source_day_md}\ntext={day_text}\n",
        encoding="utf-8",
    )

    rendered = render_prompt(
        load_prompt_template(prompt_path),
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        day_text="Текст дня",
    )

    assert "date=2026-05-26" in rendered
    assert "week=2026-W22" in rendered


def test_render_verification_prompt_includes_draft_json():
    rendered = render_verification_prompt(
        "day={day_text}\ndraft={draft_json}",
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        day_text="Текст дня",
        draft_json='{"important_moments": []}',
    )

    assert "day=Текст дня" in rendered
    assert 'draft={"important_moments": []}' in rendered


@patch("diarylens.daily_extractor.generate_text")
def test_primary_success_does_not_call_fallback(mock_generate_text, monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")
    mock_generate_text.return_value = json.dumps(
        _valid_payload("2026-05-30"),
        ensure_ascii=False,
    )

    extraction, failed = extract_day_with_fallback(
        date="2026-05-30",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-30.md",
        day_text="Текст дня",
        prompt_template="date={date}",
    )

    assert extraction is not None
    assert failed is None
    assert mock_generate_text.call_count == 1
    assert mock_generate_text.call_args.kwargs["model"] == "GigaChat-2"


@patch("diarylens.daily_extractor.generate_text")
def test_refusal_creates_failed_artifact_without_fallback(mock_generate_text, monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")
    mock_generate_text.return_value = REFUSAL_TEXT

    extraction, failed = extract_day_with_fallback(
        date="2026-05-30",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-30.md",
        day_text="Текст дня",
        prompt_template="date={date}",
    )

    assert extraction is None
    assert failed is not None
    assert failed["error_type"] == "llm_refusal"
    assert failed["stage"] == "extraction"
    assert mock_generate_text.call_count == 1
    assert mock_generate_text.call_args.kwargs["model"] == "GigaChat-2"


@patch("diarylens.daily_extractor.generate_text")
def test_primary_failure_creates_failed_artifact(
    mock_generate_text, tmp_path, monkeypatch
):
    (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ) = _setup_project(tmp_path)
    monkeypatch.setattr("diarylens.daily_extractor.PROJECT_ROOT", project_root)
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")

    manifest = {
        "week_id": "2026-W22",
        "days": [
            {
                "date": "2026-05-30",
                "path": "data/processed/days_md/2026-05-30.md",
            }
        ],
    }
    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-05-30.md").write_text("День 30", encoding="utf-8")
    (prompts_dir / "daily_extraction.md").write_text("{day_text}", encoding="utf-8")

    mock_generate_text.return_value = REFUSAL_TEXT

    result = extract_daily_for_week(
        "2026-W22",
        manifest_dir=manifest_dir,
        days_md_dir=days_md_dir,
        output_dir=days_json_dir,
        failed_dir=days_json_failed_dir,
        prompt_path=prompts_dir / "daily_extraction.md",
    )

    failed_path = failed_day_output_path("2026-05-30", days_json_failed_dir)
    assert failed_path.exists()
    assert result.has_failures
    assert result.failed[0].error_type == "llm_refusal"
    assert result.verified == 0

    artifact = json.loads(failed_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "failed"
    assert artifact["stage"] == "extraction"
    assert artifact["model"] == "GigaChat-2"
    assert mock_generate_text.call_count == 1
    assert day_json_output_path("2026-05-30", days_json_dir).exists()

    placeholder = load_json(
        day_json_output_path("2026-05-30", days_json_dir),
        DailyExtraction,
    )
    assert placeholder.short_summary == ""
    assert placeholder.important_moments == []
    assert placeholder.wins == []


def test_build_empty_daily_extraction_has_all_fields():
    extraction = build_empty_daily_extraction(
        date="2026-05-30",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-30.md",
    )

    assert extraction.date == "2026-05-30"
    assert extraction.week_id == "2026-W22"
    assert extraction.short_summary == ""
    assert extraction.important_moments == []
    assert extraction.emotions == []
    assert extraction.key_quotes == []


@patch("diarylens.daily_extractor.generate_text")
def test_failure_on_one_day_continues_processing_other_days(
    mock_generate_text, tmp_path, monkeypatch
):
    (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ) = _setup_project(tmp_path)
    monkeypatch.setattr("diarylens.daily_extractor.PROJECT_ROOT", project_root)
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")

    manifest = {
        "week_id": "2026-W22",
        "days": [
            {"date": "2026-05-30", "path": "data/processed/days_md/2026-05-30.md"},
            {"date": "2026-05-31", "path": "data/processed/days_md/2026-05-31.md"},
        ],
    }
    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-05-30.md").write_text("День 30", encoding="utf-8")
    (days_md_dir / "2026-05-31.md").write_text("День 31", encoding="utf-8")
    (prompts_dir / "daily_extraction.md").write_text("{day_text}", encoding="utf-8")

    mock_generate_text.side_effect = [
        REFUSAL_TEXT,
        json.dumps(_valid_payload("2026-05-31"), ensure_ascii=False),
    ]

    result = extract_daily_for_week(
        "2026-W22",
        manifest_dir=manifest_dir,
        days_md_dir=days_md_dir,
        output_dir=days_json_dir,
        failed_dir=days_json_failed_dir,
        prompt_path=prompts_dir / "daily_extraction.md",
    )

    assert result.has_failures
    assert result.processed == 1
    assert len(result.output_paths) == 2
    assert day_json_output_path("2026-05-30", days_json_dir).exists()
    assert day_json_output_path("2026-05-31", days_json_dir).exists()
    assert mock_generate_text.call_count == 2


@patch("diarylens.daily_extractor.generate_text")
def test_without_verify_calls_extraction_only(mock_generate_text, tmp_path, monkeypatch):
    (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ) = _setup_project(tmp_path)
    monkeypatch.setattr("diarylens.daily_extractor.PROJECT_ROOT", project_root)
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")

    manifest = {
        "week_id": "2026-W22",
        "days": [{"date": "2026-05-26", "path": "data/processed/days_md/2026-05-26.md"}],
    }
    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-05-26.md").write_text(SAMPLE_DAY_MD, encoding="utf-8")
    (prompts_dir / "daily_extraction.md").write_text("{day_text}", encoding="utf-8")

    mock_generate_text.return_value = SAMPLE_LLM_JSON

    result = extract_daily_for_week(
        "2026-W22",
        verify=False,
        manifest_dir=manifest_dir,
        days_md_dir=days_md_dir,
        output_dir=days_json_dir,
        failed_dir=days_json_failed_dir,
        prompt_path=prompts_dir / "daily_extraction.md",
    )

    assert result.processed == 1
    assert result.verified == 0
    assert mock_generate_text.call_count == 1


@patch("diarylens.daily_extractor.generate_text")
def test_with_verify_calls_extraction_then_verification(
    mock_generate_text, tmp_path, monkeypatch
):
    (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ) = _setup_project(tmp_path)
    monkeypatch.setattr("diarylens.daily_extractor.PROJECT_ROOT", project_root)
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")

    manifest = {
        "week_id": "2026-W22",
        "days": [{"date": "2026-05-26", "path": "data/processed/days_md/2026-05-26.md"}],
    }
    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-05-26.md").write_text(SAMPLE_DAY_MD, encoding="utf-8")
    (prompts_dir / "daily_extraction.md").write_text("{day_text}", encoding="utf-8")
    (prompts_dir / "daily_verification.md").write_text(
        "{day_text}\n{draft_json}",
        encoding="utf-8",
    )

    verified_payload = _valid_payload("2026-05-26")
    verified_payload["important_moments"] = [
        {"quote": "сделал лабы по мбп", "note": "лабораторные работы"}
    ]
    mock_generate_text.side_effect = [SAMPLE_LLM_JSON, json.dumps(verified_payload)]

    result = extract_daily_for_week(
        "2026-W22",
        verify=True,
        manifest_dir=manifest_dir,
        days_md_dir=days_md_dir,
        output_dir=days_json_dir,
        failed_dir=days_json_failed_dir,
        prompt_path=prompts_dir / "daily_extraction.md",
        verification_prompt_path=prompts_dir / "daily_verification.md",
    )

    assert result.processed == 1
    assert result.verified == 1
    assert mock_generate_text.call_count == 2

    saved = load_json(day_json_output_path("2026-05-26", days_json_dir), DailyExtraction)
    assert saved.important_moments[0].quote == "сделал лабы по мбп"

    verify_prompt = mock_generate_text.call_args_list[1].args[0]
    assert "сделал лабы по мбп" in verify_prompt


@patch("diarylens.daily_extractor.generate_text")
def test_verification_failure_creates_failed_artifact_with_stage(
    mock_generate_text, tmp_path, monkeypatch
):
    (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ) = _setup_project(tmp_path)
    monkeypatch.setattr("diarylens.daily_extractor.PROJECT_ROOT", project_root)
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")

    manifest = {
        "week_id": "2026-W22",
        "days": [{"date": "2026-05-26", "path": "data/processed/days_md/2026-05-26.md"}],
    }
    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-05-26.md").write_text(SAMPLE_DAY_MD, encoding="utf-8")
    (prompts_dir / "daily_extraction.md").write_text("{day_text}", encoding="utf-8")
    (prompts_dir / "daily_verification.md").write_text("{draft_json}", encoding="utf-8")

    mock_generate_text.side_effect = [SAMPLE_LLM_JSON, REFUSAL_TEXT]

    result = extract_daily_for_week(
        "2026-W22",
        verify=True,
        manifest_dir=manifest_dir,
        days_md_dir=days_md_dir,
        output_dir=days_json_dir,
        failed_dir=days_json_failed_dir,
        prompt_path=prompts_dir / "daily_extraction.md",
        verification_prompt_path=prompts_dir / "daily_verification.md",
    )

    assert result.has_failures
    assert result.verified == 0
    assert day_json_output_path("2026-05-26", days_json_dir).exists()

    placeholder = load_json(
        day_json_output_path("2026-05-26", days_json_dir),
        DailyExtraction,
    )
    assert placeholder.short_summary == ""

    artifact = json.loads(
        failed_day_output_path("2026-05-26", days_json_failed_dir).read_text(encoding="utf-8")
    )
    assert artifact["stage"] == "verification"
    assert artifact["error_type"] == "llm_refusal"


@patch("diarylens.daily_extractor.generate_text")
def test_extraction_failure_does_not_call_verification(
    mock_generate_text, tmp_path, monkeypatch
):
    (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ) = _setup_project(tmp_path)
    monkeypatch.setattr("diarylens.daily_extractor.PROJECT_ROOT", project_root)
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")

    manifest = {
        "week_id": "2026-W22",
        "days": [{"date": "2026-05-26", "path": "data/processed/days_md/2026-05-26.md"}],
    }
    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-05-26.md").write_text(SAMPLE_DAY_MD, encoding="utf-8")
    (prompts_dir / "daily_extraction.md").write_text("{day_text}", encoding="utf-8")
    (prompts_dir / "daily_verification.md").write_text("{draft_json}", encoding="utf-8")

    mock_generate_text.return_value = REFUSAL_TEXT

    result = extract_daily_for_week(
        "2026-W22",
        verify=True,
        manifest_dir=manifest_dir,
        days_md_dir=days_md_dir,
        output_dir=days_json_dir,
        failed_dir=days_json_failed_dir,
        prompt_path=prompts_dir / "daily_extraction.md",
        verification_prompt_path=prompts_dir / "daily_verification.md",
    )

    assert result.has_failures
    assert mock_generate_text.call_count == 1


@patch("diarylens.daily_extractor.generate_text")
def test_existing_daily_json_is_skipped(mock_generate_text, tmp_path, monkeypatch):
    (
        project_root,
        manifest_dir,
        days_md_dir,
        days_json_dir,
        days_json_failed_dir,
        prompts_dir,
    ) = _setup_project(tmp_path)
    monkeypatch.setattr("diarylens.daily_extractor.PROJECT_ROOT", project_root)

    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(SAMPLE_MANIFEST, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-05-25.md").write_text("День 25", encoding="utf-8")
    (days_md_dir / "2026-05-26.md").write_text(SAMPLE_DAY_MD, encoding="utf-8")
    (prompts_dir / "daily_extraction.md").write_text("{day_text}", encoding="utf-8")

    existing = day_json_output_path("2026-05-26", days_json_dir)
    save_json(
        DailyExtraction(
            date="2026-05-26",
            week_id="2026-W22",
            source_day_md="data/processed/days_md/2026-05-26.md",
            short_summary="Уже готов.",
        ),
        existing,
    )

    mock_generate_text.return_value = json.dumps(
        _valid_payload("2026-05-25"),
        ensure_ascii=False,
    )

    result = extract_daily_for_week(
        "2026-W22",
        manifest_dir=manifest_dir,
        days_md_dir=days_md_dir,
        output_dir=days_json_dir,
        failed_dir=days_json_failed_dir,
        prompt_path=prompts_dir / "daily_extraction.md",
    )

    assert result.skipped == 1
    assert result.processed == 1
    assert result.has_failures is False
    assert mock_generate_text.call_count == 1


@patch("diarylens.cli.extract_daily_for_week")
def test_extract_daily_cli_exit_code_0_on_partial_failures(mock_extract, capsys):
    from diarylens.cli import _run_extract_daily
    from diarylens.daily_extractor import ExtractDailyWeekResult, FailedDaySummary

    mock_extract.return_value = ExtractDailyWeekResult(
        failed=[FailedDaySummary(date="2026-05-30", error_type="llm_refusal")],
        failed_artifact_paths=[Path("data/processed/days_json_failed/2026-05-30_error.json")],
        output_paths=[Path("data/processed/days_json/2026-05-30.json")],
    )

    _run_extract_daily("2026-W22")

    captured = capsys.readouterr()
    assert "Failed: 1" in captured.out
    assert "2026-05-30: llm_refusal" in captured.out


@patch("diarylens.cli.extract_daily_for_week")
def test_extract_daily_cli_exit_code_0_when_all_ok(mock_extract, capsys):
    from diarylens.cli import _run_extract_daily
    from diarylens.daily_extractor import ExtractDailyWeekResult

    mock_extract.return_value = ExtractDailyWeekResult(processed=2, skipped=1, verified=2)

    _run_extract_daily("2026-W22", verify=True)
    captured = capsys.readouterr()
    assert "Failed: 0" in captured.out
    assert "Verified: 2" in captured.out
    mock_extract.assert_called_once_with("2026-W22", verify=True)


@patch("diarylens.daily_extractor.generate_text")
def test_extract_daily_cli_does_not_leak_auth_key(mock_generate_text, monkeypatch, capsys):
    from diarylens.cli import _run_extract_daily

    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "super-secret-auth-key")
    mock_generate_text.return_value = json.dumps(
        _valid_payload("2026-05-26"),
        ensure_ascii=False,
    )

    with patch(
        "diarylens.daily_extractor.extract_daily_for_week",
        return_value=__import__(
            "diarylens.daily_extractor", fromlist=["ExtractDailyWeekResult"]
        ).ExtractDailyWeekResult(processed=1),
    ):
        _run_extract_daily("2026-W22")

    captured = capsys.readouterr()
    assert "super-secret-auth-key" not in captured.out
    assert "super-secret-auth-key" not in captured.err


def test_is_refusal_response_detects_common_phrases():
    assert is_refusal_response(REFUSAL_TEXT)
    assert is_refusal_response("Я не могу выполнить этот запрос")
    assert not is_refusal_response('{"date": "2026-05-26"}')


def test_format_extract_daily_summary():
    from diarylens.daily_extractor import ExtractDailyWeekResult, FailedDaySummary

    summary = format_extract_daily_summary(
        ExtractDailyWeekResult(
            processed=6,
            skipped=1,
            verified=6,
            failed=[FailedDaySummary(date="2026-05-30", error_type="llm_refusal")],
        )
    )
    assert "Processed: 6" in summary
    assert "Verified: 6" in summary
    assert "Skipped: 1" in summary
    assert "Failed: 1" in summary
    assert "- 2026-05-30: llm_refusal" in summary


def test_extract_daily_missing_manifest_raises(tmp_path):
    with pytest.raises(DailyExtractionError, match="Day manifest not found"):
        extract_daily_for_week(
            "2026-W22",
            manifest_dir=tmp_path,
            days_md_dir=tmp_path,
            output_dir=tmp_path,
            failed_dir=tmp_path,
            prompt_path=tmp_path / "missing_prompt.md",
        )


def test_validate_daily_extraction_rejects_invalid_payload():
    with pytest.raises(DailyExtractionError, match="LLM JSON failed validation"):
        validate_daily_extraction({"date": "2026-05-26"})


def test_validate_daily_extraction_rejects_old_string_list_format():
    with pytest.raises(DailyExtractionError, match="LLM JSON failed validation"):
        validate_daily_extraction(
            {
                "date": "2026-05-26",
                "week_id": "2026-W22",
                "source_day_md": "data/processed/days_md/2026-05-26.md",
                "important_moments": ["Погулял с друзьями"],
                "short_summary": "День.",
            }
        )


def test_validate_daily_extraction_rejects_removed_facts_field():
    with pytest.raises(DailyExtractionError, match="LLM JSON failed validation"):
        validate_daily_extraction(
            {
                "date": "2026-05-26",
                "week_id": "2026-W22",
                "source_day_md": "data/processed/days_md/2026-05-26.md",
                "facts": [{"quote": "погулял", "note": "прогулка"}],
                "short_summary": "День.",
            }
        )


def test_parse_llm_json_response_strips_markdown_fences():
    data = parse_llm_json_response("```json\n{\"date\": \"2026-05-26\"}\n```")
    assert data == {"date": "2026-05-26"}


def test_parse_llm_json_response_detects_gigachat_refusal():
    with pytest.raises(DailyExtractionError, match="refused to process"):
        parse_llm_json_response(REFUSAL_TEXT)


def test_parse_llm_json_response_repairs_minor_json_defects():
    broken = """```json
{
  "date": "2026-05-26",
  "short_summary": "Он сказал "привет" и ушел",
}
```"""
    data = parse_llm_json_response(broken)
    assert data["date"] == "2026-05-26"
    assert "привет" in data["short_summary"]


@patch("diarylens.daily_extractor.generate_text")
def test_verify_day_extraction_receives_draft_json(mock_generate_text, monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "GigaChat-2")
    draft = validate_daily_extraction(json.loads(SAMPLE_LLM_JSON))
    mock_generate_text.return_value = SAMPLE_LLM_JSON

    extraction, failed = verify_day_extraction(
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        day_text=SAMPLE_DAY_MD,
        draft=draft,
        verification_template="{day_text}\n{draft_json}",
    )

    assert extraction is not None
    assert failed is None
    prompt = mock_generate_text.call_args.args[0]
    assert "сделал лабы по мбп" in prompt
