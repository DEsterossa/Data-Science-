import json
from pathlib import Path
from unittest.mock import patch

import pytest

from diarylens.schemas import DailyExtraction, WeeklyAggregation, load_json, save_json
from diarylens.weekly_aggregator import (
    WeeklyAggregationError,
    aggregate_week,
    build_daily_jsons_payload,
    collect_daily_jsons_for_week,
    format_missing_days_warning,
    normalize_source_field,
    normalize_weekly_aggregation_data,
    render_weekly_prompt,
    validate_weekly_aggregation,
    week_json_output_path,
)


SAMPLE_MANIFEST = {
    "week_id": "2026-W22",
    "days": [
        {"date": "2026-05-25", "path": "data/processed/days_md/2026-05-25.md"},
        {"date": "2026-05-26", "path": "data/processed/days_md/2026-05-26.md"},
        {"date": "2026-05-30", "path": "data/processed/days_md/2026-05-30.md"},
    ],
}

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


def _setup_project(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    project_root = tmp_path / "project"
    manifest_dir = project_root / "data" / "processed" / "day_manifests"
    days_json_dir = project_root / "data" / "processed" / "days_json"
    weeks_json_dir = project_root / "data" / "processed" / "weeks_json"
    prompts_dir = project_root / "prompts"
    for directory in (manifest_dir, days_json_dir, weeks_json_dir, prompts_dir):
        directory.mkdir(parents=True)
    return project_root, manifest_dir, days_json_dir, weeks_json_dir, prompts_dir


def _save_daily_json(days_json_dir: Path, date: str) -> None:
    save_json(
        DailyExtraction(
            date=date,
            week_id="2026-W22",
            source_day_md=f"data/processed/days_md/{date}.md",
            important_moments=[{"quote": "делал лабы", "note": "учёба"}],
            short_summary=f"День {date}.",
        ),
        days_json_dir / f"{date}.json",
    )


def test_build_daily_jsons_payload_includes_source_daily_json():
    daily = DailyExtraction(
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        short_summary="День.",
    )

    payload = build_daily_jsons_payload([daily])

    assert payload == [
        {
            "date": "2026-05-26",
            "source_daily_json": "data/processed/days_json/2026-05-26.json",
            "daily": daily.model_dump(),
        }
    ]


def test_render_weekly_prompt_uses_json_not_python_repr():
    rendered = render_weekly_prompt(
        "included={days_included}\nmissing={missing_days}\n{daily_jsons}",
        week_id="2026-W22",
        start_date="2026-05-25",
        end_date="2026-05-31",
        days_included=["2026-05-25"],
        missing_days=["2026-05-30"],
        daily_jsons=json.dumps(
            [{"date": "2026-05-25", "source_daily_json": "data/processed/days_json/2026-05-25.json"}],
            ensure_ascii=False,
        ),
    )

    assert '["2026-05-25"]' in rendered
    assert '["2026-05-30"]' in rendered
    assert "'2026-05-25'" not in rendered


def test_render_weekly_prompt_replaces_placeholders():
    rendered = render_weekly_prompt(
        "week={week_id}\nstart={start_date}\nend={end_date}\n"
        "included={days_included}\nmissing={missing_days}\n{daily_jsons}",
        week_id="2026-W22",
        start_date="2026-05-25",
        end_date="2026-05-31",
        days_included=["2026-05-25"],
        missing_days=["2026-05-30"],
        daily_jsons='[{"date": "2026-05-25"}]',
    )

    assert "week=2026-W22" in rendered
    assert "start=2026-05-25" in rendered
    assert "end=2026-05-31" in rendered
    assert '"2026-05-25"' in rendered
    assert '"2026-05-30"' in rendered
    assert '[{"date": "2026-05-25"}]' in rendered


def test_collect_daily_jsons_loads_existing_and_tracks_missing(tmp_path):
    _, _, days_json_dir, _, _ = _setup_project(tmp_path)
    _save_daily_json(days_json_dir, "2026-05-25")

    daily_extractions, days_included, missing_days = collect_daily_jsons_for_week(
        SAMPLE_MANIFEST,
        days_json_dir=days_json_dir,
    )

    assert len(daily_extractions) == 1
    assert days_included == ["2026-05-25"]
    assert missing_days == ["2026-05-26", "2026-05-30"]


def test_aggregate_week_raises_when_no_daily_json(tmp_path):
    manifest_dir, days_json_dir, weeks_json_dir, prompts_dir = _setup_project(tmp_path)[1:]

    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(SAMPLE_MANIFEST, ensure_ascii=False),
        encoding="utf-8",
    )
    (prompts_dir / "weekly_aggregation.md").write_text("prompt", encoding="utf-8")

    with pytest.raises(
        WeeklyAggregationError,
        match="no valid daily JSON files found for week 2026-W22",
    ):
        aggregate_week(
            "2026-W22",
            manifest_dir=manifest_dir,
            days_json_dir=days_json_dir,
            output_dir=weeks_json_dir,
            prompt_path=prompts_dir / "weekly_aggregation.md",
        )


@patch("diarylens.weekly_aggregator.generate_text")
def test_aggregate_week_works_with_missing_days(mock_generate_text, tmp_path, monkeypatch):
    manifest_dir, days_json_dir, weeks_json_dir, prompts_dir = _setup_project(tmp_path)[1:]
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "GigaChat-2-Pro")

    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(SAMPLE_MANIFEST, ensure_ascii=False),
        encoding="utf-8",
    )
    _save_daily_json(days_json_dir, "2026-05-25")
    _save_daily_json(days_json_dir, "2026-05-26")
    (prompts_dir / "weekly_aggregation.md").write_text(
        "week={week_id}\n{daily_jsons}",
        encoding="utf-8",
    )

    mock_generate_text.return_value = json.dumps(MOCK_WEEKLY_JSON, ensure_ascii=False)

    result = aggregate_week(
        "2026-W22",
        manifest_dir=manifest_dir,
        days_json_dir=days_json_dir,
        output_dir=weeks_json_dir,
        prompt_path=prompts_dir / "weekly_aggregation.md",
    )

    assert result.days_included == ["2026-05-25", "2026-05-26"]
    assert result.missing_days == ["2026-05-30"]
    assert result.output_path == week_json_output_path("2026-W22", weeks_json_dir)
    assert result.output_path.exists()

    loaded = load_json(result.output_path, WeeklyAggregation)
    assert loaded.week_id == "2026-W22"
    assert loaded.short_summary == "Короткое резюме недели."

    mock_generate_text.assert_called_once()
    assert mock_generate_text.call_args.kwargs["model"] == "GigaChat-2-Pro"

    prompt = mock_generate_text.call_args.args[0]
    assert "week=2026-W22" in prompt
    assert "2026-05-25" in prompt
    assert "2026-05-26" in prompt
    assert '"source_daily_json": "data/processed/days_json/2026-05-25.json"' in prompt
    assert '"source_daily_json": "data/processed/days_json/2026-05-26.json"' in prompt


@patch("diarylens.weekly_aggregator.generate_text")
def test_aggregate_week_uses_manifest_date_range(mock_generate_text, tmp_path, monkeypatch):
    manifest_dir, days_json_dir, weeks_json_dir, prompts_dir = _setup_project(tmp_path)[1:]
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "GigaChat-2-Pro")

    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(SAMPLE_MANIFEST, ensure_ascii=False),
        encoding="utf-8",
    )
    _save_daily_json(days_json_dir, "2026-05-25")
    (prompts_dir / "weekly_aggregation.md").write_text(
        "{start_date}|{end_date}|{days_included}|{missing_days}",
        encoding="utf-8",
    )
    mock_generate_text.return_value = json.dumps(MOCK_WEEKLY_JSON, ensure_ascii=False)

    aggregate_week(
        "2026-W22",
        manifest_dir=manifest_dir,
        days_json_dir=days_json_dir,
        output_dir=weeks_json_dir,
        prompt_path=prompts_dir / "weekly_aggregation.md",
    )

    prompt = mock_generate_text.call_args.args[0]
    assert "2026-05-25|2026-05-30" in prompt
    assert '"2026-05-25"' in prompt
    assert '"2026-05-26"' in prompt
    assert '"2026-05-30"' in prompt


@patch("diarylens.weekly_aggregator.generate_text")
def test_aggregate_week_raises_on_invalid_llm_json(mock_generate_text, tmp_path, monkeypatch):
    manifest_dir, days_json_dir, weeks_json_dir, prompts_dir = _setup_project(tmp_path)[1:]
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "GigaChat-2-Pro")

    (manifest_dir / "2026-W22_days.json").write_text(
        json.dumps(SAMPLE_MANIFEST, ensure_ascii=False),
        encoding="utf-8",
    )
    _save_daily_json(days_json_dir, "2026-05-25")
    (prompts_dir / "weekly_aggregation.md").write_text("prompt", encoding="utf-8")
    mock_generate_text.return_value = "not json"

    with pytest.raises(WeeklyAggregationError, match="JSON"):
        aggregate_week(
            "2026-W22",
            manifest_dir=manifest_dir,
            days_json_dir=days_json_dir,
            output_dir=weeks_json_dir,
            prompt_path=prompts_dir / "weekly_aggregation.md",
        )

    assert not week_json_output_path("2026-W22", weeks_json_dir).exists()


def test_validate_weekly_aggregation_rejects_invalid_payload():
    with pytest.raises(WeeklyAggregationError, match="LLM JSON failed validation"):
        validate_weekly_aggregation({"week_id": "2026-W22"})


def test_normalize_weekly_aggregation_drops_items_without_evidence():
    repaired = normalize_weekly_aggregation_data(
        {
            "week_essence": [
                "Автор переживал нагрузку.",
                {
                    "summary": "Был учебный блок.",
                    "evidence": [
                        {
                            "date": "2026-05-26",
                            "source_day_md": "data/processed/days_md/2026-05-26.md",
                            "source_daily_json": "data/processed/days_json/2026-05-26.json",
                            "source_field": "study_signals",
                            "quote": "поделал лабы",
                        }
                    ],
                },
            ],
            "risks_next_week": [
                {
                    "summary": "Риск выгорания.",
                    "evidence": [
                        {"date": "2026-05-26"},
                        {
                            "date": "2026-05-29",
                            "quote": "устал после работы",
                        },
                    ],
                }
            ],
        }
    )

    assert len(repaired["week_essence"]) == 1
    assert repaired["week_essence"][0]["summary"] == "Был учебный блок."
    assert repaired["risks_next_week"] == []


@pytest.mark.parametrize(
    ("legacy_field", "canonical_field"),
    [
        ("emotions", "emotional_signals"),
        ("problems", "tensions"),
        ("energy_signals", "body_energy_signals"),
        ("health_signals", "body_energy_signals"),
        ("open_questions", "open_loops"),
    ],
)
def test_normalize_source_field_maps_legacy_daily_fields(legacy_field, canonical_field):
    assert normalize_source_field(legacy_field) == canonical_field
    assert normalize_source_field(canonical_field) == canonical_field


def test_validate_weekly_aggregation_normalizes_legacy_source_field():
    weekly = validate_weekly_aggregation(
        {
            "type": "week",
            "week_id": "2026-W22",
            "start_date": "2026-05-25",
            "end_date": "2026-05-31",
            "days_included": ["2026-05-26"],
            "missing_days": [],
            "week_essence": [
                {
                    "summary": "Эмоциональный фон.",
                    "evidence": [
                        {
                            "date": "2026-05-26",
                            "source_day_md": "data/processed/days_md/2026-05-26.md",
                            "source_daily_json": "data/processed/days_json/2026-05-26.json",
                            "source_field": "emotions",
                            "quote": "скучно",
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
            "short_summary": "Короткое резюме.",
        }
    )

    assert weekly.week_essence[0].evidence[0].source_field == "emotional_signals"


def test_validate_weekly_aggregation_fills_missing_short_summary_with_empty_string():
    weekly = validate_weekly_aggregation(
        {
            "type": "week",
            "week_id": "2026-W23",
            "start_date": "2026-05-31",
            "end_date": "2026-06-06",
            "days_included": ["2026-06-01"],
            "missing_days": [],
            "week_essence": [],
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
        }
    )

    assert weekly.short_summary == ""


def test_validate_weekly_aggregation_derives_short_summary_from_week_essence():
    weekly = validate_weekly_aggregation(
        {
            "type": "week",
            "week_id": "2026-W23",
            "start_date": "2026-05-31",
            "end_date": "2026-06-06",
            "days_included": ["2026-06-01"],
            "missing_days": [],
            "week_essence": [
                {
                    "summary": "Неделя про проект и отдых.",
                    "evidence": [
                        {
                            "date": "2026-06-01",
                            "source_day_md": "data/processed/days_md/2026-06-01.md",
                            "source_daily_json": "data/processed/days_json/2026-06-01.json",
                            "source_field": "study_signals",
                            "quote": "проект",
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
        }
    )

    assert weekly.short_summary == "Неделя про проект и отдых."


def test_validate_weekly_aggregation_rejects_null_source_daily_json():
    with pytest.raises(WeeklyAggregationError, match="LLM JSON failed validation"):
        validate_weekly_aggregation(
            {
                "type": "week",
                "week_id": "2026-W22",
                "start_date": "2026-05-25",
                "end_date": "2026-05-31",
                "days_included": ["2026-05-26"],
                "missing_days": [],
                "week_essence": [
                    {
                        "summary": "Учёба.",
                        "evidence": [
                            {
                                "date": "2026-05-26",
                                "source_day_md": "data/processed/days_md/2026-05-26.md",
                                "source_daily_json": None,
                                "source_field": "study_signals",
                                "quote": "поделал лабы",
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
                "short_summary": "Короткое резюме.",
            }
        )


def test_format_missing_days_warning():
    warning = format_missing_days_warning(["2026-05-30", "2026-05-31"])
    assert "partial week aggregation" in warning
    assert "2026-05-30" in warning
    assert format_missing_days_warning([]) == ""


@patch("diarylens.cli.aggregate_week")
def test_aggregate_week_cli_success(mock_aggregate, capsys):
    from diarylens.cli import _run_aggregate_week
    from diarylens.weekly_aggregator import AggregateWeekResult

    output_path = Path("data/processed/weeks_json/2026-W22.json")
    mock_aggregate.return_value = AggregateWeekResult(
        week_id="2026-W22",
        output_path=output_path,
        days_included=["2026-05-25", "2026-05-26"],
        missing_days=["2026-05-30"],
    )

    _run_aggregate_week("2026-W22")
    captured = capsys.readouterr()

    assert "Weekly aggregation completed." in captured.out
    assert "Days included: 2" in captured.out
    assert "Missing days: 1" in captured.out
    assert str(output_path) in captured.out
    assert "partial week aggregation" in captured.err
    assert "2026-05-30" in captured.err


@patch("diarylens.weekly_aggregator.generate_text")
def test_aggregate_week_cli_does_not_leak_auth_key(mock_generate_text, monkeypatch, capsys):
    from diarylens.cli import _run_aggregate_week

    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "super-secret-auth-key")
    mock_generate_text.return_value = json.dumps(MOCK_WEEKLY_JSON, ensure_ascii=False)

    with patch(
        "diarylens.weekly_aggregator.aggregate_week",
        return_value=__import__(
            "diarylens.weekly_aggregator", fromlist=["AggregateWeekResult"]
        ).AggregateWeekResult(
            week_id="2026-W22",
            output_path=Path("data/processed/weeks_json/2026-W22.json"),
            days_included=["2026-05-25"],
            missing_days=[],
        ),
    ):
        _run_aggregate_week("2026-W22")

    captured = capsys.readouterr()
    assert "super-secret-auth-key" not in captured.out
    assert "super-secret-auth-key" not in captured.err
