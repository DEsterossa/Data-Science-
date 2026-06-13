import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from diarylens.schemas import (
    DailyExtraction,
    ExtractedItem,
    WeeklyAggregation,
    WeeklyEvidence,
    WeeklyItem,
    load_json,
    save_json,
)


def test_extracted_item_with_quote_and_note():
    item = ExtractedItem(quote="съел удон в обед", note="еда")

    assert item.quote == "съел удон в обед"
    assert item.note == "еда"


def test_extracted_item_note_optional():
    item = ExtractedItem(quote="пошёл на турники вечером")

    assert item.note is None


def test_daily_extraction_minimal_fields():
    daily = DailyExtraction(
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        short_summary="Короткий день.",
    )

    assert daily.date == "2026-05-26"
    assert daily.week_id == "2026-W22"
    assert daily.short_summary == "Короткий день."


def test_daily_extraction_default_lists_are_empty():
    daily = DailyExtraction(
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        short_summary="Короткий день.",
    )

    assert daily.important_moments == []
    assert daily.wins == []
    assert daily.tensions == []
    assert daily.emotions == []
    assert daily.body_energy_signals == []
    assert daily.study_signals == []
    assert daily.ml_ds_signals == []
    assert daily.social_signals == []
    assert daily.decisions == []
    assert daily.open_questions == []
    assert daily.key_quotes == []


def test_daily_extraction_rejects_old_string_list_format():
    with pytest.raises(ValidationError):
        DailyExtraction.model_validate(
            {
                "date": "2026-05-26",
                "week_id": "2026-W22",
                "source_day_md": "data/processed/days_md/2026-05-26.md",
                "important_moments": ["Погулял с друзьями"],
                "short_summary": "День.",
            }
        )


def test_daily_extraction_rejects_removed_facts_field():
    with pytest.raises(ValidationError):
        DailyExtraction.model_validate(
            {
                "date": "2026-05-26",
                "week_id": "2026-W22",
                "source_day_md": "data/processed/days_md/2026-05-26.md",
                "facts": [{"quote": "погулял", "note": "прогулка"}],
                "short_summary": "День.",
            }
        )


def test_weekly_evidence_creates():
    evidence = WeeklyEvidence(
        date="2026-05-26",
        source_day_md="data/processed/days_md/2026-05-26.md",
        source_daily_json="data/processed/days_json/2026-05-26.json",
        source_field="tensions",
        quote="боюсь наступить не туда",
        note="страх",
    )

    assert evidence.date == "2026-05-26"
    assert evidence.source_field == "tensions"


def test_weekly_item_creates_with_summary_and_evidence():
    item = WeeklyItem(
        summary="Неделя включала учёбу и напряжение вокруг выбора.",
        evidence=[
            WeeklyEvidence(
                date="2026-05-26",
                source_day_md="data/processed/days_md/2026-05-26.md",
                source_daily_json="data/processed/days_json/2026-05-26.json",
                source_field="study_signals",
                quote="сделал лабы по мбп",
            )
        ],
    )

    assert item.summary.startswith("Неделя")
    assert len(item.evidence) == 1
    assert item.evidence[0].quote == "сделал лабы по мбп"


def test_weekly_aggregation_minimal_fields():
    weekly = WeeklyAggregation(
        week_id="2026-W22",
        start_date="2026-05-25",
        end_date="2026-05-31",
        short_summary="Неделя про отдых и рефлексию.",
    )

    assert weekly.type == "week"
    assert weekly.week_id == "2026-W22"
    assert weekly.start_date == "2026-05-25"
    assert weekly.end_date == "2026-05-31"
    assert weekly.days_included == []
    assert weekly.missing_days == []
    assert weekly.main_events == []
    assert weekly.open_loops == []
    assert weekly.what_not_to_do == []


def test_weekly_aggregation_default_lists_are_empty():
    weekly = WeeklyAggregation(
        week_id="2026-W22",
        start_date="2026-05-25",
        end_date="2026-05-31",
        short_summary="Короткая неделя.",
    )

    assert weekly.week_essence == []
    assert weekly.main_wins == []
    assert weekly.main_tensions == []
    assert weekly.emotional_background == []
    assert weekly.body_energy == []
    assert weekly.study_and_projects == []
    assert weekly.social_context == []
    assert weekly.actual_focus == []
    assert weekly.repeated_topics == []
    assert weekly.important_contradictions == []
    assert weekly.risks_next_week == []
    assert weekly.next_week_focus_candidates == []


def test_weekly_evidence_accepts_current_source_field():
    evidence = WeeklyEvidence(
        date="2026-05-26",
        source_day_md="data/processed/days_md/2026-05-26.md",
        source_daily_json="data/processed/days_json/2026-05-26.json",
        source_field="emotional_signals",
        quote="скучно",
    )

    assert evidence.source_field == "emotional_signals"


@pytest.mark.parametrize(
    "legacy_field",
    ["problems", "emotions", "energy_signals", "health_signals", "open_questions"],
)
def test_weekly_evidence_rejects_legacy_source_field(legacy_field):
    with pytest.raises(ValidationError):
        WeeklyEvidence(
            date="2026-05-26",
            source_day_md="data/processed/days_md/2026-05-26.md",
            source_daily_json="data/processed/days_json/2026-05-26.json",
            source_field=legacy_field,
            quote="цитата",
        )


def test_weekly_evidence_requires_source_daily_json():
    with pytest.raises(ValidationError):
        WeeklyEvidence(
            date="2026-05-26",
            source_day_md="data/processed/days_md/2026-05-26.md",
            source_daily_json=None,
            source_field="tensions",
            quote="цитата",
        )


def test_weekly_evidence_rejects_empty_quote():
    with pytest.raises(ValidationError):
        WeeklyEvidence(
            date="2026-05-26",
            source_day_md="data/processed/days_md/2026-05-26.md",
            source_daily_json="data/processed/days_json/2026-05-26.json",
            source_field="tensions",
            quote="",
        )


def test_weekly_item_requires_evidence():
    with pytest.raises(ValidationError):
        WeeklyItem(summary="Без evidence", evidence=[])


def test_weekly_aggregation_save_and_load_roundtrip(tmp_path):
    weekly = WeeklyAggregation(
        week_id="2026-W22",
        start_date="2026-05-25",
        end_date="2026-05-31",
        days_included=["2026-05-25", "2026-05-26"],
        main_tensions=[
            WeeklyItem(
                summary="Повторялось напряжение вокруг выбора пути.",
                evidence=[
                    WeeklyEvidence(
                        date="2026-05-26",
                        source_day_md="data/processed/days_md/2026-05-26.md",
                        source_daily_json="data/processed/days_json/2026-05-26.json",
                        source_field="tensions",
                        quote="боюсь наступить не туда",
                    )
                ],
            )
        ],
        short_summary="Неделя с учёбой и сомнениями.",
    )
    output_path = tmp_path / "2026-W22.json"
    save_json(weekly, output_path)

    loaded = load_json(output_path, WeeklyAggregation)

    assert loaded.week_id == "2026-W22"
    assert loaded.days_included == ["2026-05-25", "2026-05-26"]
    assert len(loaded.main_tensions) == 1
    assert loaded.main_tensions[0].summary.startswith("Повторялось")
    assert loaded.main_tensions[0].evidence[0].quote == "боюсь наступить не туда"


def test_weekly_aggregation_save_preserves_cyrillic(tmp_path):
    weekly = WeeklyAggregation(
        week_id="2026-W22",
        start_date="2026-05-25",
        end_date="2026-05-31",
        short_summary="Здоровья, здоровья, здоровья и силы в момент выбора!",
    )
    output_path = tmp_path / "weekly_cyrillic.json"
    save_json(weekly, output_path)

    raw_text = output_path.read_text(encoding="utf-8")
    assert "Здоровья" in raw_text
    assert "\\u" not in raw_text

    parsed = json.loads(raw_text)
    assert parsed["short_summary"] == "Здоровья, здоровья, здоровья и силы в момент выбора!"


def test_save_json_writes_file(tmp_path):
    daily = DailyExtraction(
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        short_summary="День с прогулкой.",
    )
    output_path = tmp_path / "nested" / "2026-05-26.json"

    saved_path = save_json(daily, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.parent.exists()


def test_load_json_reads_and_validates_file(tmp_path):
    daily = DailyExtraction(
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        important_moments=[ExtractedItem(quote="погулял с друзьями", note="прогулка")],
        key_quotes=[ExtractedItem(quote="Я просто хочу отдохнуть", note="важная мысль")],
        short_summary="Хороший день.",
    )
    output_path = tmp_path / "2026-05-26.json"
    save_json(daily, output_path)

    loaded = load_json(output_path, DailyExtraction)

    assert loaded.date == "2026-05-26"
    assert len(loaded.important_moments) == 1
    assert loaded.important_moments[0].quote == "погулял с друзьями"
    assert len(loaded.key_quotes) == 1
    assert loaded.key_quotes[0].quote == "Я просто хочу отдохнуть"


def test_save_json_preserves_cyrillic_without_ascii_escapes(tmp_path):
    daily = DailyExtraction(
        date="2026-05-26",
        week_id="2026-W22",
        source_day_md="data/processed/days_md/2026-05-26.md",
        short_summary="Здоровья, здоровья, здоровья и силы в момент выбора!",
    )
    output_path = tmp_path / "cyrillic.json"
    save_json(daily, output_path)

    raw_text = output_path.read_text(encoding="utf-8")
    assert "Здоровья" in raw_text
    assert "\\u" not in raw_text

    parsed = json.loads(raw_text)
    assert parsed["short_summary"] == "Здоровья, здоровья, здоровья и силы в момент выбора!"
