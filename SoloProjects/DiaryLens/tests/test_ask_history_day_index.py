import json
from pathlib import Path

from diarylens.ask_history.day_index import (
    build_day_embedding_text,
    build_day_index,
    load_day_index_csv,
    resolve_source_day_md,
    save_day_index_csv,
)
from diarylens.ask_history.models import AskHistoryConfig


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_build_day_embedding_text_collects_summary_and_quote_note():
    text = build_day_embedding_text(
        {
            "date": "2026-06-12",
            "week_id": "2026-W24",
            "short_summary": "День с прогулкой и размышлениями.",
            "important_moments": [
                {"quote": "Много думаю о себе.", "note": "рефлексия"}
            ],
            "tensions": ["Скучные моменты тоже важны."],
        }
    )

    assert "date: 2026-06-12" in text
    assert "week_id: 2026-W24" in text
    assert "short_summary:" in text
    assert "День с прогулкой" in text
    assert "quote: Много думаю о себе. | note: рефлексия" in text
    assert "Скучные моменты тоже важны." in text


def test_empty_daily_json_uses_raw_fallback(tmp_path):
    days_json_dir = tmp_path / "days_json"
    days_md_dir = tmp_path / "days_md"
    days_md_dir.mkdir()
    (days_md_dir / "2026-06-11.md").write_text(
        "# Diary day: 2026-06-11\n\nСырой текст дня про скуку и прогулку.",
        encoding="utf-8",
    )
    _write_json(
        days_json_dir / "2026-06-11.json",
        {
            "date": "2026-06-11",
            "week_id": "2026-W24",
            "source_day_md": "data/processed/days_md/2026-06-11.md",
            "short_summary": "",
        },
    )

    records = build_day_index(days_json_dir, days_md_dir, AskHistoryConfig())

    assert records[0].index_source == "raw_fallback"
    assert "Сырой текст дня про скуку" in records[0].embedding_text


def test_resolve_source_day_md_prefers_daily_json_value():
    daily_json_path = Path("data/processed/days_json/2026-06-12.json")
    path = resolve_source_day_md(
        {"source_day_md": "data/processed/days_md/2026-06-12.md"},
        daily_json_path,
    )

    assert path.as_posix().endswith("data/processed/days_md/2026-06-12.md")


def test_save_and_load_day_index_csv_preserves_cyrillic(tmp_path):
    days_json_dir = tmp_path / "days_json"
    days_md_dir = tmp_path / "days_md"
    days_md_dir.mkdir()
    (days_md_dir / "2026-06-12.md").write_text("Сырой день", encoding="utf-8")
    _write_json(
        days_json_dir / "2026-06-12.json",
        {
            "date": "2026-06-12",
            "week_id": "2026-W24",
            "source_day_md": "data/processed/days_md/2026-06-12.md",
            "important_moments": [{"quote": "Прогулка", "note": "город"}],
            "short_summary": "Кириллица сохраняется.",
        },
    )
    records = build_day_index(
        days_json_dir,
        days_md_dir,
        AskHistoryConfig(min_day_index_text_len=10),
    )
    path = tmp_path / "memory" / "day_index.csv"

    save_day_index_csv(records, path)
    loaded = load_day_index_csv(path)

    assert loaded[0].embedding_text == records[0].embedding_text
    assert "Кириллица" in loaded[0].embedding_text


def test_embedding_text_hash_changes_when_text_changes(tmp_path):
    days_json_dir = tmp_path / "days_json"
    days_md_dir = tmp_path / "days_md"
    days_md_dir.mkdir()
    (days_md_dir / "2026-06-12.md").write_text("Сырой день", encoding="utf-8")
    path = days_json_dir / "2026-06-12.json"
    payload = {
        "date": "2026-06-12",
        "week_id": "2026-W24",
        "source_day_md": "data/processed/days_md/2026-06-12.md",
        "important_moments": [{"quote": "Первый текст", "note": "один"}],
        "short_summary": "День.",
    }
    _write_json(path, payload)

    first = build_day_index(
        days_json_dir,
        days_md_dir,
        AskHistoryConfig(min_day_index_text_len=10),
    )[0]
    payload["important_moments"][0]["quote"] = "Второй текст"
    _write_json(path, payload)
    second = build_day_index(
        days_json_dir,
        days_md_dir,
        AskHistoryConfig(min_day_index_text_len=10),
    )[0]

    assert first.embedding_text_hash != second.embedding_text_hash
