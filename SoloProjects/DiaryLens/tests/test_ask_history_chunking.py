from pathlib import Path

from diarylens.ask_history.chunking import (
    build_local_chunks_from_days,
    chunk_text_by_chars,
    normalize_raw_day_text,
)
from diarylens.ask_history.models import AskHistoryConfig, DaySearchResult


def test_normalize_raw_day_text_removes_page_markers():
    text = "a\r\n\r\n\r\n<!-- page 12 -->\r\nb"

    normalized = normalize_raw_day_text(text)

    assert "<!-- page" not in normalized
    assert "\r" not in normalized
    assert "\n\n\n" not in normalized


def test_chunking_creates_expected_sizes_and_overlap():
    chunks = chunk_text_by_chars("abcdefghij", chunk_size=5, overlap=2, min_chunk_len=1)

    assert chunks == ["abcde", "defgh", "ghij"]


def test_short_last_chunk_is_merged():
    chunks = chunk_text_by_chars("abcdefghijk", chunk_size=5, overlap=0, min_chunk_len=3)

    assert chunks == ["abcde", "fghij\nk"]


def test_empty_text_returns_no_chunks():
    assert chunk_text_by_chars("   ", chunk_size=5, overlap=1, min_chunk_len=2) == []


def test_build_local_chunks_from_days_reads_raw_markdown(tmp_path, monkeypatch):
    project_root = tmp_path
    source = project_root / "data" / "processed" / "days_md" / "2026-06-12.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Header\n\n" + "абвгд" * 80, encoding="utf-8")
    monkeypatch.setattr("diarylens.config.PROJECT_ROOT", project_root)

    chunks = build_local_chunks_from_days(
        [
            DaySearchResult(
                rank=1,
                date="2026-06-12",
                week_id="2026-W24",
                score=0.9,
                source_day_md="data/processed/days_md/2026-06-12.md",
                source_daily_json="data/processed/days_json/2026-06-12.json",
            )
        ],
        AskHistoryConfig(
            day_header_skip_chars=0,
            chunk_size=100,
            overlap=10,
            min_chunk_len=20,
        ),
    )

    assert chunks
    assert chunks[0].date == "2026-06-12"
    assert chunks[0].source_day_md.endswith("2026-06-12.md")
