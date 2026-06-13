import json
from pathlib import Path

import pytest

from diarylens.day_splitter import (
    DaySplitError,
    day_output_path,
    manifest_output_path,
    parse_date_header,
    preamble_output_path,
    split_clean_markdown,
    split_days_content,
)


SAMPLE_CLEAN = """# Clean diary text: 2026-W22

Source PDF: data/raw/weekly/week.pdf

<!-- page 1 -->

26 мая 2026 г. Вт
Tuesday entry line one
Tuesday entry line two

<!-- page 2 -->

25 мая 2026 г. Пн
Monday entry
"""


SAMPLE_CLEAN_NO_PREFIX = """26 мая 2026 г. Вт
First day

27 мая 2026 г. Ср
Second day
"""


def _setup_project_dirs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project_root = tmp_path / "project"
    clean_dir = project_root / "data" / "interim" / "clean_text"
    days_dir = project_root / "data" / "processed" / "days_md"
    manifest_dir = project_root / "data" / "processed" / "day_manifests"
    clean_dir.mkdir(parents=True)
    days_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    return project_root, clean_dir, days_dir, manifest_dir


def test_parse_date_header_converts_russian_months():
    parsed = parse_date_header("26 мая 2026 г. Вт")
    assert parsed == ("2026-05-26", "26 мая 2026 г. Вт", "Вт")

    parsed = parse_date_header("3 января 2025 г. Пн")
    assert parsed == ("2025-01-03", "3 января 2025 г. Пн", "Пн")

    assert parse_date_header("not a date") is None


def test_parse_date_header_supports_abbreviated_months():
    parsed = parse_date_header("6 июн. 2026 г. Сб")
    assert parsed == ("2026-06-06", "6 июн. 2026 г. Сб", "Сб")

    parsed = parse_date_header("1 июн. 2026 г. Пн")
    assert parsed == ("2026-06-01", "1 июн. 2026 г. Пн", "Пн")


SAMPLE_CLEAN_ABBREV_MONTHS = """# Clean diary text: 2026-W23

6 июн. 2026 г. Сб
Saturday entry

31 мая 2026 г. Вс
Sunday entry
"""


def test_split_days_supports_mixed_full_and_abbreviated_month_headers():
    result = split_days_content(SAMPLE_CLEAN_ABBREV_MONTHS)
    assert len(result.days) == 2
    dates = {day.iso_date for day in result.days}
    assert dates == {"2026-05-31", "2026-06-06"}


def test_split_days_creates_daily_markdown_for_two_dates(tmp_path, monkeypatch):
    project_root, clean_dir, days_dir, manifest_dir = _setup_project_dirs(tmp_path)
    monkeypatch.setattr("diarylens.day_splitter.PROJECT_ROOT", project_root)

    clean_file = clean_dir / "2026-W22_clean.md"
    clean_file.write_text(SAMPLE_CLEAN_NO_PREFIX, encoding="utf-8")

    created = split_clean_markdown(
        "2026-W22",
        input_dir=clean_dir,
        output_dir=days_dir,
        manifest_dir=manifest_dir,
    )

    day_26 = day_output_path("2026-05-26", days_dir)
    day_27 = day_output_path("2026-05-27", days_dir)
    manifest_path = manifest_output_path("2026-W22", manifest_dir)
    assert day_26 in created
    assert day_27 in created
    assert manifest_path in created
    assert len(created) == 3

    content_26 = day_26.read_text(encoding="utf-8")
    assert content_26.startswith("# Diary day: 2026-05-26\n")
    assert "Original header: 26 мая 2026 г. Вт" in content_26
    assert "Week ID: 2026-W22" in content_26
    assert "First day" in content_26

    content_27 = day_27.read_text(encoding="utf-8")
    assert "Second day" in content_27


def test_split_days_does_not_create_preamble_file(tmp_path, monkeypatch):
    project_root, clean_dir, days_dir, manifest_dir = _setup_project_dirs(tmp_path)
    monkeypatch.setattr("diarylens.day_splitter.PROJECT_ROOT", project_root)

    clean_file = clean_dir / "2026-W22_clean.md"
    clean_file.write_text(SAMPLE_CLEAN, encoding="utf-8")

    split_clean_markdown(
        "2026-W22",
        input_dir=clean_dir,
        output_dir=days_dir,
        manifest_dir=manifest_dir,
    )

    preamble_path = preamble_output_path("2026-W22", days_dir)
    assert not preamble_path.exists()


def test_split_days_ignores_content_before_first_date(tmp_path, monkeypatch):
    project_root, clean_dir, days_dir, manifest_dir = _setup_project_dirs(tmp_path)
    monkeypatch.setattr("diarylens.day_splitter.PROJECT_ROOT", project_root)

    clean_file = clean_dir / "2026-W22_clean.md"
    clean_file.write_text(SAMPLE_CLEAN, encoding="utf-8")

    split_clean_markdown(
        "2026-W22",
        input_dir=clean_dir,
        output_dir=days_dir,
        manifest_dir=manifest_dir,
    )

    day_26 = day_output_path("2026-05-26", days_dir).read_text(encoding="utf-8")
    assert "Source PDF" not in day_26
    assert "# Clean diary text" not in day_26
    assert "Tuesday entry line one" in day_26


def test_old_preamble_file_removed_on_rerun(tmp_path, monkeypatch):
    project_root, clean_dir, days_dir, manifest_dir = _setup_project_dirs(tmp_path)
    monkeypatch.setattr("diarylens.day_splitter.PROJECT_ROOT", project_root)

    clean_file = clean_dir / "2026-W22_clean.md"
    clean_file.write_text(SAMPLE_CLEAN_NO_PREFIX, encoding="utf-8")

    stale_preamble = preamble_output_path("2026-W22", days_dir)
    stale_preamble.write_text("# Preamble: 2026-W22\n\nold content\n", encoding="utf-8")
    assert stale_preamble.exists()

    split_clean_markdown(
        "2026-W22",
        input_dir=clean_dir,
        output_dir=days_dir,
        manifest_dir=manifest_dir,
    )

    assert not stale_preamble.exists()


def test_split_days_creates_manifest(tmp_path, monkeypatch):
    project_root, clean_dir, days_dir, manifest_dir = _setup_project_dirs(tmp_path)
    monkeypatch.setattr("diarylens.day_splitter.PROJECT_ROOT", project_root)

    clean_file = clean_dir / "2026-W22_clean.md"
    clean_file.write_text(SAMPLE_CLEAN, encoding="utf-8")

    split_clean_markdown(
        "2026-W22",
        input_dir=clean_dir,
        output_dir=days_dir,
        manifest_dir=manifest_dir,
    )

    manifest_path = manifest_output_path("2026-W22", manifest_dir)
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["week_id"] == "2026-W22"
    assert manifest["source_clean_path"] == "data/interim/clean_text/2026-W22_clean.md"
    assert manifest["days_count"] == 2
    assert "has_preamble" not in manifest
    assert "preamble_path" not in manifest
    assert len(manifest["days"]) == 2


def test_manifest_days_sorted_by_date(tmp_path, monkeypatch):
    project_root, clean_dir, days_dir, manifest_dir = _setup_project_dirs(tmp_path)
    monkeypatch.setattr("diarylens.day_splitter.PROJECT_ROOT", project_root)

    clean_file = clean_dir / "2026-W22_clean.md"
    clean_file.write_text(SAMPLE_CLEAN, encoding="utf-8")

    split_clean_markdown(
        "2026-W22",
        input_dir=clean_dir,
        output_dir=days_dir,
        manifest_dir=manifest_dir,
    )

    manifest = json.loads(
        manifest_output_path("2026-W22", manifest_dir).read_text(encoding="utf-8")
    )
    dates = [day["date"] for day in manifest["days"]]
    assert dates == ["2026-05-25", "2026-05-26"]


def test_manifest_source_order_preserves_source_appearance(tmp_path, monkeypatch):
    project_root, clean_dir, days_dir, manifest_dir = _setup_project_dirs(tmp_path)
    monkeypatch.setattr("diarylens.day_splitter.PROJECT_ROOT", project_root)

    clean_file = clean_dir / "2026-W22_clean.md"
    clean_file.write_text(SAMPLE_CLEAN, encoding="utf-8")

    split_clean_markdown(
        "2026-W22",
        input_dir=clean_dir,
        output_dir=days_dir,
        manifest_dir=manifest_dir,
    )

    manifest = json.loads(
        manifest_output_path("2026-W22", manifest_dir).read_text(encoding="utf-8")
    )
    by_date = {day["date"]: day for day in manifest["days"]}
    assert by_date["2026-05-26"]["source_order"] == 1
    assert by_date["2026-05-25"]["source_order"] == 2
    assert by_date["2026-05-26"]["weekday"] == "Вт"
    assert by_date["2026-05-25"]["weekday"] == "Пн"


def test_split_days_preserves_page_markers_inside_day():
    result = split_days_content(SAMPLE_CLEAN_NO_PREFIX)
    assert len(result.days) == 2

    clean_with_marker = (
        "26 мая 2026 г. Вт\n"
        "Line one\n"
        "\n"
        "<!-- page 2 -->\n"
        "Line two\n"
    )
    result = split_days_content(clean_with_marker)
    body = "\n".join(result.days[0].body_lines)
    assert "<!-- page 2 -->" in body
    assert "Line one" in body
    assert "Line two" in body


def test_split_days_no_dates_raises():
    with pytest.raises(DaySplitError, match="No diary dates found"):
        split_days_content("# Clean diary text\n\nNo dates here")


def test_split_days_missing_clean_file_raises(tmp_path):
    with pytest.raises(DaySplitError, match="Clean markdown not found"):
        split_clean_markdown("2026-W22", input_dir=tmp_path, output_dir=tmp_path)


def test_split_days_requires_week_id():
    with pytest.raises(DaySplitError, match="week_id is required"):
        split_clean_markdown("")
