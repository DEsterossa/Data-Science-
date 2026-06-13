"""Split clean weekly markdown into daily markdown files."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from diarylens.config import (
    DAY_MANIFESTS_DIR,
    DAYS_MD_DIR,
    INTERIM_CLEAN_TEXT_DIR,
    PROJECT_ROOT,
)


class DaySplitError(Exception):
    """Raised when day splitting fails."""


RUSSIAN_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

MONTH_ALIASES: dict[str, int] = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
    "янв.": 1,
    "фев.": 2,
    "февр.": 2,
    "мар.": 3,
    "апр.": 4,
    "май": 5,
    "июн.": 6,
    "июл.": 7,
    "авг.": 8,
    "сент.": 9,
    "окт.": 10,
    "нояб.": 11,
    "дек.": 12,
}

MONTH_TO_NUMBER = {name: MONTH_ALIASES[name] for name in RUSSIAN_MONTHS}

WEEKDAYS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


def _month_pattern() -> str:
    tokens = sorted(MONTH_ALIASES.keys(), key=len, reverse=True)
    return "|".join(re.escape(token) for token in tokens)


DATE_LINE_RE = re.compile(
    rf"^(\d{{1,2}})\s+({_month_pattern()})\s+(\d{{4}})\s+г\.\s+({'|'.join(WEEKDAYS)})\s*$"
)


@dataclass(frozen=True)
class DaySection:
    iso_date: str
    original_header: str
    weekday: str
    source_order: int
    body_lines: list[str]


@dataclass(frozen=True)
class SplitDaysResult:
    days: list[DaySection]


def _validate_week_id(week_id: str | None) -> str:
    if not week_id or not week_id.strip():
        raise DaySplitError("week_id is required")
    return week_id.strip()


def clean_text_input_path(week_id: str, input_dir: Path | None = None) -> Path:
    base = input_dir or INTERIM_CLEAN_TEXT_DIR
    return base / f"{week_id}_clean.md"


def day_output_path(iso_date: str, output_dir: Path | None = None) -> Path:
    base = output_dir or DAYS_MD_DIR
    return base / f"{iso_date}.md"


def preamble_output_path(week_id: str, output_dir: Path | None = None) -> Path:
    base = output_dir or DAYS_MD_DIR
    return base / f"{week_id}__preamble.md"


def manifest_output_path(week_id: str, output_dir: Path | None = None) -> Path:
    base = output_dir or DAY_MANIFESTS_DIR
    return base / f"{week_id}_days.json"


def _relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_date_header(line: str) -> tuple[str, str, str] | None:
    """Parse a diary date line into ISO date, original header, and weekday."""
    match = DATE_LINE_RE.match(line.strip())
    if not match:
        return None

    day, month_name, year, weekday = match.groups()
    month = MONTH_ALIASES[month_name]
    iso_date = f"{year}-{month:02d}-{int(day):02d}"
    return iso_date, line.strip(), weekday


def split_days_content(clean_content: str) -> SplitDaysResult:
    """Split clean markdown content into day sections."""
    if not clean_content.strip():
        raise DaySplitError("Clean markdown is empty")

    lines = clean_content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    date_indices: list[tuple[int, str, str, str]] = []

    for index, line in enumerate(lines):
        parsed = parse_date_header(line)
        if parsed:
            iso_date, original_header, weekday = parsed
            date_indices.append((index, iso_date, original_header, weekday))

    if not date_indices:
        raise DaySplitError("No diary dates found in clean markdown")

    days: list[DaySection] = []

    for idx, (line_index, iso_date, original_header, weekday) in enumerate(
        date_indices
    ):
        next_index = (
            date_indices[idx + 1][0] if idx + 1 < len(date_indices) else len(lines)
        )
        body_lines = lines[line_index + 1 : next_index]
        days.append(
            DaySection(
                iso_date=iso_date,
                original_header=original_header,
                weekday=weekday,
                source_order=idx + 1,
                body_lines=body_lines,
            )
        )

    return SplitDaysResult(days=days)


def _format_day_markdown(
    iso_date: str,
    original_header: str,
    week_id: str,
    body_lines: list[str],
) -> str:
    body = "\n".join(body_lines).rstrip()
    parts = [
        f"# Diary day: {iso_date}",
        "",
        f"Original header: {original_header}",
        f"Week ID: {week_id}",
        "",
    ]
    if body:
        parts.append(body)
    return "\n".join(parts) + "\n"


def _remove_stale_preamble_file(week_id: str, output_dir: Path) -> None:
    stale_preamble = preamble_output_path(week_id, output_dir)
    if stale_preamble.exists():
        stale_preamble.unlink()


def build_day_manifest(
    week_id: str,
    clean_path: Path,
    result: SplitDaysResult,
    *,
    days_output_dir: Path,
) -> dict:
    """Build manifest dict with days sorted by date ascending."""
    days = [
        {
            "date": day.iso_date,
            "original_header": day.original_header,
            "weekday": day.weekday,
            "path": _relative_project_path(
                day_output_path(day.iso_date, days_output_dir)
            ),
            "source_order": day.source_order,
        }
        for day in result.days
    ]
    days.sort(key=lambda entry: entry["date"])

    return {
        "week_id": week_id,
        "source_clean_path": _relative_project_path(clean_path),
        "days_count": len(result.days),
        "days": days,
    }


def split_clean_markdown(
    week_id: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    manifest_dir: Path | None = None,
) -> list[Path]:
    """Read clean markdown for a week and write daily markdown files."""
    week_id = _validate_week_id(week_id)
    clean_path = clean_text_input_path(week_id, input_dir)

    if not clean_path.exists():
        raise DaySplitError(f"Clean markdown not found: {clean_path}")

    clean_content = clean_path.read_text(encoding="utf-8")
    result = split_days_content(clean_content)

    out_dir = output_dir or DAYS_MD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_preamble_file(week_id, out_dir)

    created_paths: list[Path] = []
    for day in result.days:
        day_path = day_output_path(day.iso_date, out_dir)
        day_path.write_text(
            _format_day_markdown(
                day.iso_date,
                day.original_header,
                week_id,
                day.body_lines,
            ),
            encoding="utf-8",
        )
        created_paths.append(day_path)

    manifest = build_day_manifest(
        week_id,
        clean_path,
        result,
        days_output_dir=out_dir,
    )
    manifest_out_dir = manifest_dir or DAY_MANIFESTS_DIR
    manifest_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_output_path(week_id, manifest_out_dir)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    created_paths.append(manifest_path)

    return created_paths
