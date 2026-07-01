"""Build and persist the day-level semantic index."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from diarylens import config as project_config
from diarylens.ask_history.config import TEXT_FIELDS_FOR_DAY_INDEX
from diarylens.ask_history.models import AskHistoryConfig, AskHistoryError, DayIndexRecord

DAY_INDEX_COLUMNS = [
    "date",
    "week_id",
    "source_day_md",
    "source_daily_json",
    "embedding_text",
    "embedding_text_len",
    "embedding_text_hash",
    "index_source",
]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_relative_source(path: Path) -> Path:
    if path.is_absolute():
        return path

    project_candidate = project_config.PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    return path


def resolve_source_day_md(daily_json: dict, daily_json_path: Path) -> Path:
    """Resolve the raw day markdown path named by a daily JSON payload."""
    source_day_md = daily_json.get("source_day_md")
    if isinstance(source_day_md, str) and source_day_md.strip():
        return _resolve_relative_source(Path(source_day_md.strip()))

    date = daily_json.get("date") or daily_json_path.stem
    return Path(f"{date}.md")


def load_daily_records(days_json_dir: Path) -> list[tuple[Path, dict]]:
    """Load all daily JSON files in sorted date order."""
    if not days_json_dir.exists():
        return []

    records: list[tuple[Path, dict]] = []
    for path in sorted(days_json_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AskHistoryError(f"invalid daily JSON: {_display_path(path)}") from exc
        if isinstance(data, dict):
            records.append((path, data))
    return records


def _append_value(lines: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            lines.append(value.strip())
        return
    if isinstance(value, dict):
        quote = value.get("quote")
        note = value.get("note")
        parts = []
        if isinstance(quote, str) and quote.strip():
            parts.append(f"quote: {quote.strip()}")
        if isinstance(note, str) and note.strip():
            parts.append(f"note: {note.strip()}")
        if parts:
            lines.append(" | ".join(parts))
        return
    if isinstance(value, list):
        for item in value:
            _append_value(lines, item)


def build_day_embedding_text(daily_json: dict) -> str:
    """Build compact searchable text from a daily JSON artifact."""
    lines: list[str] = []
    date = daily_json.get("date")
    week_id = daily_json.get("week_id")
    if isinstance(date, str) and date.strip():
        lines.append(f"date: {date.strip()}")
    if isinstance(week_id, str) and week_id.strip():
        lines.append(f"week_id: {week_id.strip()}")

    short_summary = daily_json.get("short_summary")
    if isinstance(short_summary, str) and short_summary.strip():
        lines.extend(["", "short_summary:", short_summary.strip()])

    for field_name in TEXT_FIELDS_FOR_DAY_INDEX:
        value = daily_json.get(field_name)
        field_lines: list[str] = []
        _append_value(field_lines, value)
        if field_lines:
            lines.extend(["", f"{field_name}:"])
            lines.extend(field_lines)

    return "\n".join(lines).strip()


def read_day_raw_text(source_day_md: Path) -> str:
    """Read source markdown for a diary day."""
    if not source_day_md.exists():
        raise AskHistoryError(
            f"source day markdown not found: {_display_path(source_day_md)}"
        )
    return source_day_md.read_text(encoding="utf-8")


def build_fallback_embedding_text(
    date: str,
    week_id: str | None,
    raw_text: str,
    raw_fallback_chars: int,
) -> str:
    """Build searchable text from raw markdown when daily JSON is sparse."""
    lines = [f"date: {date}"]
    if week_id:
        lines.append(f"week_id: {week_id}")
    lines.extend(["", raw_text[:raw_fallback_chars].strip()])
    return "\n".join(line for line in lines if line is not None).strip()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_day_md_for_index(
    source_day_md: Path,
    days_md_dir: Path,
    date: str,
) -> Path:
    if source_day_md.name:
        candidate = days_md_dir / source_day_md.name
        if candidate.exists():
            return candidate

    candidate = days_md_dir / f"{date}.md"
    if candidate.exists():
        return candidate

    if source_day_md.is_absolute() and source_day_md.exists():
        return source_day_md

    if source_day_md.exists():
        return source_day_md

    return source_day_md


def build_day_index(
    days_json_dir: Path,
    days_md_dir: Path,
    config: AskHistoryConfig,
) -> list[DayIndexRecord]:
    """Create day index records from daily JSON and raw day markdown."""
    daily_records = load_daily_records(days_json_dir)
    if not daily_records:
        raise AskHistoryError(
            f"no daily JSON files found in {_display_path(days_json_dir)}. "
            "Run `diarylens run-week ...` first."
        )

    index_records: list[DayIndexRecord] = []
    for daily_json_path, daily_json in daily_records:
        date = str(daily_json.get("date") or daily_json_path.stem)
        raw_week_id = daily_json.get("week_id")
        week_id = str(raw_week_id) if raw_week_id else None
        source_day_md = _resolve_day_md_for_index(
            resolve_source_day_md(daily_json, daily_json_path),
            days_md_dir,
            date,
        )

        embedding_text = build_day_embedding_text(daily_json)
        index_source = "daily_json"
        if len(embedding_text) < config.min_day_index_text_len:
            raw_text = read_day_raw_text(source_day_md)
            embedding_text = build_fallback_embedding_text(
                date,
                week_id,
                raw_text,
                config.raw_fallback_chars,
            )
            index_source = "raw_fallback"

        index_records.append(
            DayIndexRecord(
                date=date,
                week_id=week_id,
                source_day_md=_display_path(source_day_md),
                source_daily_json=_display_path(daily_json_path),
                embedding_text=embedding_text,
                embedding_text_len=len(embedding_text),
                embedding_text_hash=_hash_text(embedding_text),
                index_source=index_source,
            )
        )

    return index_records


def save_day_index_csv(records: list[DayIndexRecord], path: Path) -> None:
    """Persist day index records as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DAY_INDEX_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())


def load_day_index_csv(path: Path) -> list[DayIndexRecord]:
    """Load day index records from CSV."""
    if not path.exists():
        raise AskHistoryError(f"day index not found: {_display_path(path)}")

    records: list[DayIndexRecord] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["embedding_text_len"] = int(row["embedding_text_len"])
            row["week_id"] = row.get("week_id") or None
            records.append(DayIndexRecord.model_validate(row))
    return records
