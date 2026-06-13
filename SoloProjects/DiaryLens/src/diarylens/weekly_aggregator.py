"""Aggregate daily JSON files into weekly JSON using an LLM."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from diarylens.config import (
    DAY_MANIFESTS_DIR,
    DAYS_JSON_DIR,
    PROMPTS_DIR,
    WEEKS_JSON_DIR,
)
from diarylens.daily_extractor import DailyExtractionError, load_week_manifest
from diarylens.json_utils import LLMJsonParseError, parse_llm_json_response
from diarylens.llm_client import LLMError, generate_text, resolve_model_for_kind
from diarylens.schemas import DailyExtraction, WeeklyAggregation, load_json, save_json


class WeeklyAggregationError(Exception):
    """Raised when weekly aggregation fails."""


WEEKLY_AGGREGATION_PROMPT_PATH = PROMPTS_DIR / "weekly_aggregation.md"

SOURCE_FIELD_ALIASES: dict[str, str] = {
    "problems": "tensions",
    "emotions": "emotional_signals",
    "energy_signals": "body_energy_signals",
    "health_signals": "body_energy_signals",
    "open_questions": "open_loops",
}

WEEKLY_ITEM_LIST_FIELDS = (
    "week_essence",
    "main_events",
    "main_wins",
    "main_tensions",
    "emotional_background",
    "body_energy",
    "study_and_projects",
    "social_context",
    "actual_focus",
    "repeated_topics",
    "important_contradictions",
    "open_loops",
    "risks_next_week",
    "next_week_focus_candidates",
    "what_not_to_do",
)


@dataclass(frozen=True)
class AggregateWeekResult:
    week_id: str
    output_path: Path
    days_included: list[str]
    missing_days: list[str]


def _validate_week_id(week_id: str | None) -> str:
    if not week_id or not week_id.strip():
        raise WeeklyAggregationError("week_id is required")
    return week_id.strip()


def week_json_output_path(week_id: str, output_dir: Path | None = None) -> Path:
    base = output_dir or WEEKS_JSON_DIR
    return base / f"{week_id}.json"


def day_json_input_path(date: str, input_dir: Path | None = None) -> Path:
    base = input_dir or DAYS_JSON_DIR
    return base / f"{date}.json"


def daily_json_relative_path(date: str) -> str:
    """Return the canonical relative path for a daily JSON file."""
    return f"data/processed/days_json/{date}.json"


def build_daily_jsons_payload(
    daily_extractions: list[DailyExtraction],
) -> list[dict]:
    """Build the daily JSON wrapper payload for the weekly aggregation prompt."""
    return [
        {
            "date": daily.date,
            "source_daily_json": daily_json_relative_path(daily.date),
            "daily": daily.model_dump(),
        }
        for daily in daily_extractions
    ]


def load_weekly_prompt_template(prompt_path: Path | None = None) -> str:
    path = prompt_path or WEEKLY_AGGREGATION_PROMPT_PATH
    if not path.exists():
        raise WeeklyAggregationError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_weekly_prompt(
    template: str,
    *,
    week_id: str,
    start_date: str,
    end_date: str,
    days_included: list[str],
    missing_days: list[str],
    daily_jsons: str,
) -> str:
    """Fill weekly aggregation prompt placeholders."""
    return (
        template.replace("{week_id}", week_id)
        .replace("{start_date}", start_date)
        .replace("{end_date}", end_date)
        .replace("{days_included}", json.dumps(days_included, ensure_ascii=False))
        .replace("{missing_days}", json.dumps(missing_days, ensure_ascii=False))
        .replace("{daily_jsons}", daily_jsons)
    )


def collect_daily_jsons_for_week(
    manifest: dict,
    *,
    days_json_dir: Path | None = None,
) -> tuple[list[DailyExtraction], list[str], list[str]]:
    """Load available daily JSON files listed in a week manifest."""
    days = manifest.get("days", [])
    if not days:
        raise WeeklyAggregationError("Day manifest contains no days")

    daily_extractions: list[DailyExtraction] = []
    days_included: list[str] = []
    missing_days: list[str] = []

    for day_entry in days:
        date = day_entry["date"]
        json_path = day_json_input_path(date, days_json_dir)
        if not json_path.exists():
            missing_days.append(date)
            continue

        try:
            daily = load_json(json_path, DailyExtraction)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise WeeklyAggregationError(
                f"Invalid daily JSON for {date}: {json_path}"
            ) from exc

        daily_extractions.append(daily)
        days_included.append(date)

    return daily_extractions, days_included, missing_days


def normalize_source_field(source_field: object) -> str | None:
    """Map legacy daily field names to canonical WeeklySourceField values."""
    if not isinstance(source_field, str) or not source_field.strip():
        return None
    canonical = source_field.strip()
    return SOURCE_FIELD_ALIASES.get(canonical, canonical)


def _normalize_weekly_evidence(evidence: object) -> dict | None:
    if not isinstance(evidence, dict):
        return None
    quote = evidence.get("quote")
    if quote is None or not str(quote).strip():
        return None
    source_field = normalize_source_field(evidence.get("source_field"))
    if source_field is None:
        return None
    normalized = dict(evidence)
    normalized["source_field"] = source_field
    normalized["quote"] = str(quote).strip()
    return normalized


def _normalize_weekly_item(item: object) -> dict | None:
    if isinstance(item, str):
        summary = item
        evidence_raw: list = []
    elif isinstance(item, dict):
        summary = item.get("summary")
        if summary is None:
            summary = item.get("text", "")
        evidence_raw = item.get("evidence", [])
        if not isinstance(evidence_raw, list):
            evidence_raw = []
    else:
        summary = str(item)
        evidence_raw = []

    evidence = [
        normalized
        for ev in evidence_raw
        if (normalized := _normalize_weekly_evidence(ev)) is not None
    ]
    if not str(summary).strip() or not evidence:
        return None
    return {"summary": str(summary), "evidence": evidence}


def _normalize_short_summary(normalized: dict) -> None:
    """Ensure required short_summary exists after common LLM omissions."""
    value = normalized.get("short_summary")
    if value is not None and str(value).strip():
        normalized["short_summary"] = str(value).strip()
        return

    for key in ("week_summary", "summary", "weekly_summary"):
        alt = normalized.get(key)
        if alt is not None and str(alt).strip():
            normalized["short_summary"] = str(alt).strip()
            return

    week_essence = normalized.get("week_essence", [])
    if isinstance(week_essence, list):
        for item in week_essence:
            if isinstance(item, dict):
                summary = item.get("summary")
                if summary and str(summary).strip():
                    normalized["short_summary"] = str(summary).strip()
                    return

    normalized["short_summary"] = ""


def normalize_weekly_aggregation_data(data: dict) -> dict:
    """Repair common LLM shape mistakes before schema validation."""
    normalized = dict(data)
    for field in WEEKLY_ITEM_LIST_FIELDS:
        raw_items = normalized.get(field, [])
        if raw_items is None:
            normalized[field] = []
        elif isinstance(raw_items, list):
            items = [_normalize_weekly_item(item) for item in raw_items]
            normalized[field] = [item for item in items if item is not None]
        else:
            normalized[field] = []
    _normalize_short_summary(normalized)
    return normalized


def validate_weekly_aggregation(data: dict) -> WeeklyAggregation:
    """Validate parsed LLM JSON against the WeeklyAggregation schema."""
    normalized = normalize_weekly_aggregation_data(data)
    try:
        return WeeklyAggregation.model_validate(normalized)
    except ValidationError as exc:
        raise WeeklyAggregationError(f"LLM JSON failed validation: {exc}") from exc


def aggregate_week(
    week_id: str,
    *,
    manifest_dir: Path | None = None,
    days_json_dir: Path | None = None,
    output_dir: Path | None = None,
    prompt_path: Path | None = None,
) -> AggregateWeekResult:
    """Aggregate daily JSON files for a week into weekly JSON."""
    week_id = _validate_week_id(week_id)
    try:
        manifest = load_week_manifest(week_id, manifest_dir=manifest_dir)
    except DailyExtractionError as exc:
        raise WeeklyAggregationError(str(exc)) from exc

    daily_extractions, days_included, missing_days = collect_daily_jsons_for_week(
        manifest,
        days_json_dir=days_json_dir,
    )
    if not daily_extractions:
        raise WeeklyAggregationError(
            f"no valid daily JSON files found for week {week_id}"
        )

    manifest_dates = sorted(day_entry["date"] for day_entry in manifest.get("days", []))
    if not manifest_dates:
        raise WeeklyAggregationError(f"No days found in manifest for week: {week_id}")

    start_date = manifest_dates[0]
    end_date = manifest_dates[-1]
    daily_jsons_payload = build_daily_jsons_payload(daily_extractions)
    daily_jsons = json.dumps(daily_jsons_payload, indent=2, ensure_ascii=False)

    prompt_template = load_weekly_prompt_template(prompt_path)
    prompt = render_weekly_prompt(
        prompt_template,
        week_id=week_id,
        start_date=start_date,
        end_date=end_date,
        days_included=days_included,
        missing_days=missing_days,
        daily_jsons=daily_jsons,
    )

    model = resolve_model_for_kind("weekly")
    try:
        raw_response = generate_text(prompt, model=model)
        data = parse_llm_json_response(
            raw_response,
            refusal_message="GigaChat refused to process weekly aggregation",
        )
        weekly = validate_weekly_aggregation(data)
    except (LLMJsonParseError, LLMError) as exc:
        raise WeeklyAggregationError(str(exc)) from exc

    out_dir = output_dir or WEEKS_JSON_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = week_json_output_path(week_id, out_dir)
    save_json(weekly, output_path)

    return AggregateWeekResult(
        week_id=week_id,
        output_path=output_path,
        days_included=days_included,
        missing_days=missing_days,
    )


def format_missing_days_warning(missing_days: list[str]) -> str:
    """Format a partial-week warning for CLI output."""
    if not missing_days:
        return ""
    missing_list = ", ".join(missing_days)
    return f"Warning: partial week aggregation. Missing daily JSON for: {missing_list}"


def format_aggregate_week_summary(result: AggregateWeekResult) -> str:
    """Format CLI summary for weekly aggregation."""
    return "\n".join(
        [
            "Weekly aggregation completed.",
            f"Week ID: {result.week_id}",
            f"Days included: {len(result.days_included)}",
            f"Missing days: {len(result.missing_days)}",
            f"Output: {result.output_path}",
        ]
    )
