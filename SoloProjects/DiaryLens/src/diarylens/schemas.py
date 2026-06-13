"""Pydantic schemas for DiaryLens structured JSON data."""

import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class ExtractedItem(BaseModel):
    """Quote-based fragment extracted from a diary day."""

    quote: str
    note: str | None = None


class EvidenceItem(BaseModel):
    """Short quote or evidence fragment from a diary entry."""

    date: str | None = None
    source_file: str | None = None
    quote: str
    note: str | None = None


class DailyExtraction(BaseModel):
    """Structured extraction for a single diary day."""

    model_config = ConfigDict(extra="forbid")

    date: str
    week_id: str
    source_day_md: str

    important_moments: list[ExtractedItem] = Field(default_factory=list)
    wins: list[ExtractedItem] = Field(default_factory=list)
    tensions: list[ExtractedItem] = Field(default_factory=list)
    emotions: list[ExtractedItem] = Field(default_factory=list)
    body_energy_signals: list[ExtractedItem] = Field(default_factory=list)
    study_signals: list[ExtractedItem] = Field(default_factory=list)
    ml_ds_signals: list[ExtractedItem] = Field(default_factory=list)
    social_signals: list[ExtractedItem] = Field(default_factory=list)
    decisions: list[ExtractedItem] = Field(default_factory=list)
    open_questions: list[ExtractedItem] = Field(default_factory=list)
    key_quotes: list[ExtractedItem] = Field(default_factory=list)

    short_summary: str


WeeklySourceField = Literal[
    "important_moments",
    "wins",
    "tensions",
    "emotional_signals",
    "body_energy_signals",
    "study_signals",
    "ml_ds_signals",
    "social_signals",
    "decisions",
    "open_loops",
    "key_quotes",
    "short_summary",
]


class WeeklyEvidence(BaseModel):
    """Evidence fragment linked to a daily JSON entry."""

    date: str
    source_day_md: str
    source_daily_json: str
    source_field: WeeklySourceField
    quote: str = Field(min_length=1)
    note: str | None = None


class WeeklyItem(BaseModel):
    """Evidence-based weekly claim with supporting daily fragments."""

    summary: str = Field(min_length=1)
    evidence: list[WeeklyEvidence] = Field(min_length=1)


class WeeklyAggregation(BaseModel):
    """Structured evidence-based aggregation for a diary week."""

    type: Literal["week"] = "week"
    week_id: str
    start_date: str
    end_date: str

    days_included: list[str] = Field(default_factory=list)
    missing_days: list[str] = Field(default_factory=list)

    week_essence: list[WeeklyItem] = Field(default_factory=list)
    main_events: list[WeeklyItem] = Field(default_factory=list)
    main_wins: list[WeeklyItem] = Field(default_factory=list)
    main_tensions: list[WeeklyItem] = Field(default_factory=list)
    emotional_background: list[WeeklyItem] = Field(default_factory=list)
    body_energy: list[WeeklyItem] = Field(default_factory=list)
    study_and_projects: list[WeeklyItem] = Field(default_factory=list)
    social_context: list[WeeklyItem] = Field(default_factory=list)
    actual_focus: list[WeeklyItem] = Field(default_factory=list)
    repeated_topics: list[WeeklyItem] = Field(default_factory=list)
    important_contradictions: list[WeeklyItem] = Field(default_factory=list)
    open_loops: list[WeeklyItem] = Field(default_factory=list)
    risks_next_week: list[WeeklyItem] = Field(default_factory=list)
    next_week_focus_candidates: list[WeeklyItem] = Field(default_factory=list)
    what_not_to_do: list[WeeklyItem] = Field(default_factory=list)

    short_summary: str


def save_json(model: BaseModel, path: Path | str) -> Path:
    """Serialize a Pydantic model to JSON with UTF-8 and pretty indentation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_json(path: Path | str, model_class: type[T]) -> T:
    """Load and validate JSON data into a Pydantic model."""
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return model_class.model_validate(data)
