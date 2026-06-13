"""Generate weekly markdown reports from weekly JSON using an LLM."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from diarylens.config import (
    GOALS_CONTEXT_PATH,
    PROMPTS_DIR,
    REPORTS_DIR,
    USER_CONTEXT_PATH,
    WEEKS_JSON_DIR,
)
from diarylens.json_utils import strip_markdown_fences
from diarylens.llm_client import LLMError, generate_text, resolve_report_model
from diarylens.schemas import WeeklyAggregation, load_json


class ReportGenerationError(Exception):
    """Raised when weekly report generation fails."""


WEEKLY_REPORT_PROMPT_PATH = PROMPTS_DIR / "weekly_report.md"
REPORT_TITLE_PREFIX = "# Weekly Review"


@dataclass(frozen=True)
class GenerateReportResult:
    week_id: str
    input_path: Path
    output_path: Path


def _validate_week_id(week_id: str | None) -> str:
    if not week_id or not week_id.strip():
        raise ReportGenerationError("week_id is required")
    return week_id.strip()


def weekly_json_input_path(week_id: str, input_dir: Path | None = None) -> Path:
    base = input_dir or WEEKS_JSON_DIR
    return base / f"{week_id}.json"


def weekly_json_relative_path(week_id: str) -> str:
    return f"data/processed/weeks_json/{week_id}.json"


def report_output_path(week_id: str, output_dir: Path | None = None) -> Path:
    base = output_dir or REPORTS_DIR
    return base / f"{week_id}_weekly_report.md"


def report_output_relative_path(week_id: str) -> str:
    return f"data/reports/{week_id}_weekly_report.md"


def load_optional_context_file(path: Path) -> str:
    """Read a context markdown file if it exists."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_weekly_report_prompt_template(prompt_path: Path | None = None) -> str:
    path = prompt_path or WEEKLY_REPORT_PROMPT_PATH
    if not path.exists():
        raise ReportGenerationError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_weekly_report_prompt(
    template: str,
    *,
    week_id: str,
    start_date: str,
    end_date: str,
    weekly_json: str,
    goals_context: str,
    user_context: str,
) -> str:
    """Fill weekly report prompt placeholders."""
    return (
        template.replace("{week_id}", week_id)
        .replace("{start_date}", start_date)
        .replace("{end_date}", end_date)
        .replace("{weekly_json}", weekly_json)
        .replace("{goals_context}", goals_context)
        .replace("{user_context}", user_context)
    )


def clean_llm_markdown_report(response_text: str, week_id: str) -> str:
    """Strip fences and trim extra text around the markdown report."""
    cleaned = strip_markdown_fences(response_text.strip())
    if not cleaned:
        raise ReportGenerationError("LLM returned an empty report")

    marker = f"{REPORT_TITLE_PREFIX} — {week_id}"
    alt_marker = f"{REPORT_TITLE_PREFIX} - {week_id}"
    start = cleaned.find(marker)
    if start == -1:
        start = cleaned.find(alt_marker)
    if start == -1:
        start = cleaned.find(REPORT_TITLE_PREFIX)
    if start != -1:
        cleaned = cleaned[start:]

    return cleaned.rstrip() + "\n"


def load_weekly_aggregation(
    week_id: str,
    *,
    input_dir: Path | None = None,
) -> tuple[WeeklyAggregation, Path]:
    """Load and validate weekly JSON for a week."""
    json_path = weekly_json_input_path(week_id, input_dir)
    if not json_path.exists():
        raise ReportGenerationError(
            f"weekly JSON not found: {weekly_json_relative_path(week_id)}\n"
            "Run aggregate-week first."
        )
    try:
        weekly = load_json(json_path, WeeklyAggregation)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ReportGenerationError(
            f"Invalid weekly JSON for {week_id}: {json_path}"
        ) from exc
    return weekly, json_path


def generate_report(
    week_id: str,
    *,
    weeks_json_dir: Path | None = None,
    reports_dir: Path | None = None,
    prompt_path: Path | None = None,
    goals_context_path: Path | None = None,
    user_context_path: Path | None = None,
) -> GenerateReportResult:
    """Generate a weekly markdown report from weekly JSON."""
    week_id = _validate_week_id(week_id)
    weekly, input_path = load_weekly_aggregation(week_id, input_dir=weeks_json_dir)

    goals_context = load_optional_context_file(goals_context_path or GOALS_CONTEXT_PATH)
    user_context = load_optional_context_file(user_context_path or USER_CONTEXT_PATH)
    weekly_json = json.dumps(weekly.model_dump(), indent=2, ensure_ascii=False)

    prompt_template = load_weekly_report_prompt_template(prompt_path)
    prompt = render_weekly_report_prompt(
        prompt_template,
        week_id=week_id,
        start_date=weekly.start_date,
        end_date=weekly.end_date,
        weekly_json=weekly_json,
        goals_context=goals_context,
        user_context=user_context,
    )

    model = resolve_report_model()
    try:
        raw_response = generate_text(prompt, model=model)
        report_markdown = clean_llm_markdown_report(raw_response, week_id)
    except LLMError as exc:
        raise ReportGenerationError(str(exc)) from exc

    out_dir = reports_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_output_path(week_id, out_dir)
    output_path.write_text(report_markdown, encoding="utf-8")

    return GenerateReportResult(
        week_id=week_id,
        input_path=input_path,
        output_path=output_path,
    )


def format_generate_report_summary(result: GenerateReportResult) -> str:
    """Format CLI summary for report generation."""
    return "\n".join(
        [
            "Weekly report generated.",
            f"Week ID: {result.week_id}",
            f"Input: {weekly_json_relative_path(result.week_id)}",
            f"Output: {report_output_relative_path(result.week_id)}",
        ]
    )
