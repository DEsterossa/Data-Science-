"""Extract structured daily JSON from daily markdown using an LLM."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from diarylens.config import (
    DAY_MANIFESTS_DIR,
    DAYS_JSON_DIR,
    DAYS_JSON_FAILED_DIR,
    PROJECT_ROOT,
    PROMPTS_DIR,
)
from diarylens.json_utils import (
    LLMJsonParseError,
    is_refusal_response,
    parse_llm_json_response as parse_json_from_llm,
)
from diarylens.llm_client import LLMError, generate_text, resolve_model_for_kind
from diarylens.schemas import DailyExtraction, save_json


class DailyExtractionError(Exception):
    """Raised when daily extraction fails."""


DAILY_EXTRACTION_PROMPT_PATH = PROMPTS_DIR / "daily_extraction.md"
DAILY_VERIFICATION_PROMPT_PATH = PROMPTS_DIR / "daily_verification.md"

@dataclass(frozen=True)
class FailedDaySummary:
    date: str
    error_type: str


@dataclass
class ExtractDailyWeekResult:
    output_paths: list[Path] = field(default_factory=list)
    failed_artifact_paths: list[Path] = field(default_factory=list)
    processed: int = 0
    verified: int = 0
    skipped: int = 0
    failed: list[FailedDaySummary] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return bool(self.failed)


def _validate_week_id(week_id: str | None) -> str:
    if not week_id or not week_id.strip():
        raise DailyExtractionError("week_id is required")
    return week_id.strip()


def manifest_input_path(week_id: str, input_dir: Path | None = None) -> Path:
    base = input_dir or DAY_MANIFESTS_DIR
    return base / f"{week_id}_days.json"


def day_json_output_path(date: str, output_dir: Path | None = None) -> Path:
    base = output_dir or DAYS_JSON_DIR
    return base / f"{date}.json"


def failed_day_output_path(date: str, failed_dir: Path | None = None) -> Path:
    base = failed_dir or DAYS_JSON_FAILED_DIR
    return base / f"{date}_error.json"


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_prompt_template(prompt_path: Path | None = None) -> str:
    path = prompt_path or DAILY_EXTRACTION_PROMPT_PATH
    if not path.exists():
        raise DailyExtractionError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_verification_prompt_template(
    prompt_path: Path | None = None,
) -> str:
    path = prompt_path or DAILY_VERIFICATION_PROMPT_PATH
    if not path.exists():
        raise DailyExtractionError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(
    template: str,
    *,
    date: str,
    week_id: str,
    source_day_md: str,
    day_text: str,
) -> str:
    """Fill prompt placeholders for a single diary day."""
    return (
        template.replace("{date}", date)
        .replace("{week_id}", week_id)
        .replace("{source_day_md}", source_day_md)
        .replace("{day_text}", day_text)
    )


def render_verification_prompt(
    template: str,
    *,
    date: str,
    week_id: str,
    source_day_md: str,
    day_text: str,
    draft_json: str,
) -> str:
    """Fill verification prompt placeholders."""
    return (
        template.replace("{date}", date)
        .replace("{week_id}", week_id)
        .replace("{source_day_md}", source_day_md)
        .replace("{day_text}", day_text)
        .replace("{draft_json}", draft_json)
    )


def parse_llm_json_response(response_text: str) -> dict:
    """Parse JSON from an LLM response for daily extraction."""
    try:
        return parse_json_from_llm(
            response_text,
            refusal_message="GigaChat refused to process this day",
        )
    except LLMJsonParseError as exc:
        raise DailyExtractionError(str(exc)) from exc


def validate_daily_extraction(data: dict) -> DailyExtraction:
    """Validate parsed LLM JSON against the DailyExtraction schema."""
    try:
        return DailyExtraction.model_validate(data)
    except ValidationError as exc:
        raise DailyExtractionError(f"LLM JSON failed validation: {exc}") from exc


def classify_extraction_error(exc: Exception, raw_response: str) -> str:
    """Map an extraction exception to a failed-artifact error type."""
    if raw_response and is_refusal_response(raw_response):
        return "llm_refusal"

    message = str(exc).lower()
    if "refused to process" in message:
        return "llm_refusal"
    if "invalid json" in message or "does not contain a json" in message:
        return "invalid_json"
    if "empty response" in message:
        return "invalid_json"
    if "failed validation" in message or "validation" in message:
        return "validation_error"
    return "unknown_error"


def _build_failed_artifact(
    *,
    date: str,
    week_id: str,
    source_day_md: str,
    stage: str,
    error_type: str,
    model: str,
    message: str,
    raw_response: str,
) -> dict:
    return {
        "date": date,
        "week_id": week_id,
        "source_day_md": source_day_md,
        "status": "failed",
        "stage": stage,
        "error_type": error_type,
        "model": model,
        "message": message,
        "raw_response_preview": raw_response[:500],
    }


@dataclass
class _ModelAttemptResult:
    extraction: DailyExtraction | None = None
    error: Exception | None = None
    raw_response: str = ""


def _attempt_extraction(
    *,
    prompt: str,
    model: str,
) -> _ModelAttemptResult:
    raw_response = ""
    try:
        raw_response = generate_text(prompt, model=model)
        data = parse_llm_json_response(raw_response)
        extraction = validate_daily_extraction(data)
        return _ModelAttemptResult(extraction=extraction, raw_response=raw_response)
    except (DailyExtractionError, LLMError) as exc:
        return _ModelAttemptResult(error=exc, raw_response=raw_response)


def extract_day_with_fallback(
    *,
    date: str,
    week_id: str,
    source_day_md: str,
    day_text: str,
    prompt_template: str,
) -> tuple[DailyExtraction | None, dict | None]:
    """Extract structured data for a single diary day using the daily model."""
    prompt = render_prompt(
        prompt_template,
        date=date,
        week_id=week_id,
        source_day_md=source_day_md,
        day_text=day_text,
    )
    model = resolve_model_for_kind("daily")

    primary_result = _attempt_extraction(prompt=prompt, model=model)
    if primary_result.extraction is not None:
        return primary_result.extraction, None

    failed_artifact = _build_failed_artifact(
        date=date,
        week_id=week_id,
        source_day_md=source_day_md,
        stage="extraction",
        error_type=classify_extraction_error(
            primary_result.error or DailyExtractionError("Unknown extraction failure"),
            primary_result.raw_response,
        ),
        model=model,
        message="Daily extraction failed.",
        raw_response=primary_result.raw_response,
    )
    return None, failed_artifact


def verify_day_extraction(
    *,
    date: str,
    week_id: str,
    source_day_md: str,
    day_text: str,
    draft: DailyExtraction,
    verification_template: str,
) -> tuple[DailyExtraction | None, dict | None]:
    """Verify and correct a draft daily extraction against the source day text."""
    draft_json = json.dumps(draft.model_dump(), indent=2, ensure_ascii=False)
    prompt = render_verification_prompt(
        verification_template,
        date=date,
        week_id=week_id,
        source_day_md=source_day_md,
        day_text=day_text,
        draft_json=draft_json,
    )
    model = resolve_model_for_kind("daily")

    verify_result = _attempt_extraction(prompt=prompt, model=model)
    if verify_result.extraction is not None:
        return verify_result.extraction, None

    failed_artifact = _build_failed_artifact(
        date=date,
        week_id=week_id,
        source_day_md=source_day_md,
        stage="verification",
        error_type=classify_extraction_error(
            verify_result.error or DailyExtractionError("Unknown verification failure"),
            verify_result.raw_response,
        ),
        model=model,
        message="Daily verification failed.",
        raw_response=verify_result.raw_response,
    )
    return None, failed_artifact


def build_empty_daily_extraction(
    *,
    date: str,
    week_id: str,
    source_day_md: str,
) -> DailyExtraction:
    """Build a valid daily JSON placeholder with empty field values."""
    return DailyExtraction(
        date=date,
        week_id=week_id,
        source_day_md=source_day_md,
        short_summary="",
    )


def _save_failed_day_with_placeholder(
    *,
    date: str,
    week_id: str,
    source_day_md: str,
    failed_artifact: dict,
    output_path: Path,
    failed_out_dir: Path,
    result: ExtractDailyWeekResult,
) -> None:
    """Save failed-artifact diagnostics and an empty daily JSON placeholder."""
    failed_path = save_failed_artifact(
        failed_artifact,
        failed_day_output_path(date, failed_out_dir),
    )
    result.failed_artifact_paths.append(failed_path)
    result.failed.append(
        FailedDaySummary(
            date=date,
            error_type=failed_artifact["error_type"],
        )
    )

    placeholder = build_empty_daily_extraction(
        date=date,
        week_id=week_id,
        source_day_md=source_day_md,
    )
    saved_path = save_json(placeholder, output_path)
    result.output_paths.append(saved_path)


def save_failed_artifact(artifact: dict, output_path: Path) -> Path:
    """Save a failed daily extraction artifact as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_week_manifest(
    week_id: str,
    *,
    manifest_dir: Path | None = None,
) -> dict:
    manifest_path = manifest_input_path(week_id, manifest_dir)
    if not manifest_path.exists():
        raise DailyExtractionError(f"Day manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DailyExtractionError(f"Invalid day manifest JSON: {manifest_path}") from exc

    if not isinstance(manifest, dict) or "days" not in manifest:
        raise DailyExtractionError(f"Day manifest is missing 'days': {manifest_path}")

    return manifest


def extract_daily_for_week(
    week_id: str,
    *,
    verify: bool = False,
    force: bool = False,
    manifest_dir: Path | None = None,
    days_md_dir: Path | None = None,
    output_dir: Path | None = None,
    failed_dir: Path | None = None,
    prompt_path: Path | None = None,
    verification_prompt_path: Path | None = None,
) -> ExtractDailyWeekResult:
    """Extract daily JSON files for all days listed in a week manifest."""
    week_id = _validate_week_id(week_id)
    manifest = load_week_manifest(week_id, manifest_dir=manifest_dir)
    prompt_template = load_prompt_template(prompt_path)
    verification_template = (
        load_verification_prompt_template(verification_prompt_path) if verify else None
    )

    days = manifest.get("days", [])
    if not days:
        raise DailyExtractionError(f"No days found in manifest for week: {week_id}")

    out_dir = output_dir or DAYS_JSON_DIR
    failed_out_dir = failed_dir or DAYS_JSON_FAILED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_out_dir.mkdir(parents=True, exist_ok=True)

    result = ExtractDailyWeekResult()
    for day_entry in days:
        date = day_entry["date"]
        source_day_md = day_entry["path"]
        output_path = day_json_output_path(date, out_dir)
        if output_path.exists() and not force:
            result.output_paths.append(output_path)
            result.skipped += 1
            continue

        day_md_path = _resolve_project_path(source_day_md)
        if days_md_dir is not None:
            day_md_path = days_md_dir / f"{date}.md"

        if not day_md_path.exists():
            raise DailyExtractionError(f"Daily markdown not found: {day_md_path}")

        day_text = day_md_path.read_text(encoding="utf-8")
        extraction, failed_artifact = extract_day_with_fallback(
            date=date,
            week_id=week_id,
            source_day_md=source_day_md,
            day_text=day_text,
            prompt_template=prompt_template,
        )

        if extraction is None:
            assert failed_artifact is not None
            _save_failed_day_with_placeholder(
                date=date,
                week_id=week_id,
                source_day_md=source_day_md,
                failed_artifact=failed_artifact,
                output_path=output_path,
                failed_out_dir=failed_out_dir,
                result=result,
            )
            continue

        if verify:
            assert verification_template is not None
            extraction, failed_artifact = verify_day_extraction(
                date=date,
                week_id=week_id,
                source_day_md=source_day_md,
                day_text=day_text,
                draft=extraction,
                verification_template=verification_template,
            )
            if extraction is None:
                assert failed_artifact is not None
                _save_failed_day_with_placeholder(
                    date=date,
                    week_id=week_id,
                    source_day_md=source_day_md,
                    failed_artifact=failed_artifact,
                    output_path=output_path,
                    failed_out_dir=failed_out_dir,
                    result=result,
                )
                continue
            result.verified += 1

        saved_path = save_json(extraction, output_path)
        result.output_paths.append(saved_path)
        result.processed += 1

    return result


def format_extract_daily_summary(result: ExtractDailyWeekResult) -> str:
    """Format CLI summary for daily extraction."""
    lines = [
        f"Processed: {result.processed}",
        f"Verified: {result.verified}",
        f"Skipped: {result.skipped}",
        f"Failed: {len(result.failed)}",
    ]
    if result.failed:
        lines.append("Failed days:")
        for item in result.failed:
            lines.append(f"- {item.date}: {item.error_type}")
    return "\n".join(lines)
