"""Run the full weekly DiaryLens pipeline in one command."""

import sys
from pathlib import Path

from diarylens.config import PROJECT_ROOT
from diarylens.daily_extractor import (
    DailyExtractionError,
    extract_daily_for_week,
    load_week_manifest,
)
from diarylens.day_splitter import (
    DaySplitError,
    manifest_output_path,
    split_clean_markdown,
)
from diarylens.pdf_extractor import (
    PdfExtractionError,
    extract_pdf_to_raw_markdown,
    raw_text_output_path,
)
from diarylens.report_generator import (
    ReportGenerationError,
    generate_report,
    report_output_path,
    report_output_relative_path,
)
from diarylens.text_cleaner import (
    TextCleaningError,
    clean_raw_markdown,
    clean_text_output_path,
)
from diarylens.weekly_aggregator import (
    WeeklyAggregationError,
    aggregate_week,
    format_missing_days_warning,
    week_json_output_path,
)


class WeekRunnerError(Exception):
    """Raised when the weekly pipeline fails."""


def _validate_week_id(week_id: str | None) -> str:
    if not week_id or not week_id.strip():
        raise WeekRunnerError("week_id is required")
    return week_id.strip()


def _validate_pdf_path(pdf_path: Path) -> Path:
    resolved = pdf_path.resolve()
    if not resolved.exists():
        raise WeekRunnerError(f"PDF not found: {pdf_path}")
    if resolved.suffix.lower() != ".pdf":
        raise WeekRunnerError(f"Not a PDF file: {pdf_path}")
    return resolved


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_days_count(week_id: str) -> int:
    manifest = load_week_manifest(week_id)
    return len(manifest.get("days", []))


def run_week_pipeline(
    week_id: str,
    pdf_path: str | Path,
    *,
    verify_daily: bool = True,
    force: bool = False,
) -> Path:
    """Run all weekly pipeline steps and return the final report path."""
    week_id = _validate_week_id(week_id)
    pdf_path = _validate_pdf_path(Path(pdf_path))
    final_report_path = report_output_path(week_id)

    if final_report_path.exists() and not force:
        raise WeekRunnerError(
            f"report already exists: {report_output_relative_path(week_id)}\n"
            "Use --force to overwrite."
        )

    print("Running DiaryLens weekly pipeline")
    print(f"Week ID: {week_id}")
    print(f"PDF: {_display_path(pdf_path)}")
    print()

    raw_path = raw_text_output_path(week_id)
    print("[1/6] Extracting PDF...")
    if raw_path.exists() and not force:
        print(f"Reusing existing file: {_display_path(raw_path)}")
    else:
        try:
            raw_path = extract_pdf_to_raw_markdown(pdf_path, week_id)
        except PdfExtractionError as exc:
            raise WeekRunnerError(str(exc)) from exc
        print(f"Output: {_display_path(raw_path)}")
    print()

    clean_path = clean_text_output_path(week_id)
    print("[2/6] Cleaning text...")
    if clean_path.exists() and not force:
        print(f"Reusing existing file: {_display_path(clean_path)}")
    else:
        try:
            clean_path = clean_raw_markdown(week_id)
        except TextCleaningError as exc:
            raise WeekRunnerError(str(exc)) from exc
        print(f"Output: {_display_path(clean_path)}")
    print()

    manifest_path = manifest_output_path(week_id)
    print("[3/6] Splitting days...")
    if manifest_path.exists() and not force:
        print(f"Reusing existing manifest: {_display_path(manifest_path)}")
    else:
        try:
            split_clean_markdown(week_id)
        except DaySplitError as exc:
            raise WeekRunnerError(str(exc)) from exc
        print(f"Manifest: {_display_path(manifest_path)}")
    try:
        days_count = _load_days_count(week_id)
    except DailyExtractionError as exc:
        raise WeekRunnerError(str(exc)) from exc
    print(f"Days found: {days_count}")
    print()

    print("[4/6] Extracting daily JSON...")
    print(f"Verify: {str(verify_daily).lower()}")
    try:
        daily_result = extract_daily_for_week(
            week_id,
            verify=verify_daily,
            force=force,
        )
    except DailyExtractionError as exc:
        raise WeekRunnerError(str(exc)) from exc

    if daily_result.has_failures:
        print(
            f"Warning: {len(daily_result.failed)} day(s) failed extraction; "
            "empty placeholders saved. Check data/processed/days_json_failed/.",
            file=sys.stderr,
        )
        for item in daily_result.failed:
            print(f"  - {item.date}: {item.error_type}", file=sys.stderr)

    print(f"Days processed: {len(daily_result.output_paths)}")
    print()

    weekly_path = week_json_output_path(week_id)
    print("[5/6] Aggregating week...")
    if weekly_path.exists() and not force:
        print(f"Reusing existing file: {_display_path(weekly_path)}")
    else:
        try:
            aggregate_result = aggregate_week(week_id)
        except WeeklyAggregationError as exc:
            raise WeekRunnerError(str(exc)) from exc

        warning = format_missing_days_warning(aggregate_result.missing_days)
        if warning:
            print(warning, file=sys.stderr)

        weekly_path = aggregate_result.output_path
        print(f"Output: {_display_path(weekly_path)}")

    if not weekly_path.exists():
        raise WeekRunnerError(
            "weekly aggregation failed. Weekly JSON was not created."
        )
    print()

    print("[6/6] Generating report...")
    try:
        report_result = generate_report(week_id)
    except ReportGenerationError as exc:
        raise WeekRunnerError(str(exc)) from exc
    print(f"Output: {report_output_relative_path(week_id)}")
    print()
    print("Done.")

    return report_result.output_path
