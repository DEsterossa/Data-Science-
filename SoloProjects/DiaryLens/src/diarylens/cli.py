"""Command-line interface for DiaryLens."""

import argparse
import sys
from pathlib import Path

from diarylens import __version__
from diarylens.config import ensure_project_dirs
from diarylens.pdf_extractor import PdfExtractionError, extract_pdf_to_raw_markdown
from diarylens.daily_extractor import (
    DailyExtractionError,
    extract_daily_for_week,
    format_extract_daily_summary,
)
from diarylens.day_splitter import DaySplitError, split_clean_markdown
from diarylens.llm_client import LLMError, run_test_llm
from diarylens.text_cleaner import TextCleaningError, clean_raw_markdown
from diarylens.report_generator import (
    ReportGenerationError,
    format_generate_report_summary,
    generate_report,
)
from diarylens.weekly_aggregator import (
    WeeklyAggregationError,
    aggregate_week,
    format_aggregate_week_summary,
    format_missing_days_warning,
)
from diarylens.week_runner import WeekRunnerError, run_week_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diarylens",
        description="Weekly diary analysis pipeline (v0.1)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser(
        "extract-pdf",
        help="Extract text from a weekly PDF into raw markdown",
    )
    extract_parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the weekly PDF file",
    )
    extract_parser.add_argument(
        "--week-id",
        required=True,
        help="Week identifier, e.g. 2026-W23",
    )

    clean_parser = subparsers.add_parser(
        "clean-text",
        help="Clean raw markdown into normalized clean markdown",
    )
    clean_parser.add_argument(
        "--week-id",
        required=True,
        help="Week identifier, e.g. 2026-W22",
    )

    split_parser = subparsers.add_parser(
        "split-days",
        help="Split clean markdown into daily markdown files",
    )
    split_parser.add_argument(
        "--week-id",
        required=True,
        help="Week identifier, e.g. 2026-W22",
    )

    extract_daily_parser = subparsers.add_parser(
        "extract-daily",
        help="Extract structured daily JSON from daily markdown using LLM",
    )
    extract_daily_parser.add_argument(
        "--week-id",
        required=True,
        help="Week identifier, e.g. 2026-W22",
    )
    extract_daily_parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a second verification pass against the source day text",
    )

    aggregate_week_parser = subparsers.add_parser(
        "aggregate-week",
        help="Aggregate daily JSON files into weekly JSON using LLM",
    )
    aggregate_week_parser.add_argument(
        "--week-id",
        required=True,
        help="Week identifier, e.g. 2026-W22",
    )

    generate_report_parser = subparsers.add_parser(
        "generate-report",
        help="Generate weekly markdown report from weekly JSON using LLM",
    )
    generate_report_parser.add_argument(
        "--week-id",
        required=True,
        help="Week identifier, e.g. 2026-W22",
    )

    run_week_parser = subparsers.add_parser(
        "run-week",
        help="Run the full weekly pipeline from PDF to report",
    )
    run_week_parser.add_argument(
        "--week-id",
        required=True,
        help="Week identifier, e.g. 2026-W22",
    )
    run_week_parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the weekly PDF file",
    )
    run_week_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip daily JSON verification pass",
    )
    run_week_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing pipeline outputs including the final report",
    )

    test_llm_parser = subparsers.add_parser(
        "test-llm",
        help="Send a short test prompt to GigaChat",
    )
    test_llm_parser.add_argument(
        "--model",
        choices=["daily", "weekly", "report"],
        default="daily",
        help="Which configured GigaChat model to test",
    )

    return parser


def _run_extract_pdf(pdf_path: Path, week_id: str) -> None:
    try:
        output_path = extract_pdf_to_raw_markdown(pdf_path, week_id)
    except PdfExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(output_path)


def _run_clean_text(week_id: str) -> None:
    try:
        output_path = clean_raw_markdown(week_id)
    except TextCleaningError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(output_path)


def _run_split_days(week_id: str) -> None:
    try:
        output_paths = split_clean_markdown(week_id)
    except DaySplitError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for path in output_paths:
        print(path)


def _run_extract_daily(week_id: str, *, verify: bool = False) -> None:
    try:
        result = extract_daily_for_week(week_id, verify=verify)
    except (DailyExtractionError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(format_extract_daily_summary(result))
    for path in result.output_paths:
        print(path)
    for path in result.failed_artifact_paths:
        print(path, file=sys.stderr)


def _run_aggregate_week(week_id: str) -> None:
    try:
        result = aggregate_week(week_id)
    except WeeklyAggregationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    warning = format_missing_days_warning(result.missing_days)
    if warning:
        print(warning, file=sys.stderr)

    print(format_aggregate_week_summary(result))


def _run_generate_report(week_id: str) -> None:
    try:
        result = generate_report(week_id)
    except ReportGenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(format_generate_report_summary(result))


def _run_run_week(
    week_id: str,
    pdf_path: Path,
    *,
    verify_daily: bool = True,
    force: bool = False,
) -> None:
    try:
        run_week_pipeline(
            week_id,
            pdf_path,
            verify_daily=verify_daily,
            force=force,
        )
    except WeekRunnerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _run_test_llm(model_kind: str) -> None:
    try:
        response = run_test_llm(model_kind)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(response)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ensure_project_dirs()

    if args.command == "extract-pdf":
        _run_extract_pdf(args.pdf_path, args.week_id)
    elif args.command == "clean-text":
        _run_clean_text(args.week_id)
    elif args.command == "split-days":
        _run_split_days(args.week_id)
    elif args.command == "extract-daily":
        _run_extract_daily(args.week_id, verify=args.verify)
    elif args.command == "aggregate-week":
        _run_aggregate_week(args.week_id)
    elif args.command == "generate-report":
        _run_generate_report(args.week_id)
    elif args.command == "run-week":
        _run_run_week(
            args.week_id,
            args.pdf,
            verify_daily=not args.no_verify,
            force=args.force,
        )
    elif args.command == "test-llm":
        _run_test_llm(args.model)


if __name__ == "__main__":
    main()
