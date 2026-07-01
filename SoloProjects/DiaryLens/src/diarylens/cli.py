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
from diarylens.ask_history import (
    AskHistoryConfig,
    AskHistoryDebug,
    AskHistoryError,
    ask_history,
    search_memory,
)
from diarylens.ask_history.config import RETRIEVAL_PRESET_VALUES


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

    def _add_retrieval_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "question_arg",
            nargs="?",
            help="Question to ask about diary history",
        )
        command_parser.add_argument(
            "--question",
            dest="question_option",
            help="Question to ask about diary history",
        )
        command_parser.add_argument(
            "--force-rebuild-index",
            action="store_true",
            help="Rebuild data/memory/day_index.csv before retrieval",
        )
        command_parser.add_argument(
            "--top-k-days",
            type=int,
            help="Number of semantic day results to inspect",
        )
        command_parser.add_argument(
            "--top-k-chunks",
            type=int,
            help="Number of raw evidence chunks to return",
        )
        command_parser.add_argument(
            "--chunk-size",
            type=int,
            help="Raw day chunk size in characters",
        )
        command_parser.add_argument(
            "--overlap",
            type=int,
            help="Chunk overlap in characters",
        )
        command_parser.add_argument(
            "--preset",
            choices=sorted(RETRIEVAL_PRESET_VALUES),
            default="full",
            help="Retrieval preset",
        )

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

    ask_history_parser = subparsers.add_parser(
        "ask-history",
        help="Ask a question about diary history using raw day evidence",
    )
    _add_retrieval_options(ask_history_parser)
    ask_history_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved days and evidence chunks before the answer",
    )

    search_memory_parser = subparsers.add_parser(
        "search-memory",
        help="Search diary memory and print top raw evidence chunks",
    )
    _add_retrieval_options(search_memory_parser)

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


def _resolve_question(question_arg: str | None, question_option: str | None) -> str:
    question = question_option or question_arg
    if question is None or not question.strip():
        raise AskHistoryError("question is required")
    return question.strip()


def _config_from_retrieval_args(args: argparse.Namespace) -> AskHistoryConfig:
    values = dict(RETRIEVAL_PRESET_VALUES[args.preset])
    overrides = {
        "top_k_days": args.top_k_days,
        "top_k_chunks": args.top_k_chunks,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return AskHistoryConfig(**values)


def _preview(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _print_ask_history_debug(debug: AskHistoryDebug) -> None:
    print("# Debug: day results")
    for result in debug.day_results:
        print(
            f"{result.rank} | {result.score:.4f} | {result.date} | "
            f"{result.source_day_md} | {_preview(result.embedding_text_preview)}"
        )

    print("\n# Debug: evidence chunks")
    for result in debug.evidence_results:
        print(
            f"{result.rank} | {result.score:.4f} | {result.date} | "
            f"chunk={result.chunk_index} | {result.source_day_md}"
        )
        print(_preview(result.text))


def _run_ask_history_command(args: argparse.Namespace) -> None:
    try:
        question = _resolve_question(args.question_arg, args.question_option)
        debug_or_answer = ask_history(
            question,
            config=_config_from_retrieval_args(args),
            return_debug=args.debug,
            force_rebuild_index=args.force_rebuild_index,
        )
    except AskHistoryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.debug:
        debug = debug_or_answer
        assert isinstance(debug, AskHistoryDebug)
        _print_ask_history_debug(debug)
        print("\n# Answer")
        print(debug.answer or "")
    else:
        print(debug_or_answer)


def _run_search_memory_command(args: argparse.Namespace) -> None:
    try:
        question = _resolve_question(args.question_arg, args.question_option)
        debug = search_memory(
            question,
            config=_config_from_retrieval_args(args),
            force_rebuild_index=args.force_rebuild_index,
        )
    except AskHistoryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Rank | Score | Date | Chunk | Source")
    for result in debug.evidence_results:
        print(
            f"{result.rank} | {result.score:.4f} | {result.date} | "
            f"{result.chunk_index} | {result.source_day_md}"
        )
        print(_preview(result.text))
        print()


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
    elif args.command == "ask-history":
        _run_ask_history_command(args)
    elif args.command == "search-memory":
        _run_search_memory_command(args)
    elif args.command == "test-llm":
        _run_test_llm(args.model)


if __name__ == "__main__":
    main()
