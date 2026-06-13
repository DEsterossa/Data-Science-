from pathlib import Path
from unittest.mock import patch

import pytest

from diarylens.daily_extractor import ExtractDailyWeekResult, FailedDaySummary
from diarylens.report_generator import GenerateReportResult
from diarylens.weekly_aggregator import AggregateWeekResult
from diarylens.week_runner import WeekRunnerError, run_week_pipeline


WEEK_ID = "2026-W22"


@pytest.fixture
def pipeline_paths(tmp_path, monkeypatch):
    data = tmp_path / "data"
    reports = data / "reports"
    weeks_json = data / "processed" / "weeks_json"
    interim_raw = data / "interim" / "raw_text"
    interim_clean = data / "interim" / "clean_text"
    manifests = data / "processed" / "day_manifests"

    for path in (reports, weeks_json, interim_raw, interim_clean, manifests):
        path.mkdir(parents=True)

    monkeypatch.setattr("diarylens.week_runner.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("diarylens.pdf_extractor.INTERIM_RAW_TEXT_DIR", interim_raw)
    monkeypatch.setattr("diarylens.text_cleaner.INTERIM_CLEAN_TEXT_DIR", interim_clean)
    monkeypatch.setattr("diarylens.day_splitter.DAY_MANIFESTS_DIR", manifests)
    monkeypatch.setattr("diarylens.weekly_aggregator.WEEKS_JSON_DIR", weeks_json)
    monkeypatch.setattr("diarylens.report_generator.REPORTS_DIR", reports)
    monkeypatch.setattr("diarylens.daily_extractor.DAY_MANIFESTS_DIR", manifests)

    def load_manifest_or_default(week_id: str, manifest_dir=None):
        path = manifests / f"{week_id}_days.json"
        if path.exists():
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        return {"days": [{"date": "2026-05-26"}, {"date": "2026-05-27"}]}

    monkeypatch.setattr(
        "diarylens.week_runner.load_week_manifest",
        load_manifest_or_default,
    )

    pdf_path = tmp_path / "week.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    raw_path = interim_raw / f"{WEEK_ID}_raw.md"
    clean_path = interim_clean / f"{WEEK_ID}_clean.md"
    manifest_path = manifests / f"{WEEK_ID}_days.json"
    weekly_path = weeks_json / f"{WEEK_ID}.json"
    report_path = reports / f"{WEEK_ID}_weekly_report.md"

    return {
        "pdf_path": pdf_path,
        "raw_path": raw_path,
        "clean_path": clean_path,
        "manifest_path": manifest_path,
        "weekly_path": weekly_path,
        "report_path": report_path,
    }


def _daily_result(output_paths: list[Path]) -> ExtractDailyWeekResult:
    return ExtractDailyWeekResult(output_paths=output_paths, processed=len(output_paths))


def _aggregate_result(weekly_path: Path) -> AggregateWeekResult:
    return AggregateWeekResult(
        week_id=WEEK_ID,
        output_path=weekly_path,
        days_included=["2026-05-26", "2026-05-27"],
        missing_days=[],
    )


def _report_result(report_path: Path) -> GenerateReportResult:
    return GenerateReportResult(
        week_id=WEEK_ID,
        input_path=report_path.parent / f"{WEEK_ID}.json",
        output_path=report_path,
    )


def _configure_aggregate_mock(mock_aggregate, paths: dict) -> None:
    def _run(week_id: str) -> AggregateWeekResult:
        paths["weekly_path"].write_text("{}", encoding="utf-8")
        return _aggregate_result(paths["weekly_path"])

    mock_aggregate.side_effect = _run


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_calls_steps_in_order(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
):
    paths = pipeline_paths
    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = _daily_result(
        [Path("2026-05-26.json"), Path("2026-05-27.json")]
    )
    _configure_aggregate_mock(mock_aggregate, paths)
    mock_generate_report.return_value = _report_result(paths["report_path"])

    result = run_week_pipeline(WEEK_ID, paths["pdf_path"])

    assert result == paths["report_path"]
    mock_extract_pdf.assert_called_once_with(paths["pdf_path"], WEEK_ID)
    mock_clean.assert_called_once_with(WEEK_ID)
    mock_split.assert_called_once_with(WEEK_ID)
    mock_extract_daily.assert_called_once_with(
        WEEK_ID,
        verify=True,
        force=False,
    )
    mock_aggregate.assert_called_once_with(WEEK_ID)
    mock_generate_report.assert_called_once_with(WEEK_ID)


def test_run_week_pipeline_pdf_not_found(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"
    with pytest.raises(WeekRunnerError, match="PDF not found"):
        run_week_pipeline(WEEK_ID, missing_pdf)


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_verify_daily_default_true(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
):
    paths = pipeline_paths
    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = _daily_result([Path("2026-05-26.json")])
    _configure_aggregate_mock(mock_aggregate, paths)
    mock_generate_report.return_value = _report_result(paths["report_path"])

    run_week_pipeline(WEEK_ID, paths["pdf_path"])

    assert mock_extract_daily.call_args.kwargs["verify"] is True


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_no_verify(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
):
    paths = pipeline_paths
    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = _daily_result([Path("2026-05-26.json")])
    _configure_aggregate_mock(mock_aggregate, paths)
    mock_generate_report.return_value = _report_result(paths["report_path"])

    run_week_pipeline(WEEK_ID, paths["pdf_path"], verify_daily=False)

    assert mock_extract_daily.call_args.kwargs["verify"] is False


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_force_passed_to_daily_extraction(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
):
    paths = pipeline_paths
    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = _daily_result([Path("2026-05-26.json")])
    _configure_aggregate_mock(mock_aggregate, paths)
    mock_generate_report.return_value = _report_result(paths["report_path"])

    run_week_pipeline(WEEK_ID, paths["pdf_path"], force=True)

    assert mock_extract_daily.call_args.kwargs["force"] is True
    mock_extract_pdf.assert_called_once()
    mock_clean.assert_called_once()
    mock_split.assert_called_once()
    mock_aggregate.assert_called_once()


def test_run_week_pipeline_report_exists_without_force(pipeline_paths):
    paths = pipeline_paths
    paths["report_path"].write_text("# existing report", encoding="utf-8")

    with pytest.raises(WeekRunnerError, match="report already exists"):
        run_week_pipeline(WEEK_ID, paths["pdf_path"])


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_force_overwrites_report(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
):
    paths = pipeline_paths
    paths["report_path"].write_text("# old report", encoding="utf-8")

    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = _daily_result([Path("2026-05-26.json")])
    _configure_aggregate_mock(mock_aggregate, paths)
    mock_generate_report.return_value = _report_result(paths["report_path"])

    result = run_week_pipeline(WEEK_ID, paths["pdf_path"], force=True)

    assert result == paths["report_path"]
    mock_generate_report.assert_called_once()


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_reuses_existing_intermediates_without_force(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
    capsys,
):
    paths = pipeline_paths
    paths["raw_path"].write_text("raw", encoding="utf-8")
    paths["clean_path"].write_text("clean", encoding="utf-8")
    paths["manifest_path"].write_text(
        '{"days": [{"date": "2026-05-26"}]}',
        encoding="utf-8",
    )
    paths["weekly_path"].write_text("{}", encoding="utf-8")

    mock_extract_daily.return_value = _daily_result([Path("2026-05-26.json")])
    mock_generate_report.return_value = _report_result(paths["report_path"])

    run_week_pipeline(WEEK_ID, paths["pdf_path"])

    mock_extract_pdf.assert_not_called()
    mock_clean.assert_not_called()
    mock_split.assert_not_called()
    mock_aggregate.assert_not_called()
    mock_generate_report.assert_called_once()

    output = capsys.readouterr().out
    assert "Reusing existing file" in output


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_continues_after_daily_failures_with_warning(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
    capsys,
):
    paths = pipeline_paths
    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = ExtractDailyWeekResult(
        failed=[FailedDaySummary(date="2026-05-30", error_type="llm_refusal")],
        output_paths=[
            Path("data/processed/days_json/2026-05-30.json"),
            Path("data/processed/days_json/2026-05-31.json"),
        ],
    )
    _configure_aggregate_mock(mock_aggregate, paths)
    mock_generate_report.return_value = _report_result(paths["report_path"])

    result = run_week_pipeline(WEEK_ID, paths["pdf_path"])

    assert result == paths["report_path"]
    mock_aggregate.assert_called_once()
    mock_generate_report.assert_called_once()

    captured = capsys.readouterr()
    assert "Warning:" in captured.err
    assert "2026-05-30" in captured.err


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_does_not_leak_api_keys(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
    capsys,
    monkeypatch,
):
    paths = pipeline_paths
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "super-secret-api-key-value")

    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = _daily_result([Path("2026-05-26.json")])
    _configure_aggregate_mock(mock_aggregate, paths)
    mock_generate_report.return_value = _report_result(paths["report_path"])

    run_week_pipeline(WEEK_ID, paths["pdf_path"])

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "super-secret-api-key-value" not in combined


@patch("diarylens.week_runner.generate_report")
@patch("diarylens.week_runner.aggregate_week")
@patch("diarylens.week_runner.extract_daily_for_week")
@patch("diarylens.week_runner.split_clean_markdown")
@patch("diarylens.week_runner.clean_raw_markdown")
@patch("diarylens.week_runner.extract_pdf_to_raw_markdown")
def test_run_week_pipeline_missing_days_warning(
    mock_extract_pdf,
    mock_clean,
    mock_split,
    mock_extract_daily,
    mock_aggregate,
    mock_generate_report,
    pipeline_paths,
    capsys,
):
    paths = pipeline_paths
    mock_extract_pdf.return_value = paths["raw_path"]
    mock_clean.return_value = paths["clean_path"]
    mock_split.return_value = []
    mock_extract_daily.return_value = _daily_result([Path("2026-05-26.json")])

    def aggregate_with_missing_days(week_id: str) -> AggregateWeekResult:
        paths["weekly_path"].write_text("{}", encoding="utf-8")
        return AggregateWeekResult(
            week_id=WEEK_ID,
            output_path=paths["weekly_path"],
            days_included=["2026-05-26"],
            missing_days=["2026-05-27"],
        )

    mock_aggregate.side_effect = aggregate_with_missing_days
    mock_generate_report.return_value = _report_result(paths["report_path"])

    run_week_pipeline(WEEK_ID, paths["pdf_path"])

    captured = capsys.readouterr()
    assert "missing" in captured.err.lower() or "2026-05-27" in captured.err
