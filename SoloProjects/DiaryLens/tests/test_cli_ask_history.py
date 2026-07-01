import subprocess
import sys
from pathlib import Path

import pytest

from diarylens.ask_history.models import (
    AskHistoryConfig,
    AskHistoryDebug,
    DaySearchResult,
    EvidenceSearchResult,
)
from diarylens.cli import main


def test_ask_history_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "ask-history", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ask-history" in result.stdout
    assert "--debug" in result.stdout
    assert "--force-rebuild-index" in result.stdout


def test_search_memory_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "search-memory", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "search-memory" in result.stdout
    assert "--top-k-days" in result.stdout


def test_ask_history_cli_calls_pipeline(monkeypatch, capsys):
    captured = {}

    def fake_ask_history(question, config=None, return_debug=False, force_rebuild_index=False):
        captured["question"] = question
        captured["config"] = config
        captured["return_debug"] = return_debug
        captured["force_rebuild_index"] = force_rebuild_index
        return "# Ответ\n\nГотово."

    monkeypatch.setattr("diarylens.cli.ask_history", fake_ask_history)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diarylens",
            "ask-history",
            "Когда скучно?",
            "--top-k-days",
            "3",
            "--force-rebuild-index",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "# Ответ" in output
    assert captured["question"] == "Когда скучно?"
    assert captured["config"].top_k_days == 3
    assert captured["force_rebuild_index"] is True


def test_ask_history_cli_debug_prints_day_and_evidence(monkeypatch, capsys):
    debug = AskHistoryDebug(
        question="Когда скучно?",
        config=AskHistoryConfig(),
        day_results=[
            DaySearchResult(
                rank=1,
                date="2026-06-12",
                week_id="2026-W24",
                score=0.9,
                source_day_md="data/processed/days_md/2026-06-12.md",
                source_daily_json="data/processed/days_json/2026-06-12.json",
                embedding_text_preview="скука и прогулка",
            )
        ],
        evidence_results=[
            EvidenceSearchResult(
                rank=1,
                date="2026-06-12",
                week_id="2026-W24",
                score=0.8,
                source_day_md="data/processed/days_md/2026-06-12.md",
                chunk_index=2,
                text="Сегодня была скука.",
            )
        ],
        evidence_context="context",
        prompt="prompt",
        answer="# Ответ\n\nDebug.",
    )
    monkeypatch.setattr("diarylens.cli.ask_history", lambda *args, **kwargs: debug)
    monkeypatch.setattr(sys, "argv", ["diarylens", "ask-history", "Когда скучно?", "--debug"])

    main()

    output = capsys.readouterr().out
    assert "# Debug: day results" in output
    assert "# Debug: evidence chunks" in output
    assert "# Answer" in output
    assert "2026-06-12" in output
    assert "Сегодня была скука." in output


def test_search_memory_cli_prints_top_chunks(monkeypatch, capsys):
    debug = AskHistoryDebug(
        question="Когда скучно?",
        config=AskHistoryConfig(),
        day_results=[],
        evidence_results=[
            EvidenceSearchResult(
                rank=1,
                date="2026-06-12",
                week_id="2026-W24",
                score=0.8,
                source_day_md="data/processed/days_md/2026-06-12.md",
                chunk_index=0,
                text="Скука и прогулка.",
            )
        ],
        evidence_context="context",
        prompt="",
        answer=None,
    )
    monkeypatch.setattr("diarylens.cli.search_memory", lambda *args, **kwargs: debug)
    monkeypatch.setattr(
        sys,
        "argv",
        ["diarylens", "search-memory", "--question", "Когда скучно?"],
    )

    main()

    output = capsys.readouterr().out
    assert "Rank | Score | Date | Chunk | Source" in output
    assert "Скука и прогулка." in output


def test_ask_history_cli_does_not_print_api_key(monkeypatch, capsys):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "super-secret-key")
    monkeypatch.setattr("diarylens.cli.ask_history", lambda *args, **kwargs: "# Ответ")
    monkeypatch.setattr(sys, "argv", ["diarylens", "ask-history", "Вопрос"])

    main()

    captured = capsys.readouterr()
    assert "super-secret-key" not in captured.out
    assert "super-secret-key" not in captured.err
