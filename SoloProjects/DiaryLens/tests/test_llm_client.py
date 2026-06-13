import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from diarylens.llm_client import (
    LLMError,
    generate_text,
    resolve_model_for_kind,
    resolve_report_model,
    run_test_llm,
)


def test_generate_text_requires_auth_key(monkeypatch):
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)

    with pytest.raises(LLMError, match="GIGACHAT_AUTH_KEY environment variable is not set"):
        generate_text("test prompt")


def test_resolve_model_for_kind_daily(monkeypatch):
    monkeypatch.delenv("GIGACHAT_MODEL_DAILY", raising=False)

    with pytest.raises(LLMError, match="GIGACHAT_MODEL_DAILY environment variable is not set"):
        resolve_model_for_kind("daily")

    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "Custom-Daily")
    assert resolve_model_for_kind("daily") == "Custom-Daily"


def test_resolve_model_for_kind_weekly(monkeypatch):
    monkeypatch.delenv("GIGACHAT_MODEL_WEEKLY", raising=False)

    with pytest.raises(LLMError, match="GIGACHAT_MODEL_WEEKLY environment variable is not set"):
        resolve_model_for_kind("weekly")

    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "Custom-Weekly")
    assert resolve_model_for_kind("weekly") == "Custom-Weekly"


def test_resolve_model_for_kind_report(monkeypatch):
    monkeypatch.delenv("GIGACHAT_MODEL_REPORT", raising=False)

    with pytest.raises(LLMError, match="GIGACHAT_MODEL_REPORT environment variable is not set"):
        resolve_model_for_kind("report")

    monkeypatch.setenv("GIGACHAT_MODEL_REPORT", "Custom-Report")
    assert resolve_model_for_kind("report") == "Custom-Report"


@patch("diarylens.llm_client.GigaChat")
def test_generate_text_uses_selected_model(mock_gigachat, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "secret-auth-key-value")
    monkeypatch.setenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    monkeypatch.delenv("GIGACHAT_TIMEOUT", raising=False)

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"ok": true}'
    mock_client.chat.return_value.choices = [mock_choice]
    mock_gigachat.return_value.__enter__.return_value = mock_client

    response = generate_text("hello", model="custom-explicit-model")

    assert response == '{"ok": true}'
    mock_gigachat.assert_called_once_with(
        credentials="secret-auth-key-value",
        scope="GIGACHAT_API_PERS",
        model="custom-explicit-model",
        verify_ssl_certs=False,
        timeout=120.0,
    )
    mock_client.chat.assert_called_once_with("hello")


@patch("diarylens.llm_client.GigaChat")
def test_generate_text_uses_default_timeout_when_env_not_set(mock_gigachat, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "secret-auth-key-value")
    monkeypatch.delenv("GIGACHAT_TIMEOUT", raising=False)

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"ok": true}'
    mock_client.chat.return_value.choices = [mock_choice]
    mock_gigachat.return_value.__enter__.return_value = mock_client

    generate_text("hello")

    assert mock_gigachat.call_args.kwargs["timeout"] == 120.0


@patch("diarylens.llm_client.GigaChat")
def test_generate_text_uses_timeout_from_env(mock_gigachat, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "secret-auth-key-value")
    monkeypatch.setenv("GIGACHAT_TIMEOUT", "180")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"ok": true}'
    mock_client.chat.return_value.choices = [mock_choice]
    mock_gigachat.return_value.__enter__.return_value = mock_client

    generate_text("hello")

    assert mock_gigachat.call_args.kwargs["timeout"] == 180.0


def test_generate_text_rejects_invalid_timeout(monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "secret-auth-key-value")
    monkeypatch.setenv("GIGACHAT_TIMEOUT", "abc")

    with pytest.raises(LLMError, match="Invalid GIGACHAT_TIMEOUT: expected float seconds."):
        generate_text("hello")


@patch("diarylens.llm_client.generate_text")
def test_run_test_llm_uses_daily_model(mock_generate_text, monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "env-daily-model")
    mock_generate_text.return_value = '{"ok": true}'

    response = run_test_llm("daily")

    assert response == '{"ok": true}'
    mock_generate_text.assert_called_once()
    assert mock_generate_text.call_args.kwargs["model"] == "env-daily-model"


@patch("diarylens.llm_client.generate_text")
def test_run_test_llm_uses_weekly_model(mock_generate_text, monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "env-weekly-model")
    mock_generate_text.return_value = '{"ok": true}'

    run_test_llm("weekly")

    assert mock_generate_text.call_args.kwargs["model"] == "env-weekly-model"


def test_resolve_report_model_uses_report_env_when_set(monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_REPORT", "Custom-Report")
    assert resolve_report_model() == "Custom-Report"


def test_resolve_report_model_falls_back_to_weekly(monkeypatch):
    monkeypatch.delenv("GIGACHAT_MODEL_REPORT", raising=False)
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "Custom-Weekly")
    assert resolve_report_model() == "Custom-Weekly"


def test_resolve_report_model_falls_back_to_weekly_when_report_empty(monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_REPORT", "   ")
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "Custom-Weekly")
    assert resolve_report_model() == "Custom-Weekly"


def test_resolve_model_for_kind_rejects_empty_env(monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "")

    with pytest.raises(LLMError, match="GIGACHAT_MODEL_WEEKLY environment variable is not set"):
        resolve_model_for_kind("weekly")


@patch("diarylens.llm_client.generate_text")
def test_run_test_llm_uses_report_model(mock_generate_text, monkeypatch):
    monkeypatch.setenv("GIGACHAT_MODEL_REPORT", "env-report-model")
    mock_generate_text.return_value = '{"ok": true}'

    run_test_llm("report")

    assert mock_generate_text.call_args.kwargs["model"] == "env-report-model"


def test_test_llm_command_available():
    result = subprocess.run(
        [sys.executable, "-m", "diarylens.cli", "test-llm", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "test-llm" in result.stdout
    assert "--model" in result.stdout


@patch("diarylens.llm_client.GigaChat")
def test_test_llm_cli_does_not_leak_auth_key(mock_gigachat, monkeypatch, capsys):
    from diarylens.cli import _run_test_llm

    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "super-secret-auth-key")
    monkeypatch.setenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    monkeypatch.setenv("GIGACHAT_TIMEOUT", "180")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"ok": true}'
    mock_client.chat.return_value.choices = [mock_choice]
    mock_gigachat.return_value.__enter__.return_value = mock_client

    _run_test_llm("daily")
    captured = capsys.readouterr()

    assert '{"ok": true}' in captured.out
    assert "super-secret-auth-key" not in captured.out
    assert "super-secret-auth-key" not in captured.err
