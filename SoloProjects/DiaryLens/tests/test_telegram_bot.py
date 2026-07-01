import asyncio
import re
from pathlib import Path

import pytest

from diarylens import config
from diarylens.telegram_bot import (
    BUSY_MESSAGE,
    LONG_ANSWER_NOTICE,
    TELEGRAM_MESSAGE_LIMIT,
    TelegramBotConfigError,
    _load_telegram_token,
    format_telegram_answer,
    handle_text_question,
    save_ask_history_answer_markdown,
)


class FakeMessage:
    def __init__(self, text: str | None):
        self.text = text
        self.text_replies: list[str] = []
        self.document_replies: list[str] = []

    async def reply_text(self, text: str):
        self.text_replies.append(text)

    async def reply_document(self, document, filename: str):
        self.document_replies.append(filename)
        assert document.read()


class FakeUpdate:
    def __init__(self, message: FakeMessage | None):
        self.message = message


def test_save_ask_history_answer_markdown_writes_expected_file(monkeypatch):
    answers_dir = Path("data") / "answers" / "test_telegram_bot"
    monkeypatch.setattr(config, "ASK_HISTORY_ANSWERS_DIR", answers_dir)

    path = save_ask_history_answer_markdown(
        "Когда я чувствую скуку?",
        "# Ответ\n\nКогда нет движения.",
    )

    assert path.parent == answers_dir
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{6}\.md", path.name)
    assert path.name.endswith(".md")
    assert "kogda" not in path.name.lower()
    content = path.read_text(encoding="utf-8")
    assert "# Ask History Answer" in content
    assert "## Question" in content
    assert "Когда я чувствую скуку?" in content
    assert "## Answer" in content
    assert "Когда нет движения." in content


def test_format_telegram_answer_returns_full_answer_when_it_fits():
    answer = "# Ответ\n\nКоротко."

    assert format_telegram_answer(answer) == answer


def test_format_telegram_answer_truncates_long_answer():
    answer = "а" * (TELEGRAM_MESSAGE_LIMIT + 200)

    message = format_telegram_answer(answer)

    assert len(message) <= TELEGRAM_MESSAGE_LIMIT
    assert message.startswith(LONG_ANSWER_NOTICE)
    assert message.endswith("...")


def test_load_telegram_token_requires_env(monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", Path("missing_test_project_root"))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(TelegramBotConfigError, match="TELEGRAM_BOT_TOKEN"):
        _load_telegram_token()


def test_handle_text_question_rejects_when_lock_is_busy(monkeypatch):
    called = False

    def fake_ask_history(question):
        nonlocal called
        called = True
        return "never"

    monkeypatch.setattr("diarylens.telegram_bot.ask_history", fake_ask_history)

    async def scenario():
        message = FakeMessage("Когда я чувствую скуку?")
        async with __import__("diarylens.telegram_bot").telegram_bot.ASK_HISTORY_LOCK:
            await handle_text_question(FakeUpdate(message), None)
        return message

    message = asyncio.run(scenario())

    assert called is False
    assert message.text_replies == [BUSY_MESSAGE]
    assert message.document_replies == []


def test_handle_text_question_calls_ask_history_and_sends_document(monkeypatch):
    answers_dir = Path("data") / "answers" / "test_telegram_bot"
    captured = {}

    monkeypatch.setattr(config, "ASK_HISTORY_ANSWERS_DIR", answers_dir)

    def fake_ask_history(question):
        captured["question"] = question
        return "# Ответ\n\nГотово."

    monkeypatch.setattr("diarylens.telegram_bot.ask_history", fake_ask_history)

    async def scenario():
        message = FakeMessage("  Когда я чувствую скуку?  ")
        await handle_text_question(FakeUpdate(message), None)
        return message

    message = asyncio.run(scenario())

    assert captured["question"] == "Когда я чувствую скуку?"
    assert message.text_replies == ["# Ответ\n\nГотово."]
    assert len(message.document_replies) == 1
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{6}\.md",
        message.document_replies[0],
    )
    assert list(answers_dir.glob("*.md"))