"""Telegram bot interface for ask-history questions."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from diarylens import config as project_config
from diarylens.ask_history import ask_history

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_MESSAGE_LIMIT = 3500
ASK_HISTORY_LOCK = asyncio.Lock()

START_MESSAGE = (
    "Привет. Пришли мне вопрос по дневнику обычным текстом, "
    "а я отвечу через ask-history и отправлю полный ответ .md файлом."
)
HELP_MESSAGE = (
    "Отправь вопрос обычным сообщением. Например: Когда я чувствую скуку?\n"
    "Пока бот обрабатывает один вопрос, новые вопросы не ставятся в очередь."
)
BUSY_MESSAGE = (
    "Я уже обрабатываю предыдущий вопрос. Дождись ответа и отправь следующий."
)
EMPTY_MESSAGE = "Пришли вопрос обычным текстовым сообщением."
ERROR_MESSAGE = "Не получилось обработать вопрос. Ошибка сохранена в логах."
DOCUMENT_ERROR_MESSAGE = (
    "Ответ отправлен текстом, но не получилось отправить .md файл."
)
LONG_ANSWER_NOTICE = (
    "Вот краткий preview ответа. Полный ответ отправил отдельным .md файлом.\n\n"
)

logger = logging.getLogger(__name__)


class TelegramBotConfigError(Exception):
    """Raised when Telegram bot startup configuration is invalid."""


def _load_telegram_token() -> str:
    load_dotenv(project_config.PROJECT_ROOT / ".env", override=True)
    token = os.environ.get(TELEGRAM_BOT_TOKEN_ENV, "").strip()
    if not token:
        raise TelegramBotConfigError(
            f"{TELEGRAM_BOT_TOKEN_ENV} environment variable is not set"
        )
    return token


def _short_hash(question: str, answer: str, created_at: datetime) -> str:
    payload = f"{created_at.isoformat()}\n{question}\n{answer}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:6]


def save_ask_history_answer_markdown(question: str, answer: str) -> Path:
    """Save a Telegram-facing ask-history answer as markdown."""
    answers_dir = project_config.ASK_HISTORY_ANSWERS_DIR
    answers_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now()
    filename = (
        f"{created_at.strftime('%Y-%m-%d_%H%M%S')}_"
        f"{_short_hash(question, answer, created_at)}.md"
    )
    output_path = answers_dir / filename
    content = (
        "# Ask History Answer\n\n"
        "## Question\n\n"
        f"{question.strip()}\n\n"
        "## Answer\n\n"
        f"{answer.rstrip()}\n"
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def format_telegram_answer(answer: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> str:
    """Return the full answer or a safe Telegram preview."""
    if len(answer) <= limit:
        return answer

    preview_limit = max(0, limit - len(LONG_ANSWER_NOTICE) - 3)
    preview = answer[:preview_limit].rstrip()
    return f"{LONG_ANSWER_NOTICE}{preview}..."


async def start(update: Any, context: Any) -> None:
    """Handle /start."""
    del context
    if update.message is not None:
        await update.message.reply_text(START_MESSAGE)


async def help_command(update: Any, context: Any) -> None:
    """Handle /help."""
    del context
    if update.message is not None:
        await update.message.reply_text(HELP_MESSAGE)


async def handle_text_question(update: Any, context: Any) -> None:
    """Handle normal text messages as ask-history questions."""
    del context
    message = update.message
    if message is None:
        return

    question = (message.text or "").strip()
    if not question:
        await message.reply_text(EMPTY_MESSAGE)
        return

    if ASK_HISTORY_LOCK.locked():
        await message.reply_text(BUSY_MESSAGE)
        return

    async with ASK_HISTORY_LOCK:
        try:
            answer = await asyncio.to_thread(ask_history, question)
            answer_path = await asyncio.to_thread(
                save_ask_history_answer_markdown,
                question,
                answer,
            )
        except Exception:
            logger.exception("Failed to process Telegram ask-history question")
            await message.reply_text(ERROR_MESSAGE)
            return

        await message.reply_text(format_telegram_answer(answer))

        try:
            with answer_path.open("rb") as document:
                await message.reply_document(
                    document=document,
                    filename=answer_path.name,
                )
        except Exception:
            logger.exception("Failed to send ask-history markdown document")
            await message.reply_text(DOCUMENT_ERROR_MESSAGE)


def build_application(token: str) -> Any:
    """Build the Telegram application with handlers."""
    try:
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError as exc:
        raise TelegramBotConfigError(
            "python-telegram-bot is not installed. "
            "Install project dependencies with: pip install -e ."
        ) from exc

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_question)
    )
    return application


def run_telegram_bot() -> None:
    """Start the Telegram bot polling loop."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    token = _load_telegram_token()
    application = build_application(token)
    logger.info("Starting DiaryLens Telegram bot polling")
    application.run_polling()