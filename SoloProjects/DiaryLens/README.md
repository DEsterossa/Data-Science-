# DiaryLens

DiaryLens - локальный pipeline для личного дневника: он превращает PDF в структурированные day-level артефакты и позволяет задавать вопросы к прошлым записям через `ask-history` с evidence.

Это не чат-бот. Проект устроен как воспроизводимый file-based workflow: каждый слой сохраняется на диск, поэтому результат можно проверять, дебажить и улучшать по шагам.

## Что делает проект

```text
PDF -> raw markdown -> clean markdown -> days_md -> daily JSON
                                      -> weekly JSON/report
                                      -> ask-history retrieval
```

Главная логика:

- `days_md` - источник истины по каждому дню;
- `days_json` - навигационный индекс по дням;
- weekly JSON/report - опциональный недельный snapshot;
- `ask-history` ищет релевантные raw-фрагменты и строит ответ с датами и цитатами.

## Быстрый старт

Установить проект локально:

```bash
pip install -e .
```

Создать `.env` и заполнить GigaChat credentials/model names:

```bash
cp .env.example .env
```

Проверить доступ к LLM:

```bash
diarylens test-llm --model daily
```

Запустить полный недельный pipeline:

```bash
diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/2026-W22.pdf
```

Полезные флаги:

```bash
diarylens run-week --week-id 2026-W22 --pdf path/to/week.pdf --no-verify
diarylens run-week --week-id 2026-W22 --pdf path/to/week.pdf --force
```

## Недельный отчёт

Функция недельного отчёта остаётся доступной. Она собирает `weekly JSON` из daily JSON и генерирует читаемый markdown-отчёт:

```bash
diarylens aggregate-week --week-id 2026-W22
diarylens generate-report --week-id 2026-W22
```

Полный `run-week` тоже включает генерацию недельного отчёта. Итоговый файл сохраняется сюда:

```text
data/reports/{week_id}_weekly_report.md
```

## Ask History

Найти релевантные фрагменты без LLM-ответа:

```bash
diarylens search-memory "Когда я чувствую скуку?"
```

Задать вопрос дневнику и получить markdown-ответ:

```bash
diarylens ask-history "Когда я чувствую скуку?"
diarylens ask-history "Когда я чувствую скуку?" --debug
```

`search-memory` показывает найденные raw evidence chunks. `ask-history` отправляет эти фрагменты в LLM и сохраняет ответ сюда:

```text
data/memory/ask_history_answers/
```

## Основные артефакты

| Артефакт | Путь |
| --- | --- |
| Raw text | `data/interim/raw_text/{week_id}_raw.md` |
| Clean text | `data/interim/clean_text/{week_id}_clean.md` |
| Daily markdown | `data/processed/days_md/YYYY-MM-DD.md` |
| Day manifest | `data/processed/day_manifests/{week_id}_days.json` |
| Daily JSON | `data/processed/days_json/YYYY-MM-DD.json` |
| Failed daily extraction | `data/processed/days_json_failed/YYYY-MM-DD_error.json` |
| Weekly JSON | `data/processed/weeks_json/{week_id}.json` |
| Weekly report | `data/reports/{week_id}_weekly_report.md` |
| Memory cache | `data/memory/` |

## Команды

```bash
diarylens extract-pdf --week-id 2026-W22 --pdf path/to/week.pdf
diarylens clean-text --week-id 2026-W22
diarylens split-days --week-id 2026-W22
diarylens extract-daily --week-id 2026-W22
diarylens aggregate-week --week-id 2026-W22
diarylens generate-report --week-id 2026-W22
diarylens run-week --week-id 2026-W22 --pdf path/to/week.pdf
diarylens search-memory "question"
diarylens ask-history "question"
diarylens test-llm --model daily
```

## Структура проекта

```text
src/diarylens/       Python package и CLI
prompts/             LLM prompt templates
data/                локальные inputs, outputs, reports, memory cache
docs/                архитектура, pipeline, evaluation notes
tests/               pytest suite
experiments/         локальные retrieval-эксперименты, не для public git
```

## Стек

- Python 3.10+
- argparse CLI
- Pydantic v2
- GigaChat API
- sentence-transformers
- numpy
- pypdf
- json-repair
- pytest

## Безопасность данных

Дневниковые данные чувствительные. `.gitignore` исключает `.env`, raw PDF, processed diary outputs, reports, memory cache, логи и локальные эксперименты.

LLM-шаги всё равно отправляют выбранный текст в GigaChat. Для `ask-history` это значит, что найденные дневниковые фрагменты уходят во внешний API при генерации ответа.

Перед публикацией проверь:

```bash
git status
```

Не коммить `.env` и личные файлы из `data/raw/`, `data/processed/`, `data/reports/`, `data/memory/` и `experiments/`.

## Документация

- [docs/pipeline.md](docs/pipeline.md) - команды, workflow, environment variables
- [docs/ask_history.md](docs/ask_history.md) - retrieval architecture и CLI
- [docs/architecture.md](docs/architecture.md) - дизайн проекта и data contracts
- [docs/evaluation.md](docs/evaluation.md) - ручная оценка качества
- [docs/quality_checklist.md](docs/quality_checklist.md) - quality checks

## Тесты

```bash
pytest
```

## Telegram Bot

`diarylens telegram-bot` starts a small Telegram interface for `ask-history`.
The bot accepts normal text messages as diary-history questions, calls the existing `ask_history` pipeline, sends the answer back in chat, and also sends the full answer as a `.md` document. Local copies are saved under:

```text
data/answers/ask_history/
```

Bash:

```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
diarylens telegram-bot
```

PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
diarylens telegram-bot
```

While one question is being processed, extra messages are rejected instead of queued.
