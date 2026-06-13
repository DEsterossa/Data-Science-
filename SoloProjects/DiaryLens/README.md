# DiaryLens

**MVP v0.1 — local weekly review pipeline for a personal diary.**

## What is DiaryLens?

DiaryLens — локальный pipeline для еженедельной аналитики личного дневника.

Он превращает PDF дневника за неделю в:

- structured **daily JSON** (по одному файлу на день);
- **weekly JSON** с evidence-backed claims;
- readable **weekly review report** в Markdown.

Все промежуточные артефакты сохраняются на диск — pipeline можно дебажить по слоям и сравнивать недели между собой.

## Problem

Обычный ChatGPT-чат плохо подходит для регулярного weekly review:

- плохо сохраняет структуру;
- смешивает факты и интерпретации;
- может забывать evidence;
- трудно сравнивать недели;
- трудно проверять качество.

DiaryLens решает это через воспроизводимый file-based pipeline и evidence-based JSON layers.

## Core principle

```text
diary → structure → evidence-backed claims → weekly review
```

## Current MVP pipeline

```text
PDF
→ raw markdown
→ clean markdown
→ days md
→ daily JSON
→ verified daily JSON
→ weekly JSON
→ weekly_report.md
```

Verification — опциональный второй LLM-проход для daily JSON (включён по умолчанию в `run-week`).

## Quick start

### Install

```bash
cd DiaryLens
pip install -e .
```

### Configure `.env`

```bash
cp .env.example .env
# fill GIGACHAT_AUTH_KEY and model names
```

Подробнее: [docs/pipeline.md](docs/pipeline.md#environment-variables).

### Run full pipeline

```bash
diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/2026-W22.pdf
```

Полезные флаги:

```bash
diarylens run-week --week-id 2026-W22 --pdf path/to/week.pdf --no-verify
diarylens run-week --week-id 2026-W22 --pdf path/to/week.pdf --force
```

### Test LLM connectivity

```bash
diarylens test-llm --model daily
```

## Main outputs

| Artifact | Path |
| -------- | ---- |
| Raw text | `data/interim/raw_text/{week_id}_raw.md` |
| Clean text | `data/interim/clean_text/{week_id}_clean.md` |
| Daily markdown | `data/processed/days_md/YYYY-MM-DD.md` |
| Day manifest | `data/processed/day_manifests/{week_id}_days.json` |
| Daily JSON | `data/processed/days_json/YYYY-MM-DD.json` |
| Failed daily artifacts | `data/processed/days_json_failed/YYYY-MM-DD_error.json` |
| Weekly JSON | `data/processed/weeks_json/{week_id}.json` |
| Weekly report | `data/reports/{week_id}_weekly_report.md` |

## JSON layers

**Daily JSON** — source-backed structure одного дня: цитаты, сигналы (учёба, эмоции, социальное и т.д.), `short_summary`. Каждый item содержит `quote` и опциональный `note`.

**Weekly JSON** — evidence-backed aggregation нескольких дней: weekly claims (`summary`) + массив `evidence` с привязкой к дате, daily JSON и source field.

**weekly_report.md** — readable weekly review для человека, собранный из weekly JSON (и опционального user context).

## Quality control

Чеклист: [docs/quality_checklist.md](docs/quality_checklist.md)

Качество проверяется на трёх уровнях:

1. **daily JSON** — цитаты, scope, отсутствие галлюцинаций;
2. **weekly JSON** — evidence, повторяемость тем, корректный scope;
3. **weekly report** — конкретность, сохранение tensions, полезные next steps.

## Evaluation

Human-written expected checks для eval weeks:

```text
data/eval/weeks/{week_id}/expected_checks.md
```

Подробнее: [docs/evaluation.md](docs/evaluation.md)

Текущие eval weeks: `2026-W21`, `2026-W22`, `2026-W23`.

## Documentation

| Document | Description |
| -------- | ----------- |
| [docs/architecture.md](docs/architecture.md) | Design goals, layers, data contracts |
| [docs/pipeline.md](docs/pipeline.md) | CLI commands, workflow, debugging |
| [docs/evaluation.md](docs/evaluation.md) | Manual quality evaluation |
| [docs/quality_checklist.md](docs/quality_checklist.md) | Step-by-step quality checklist |

## Project structure

```text
src/diarylens/       Python package (pipeline modules + CLI)
prompts/             LLM prompt templates
data/
  raw/weekly/        Input PDFs
  interim/           Raw and clean markdown
  processed/         days_md, days_json, weeks_json, manifests
  reports/           Generated weekly reports
  context/           Optional goals and user context
  eval/weeks/        Evaluation expected checks
docs/                Architecture, pipeline, evaluation docs
tests/               pytest suite
```

## Tech stack

Реально используется в проекте:

- **Python 3.10+**
- **argparse** CLI (`diarylens` entry point)
- **Pydantic v2** — validation of daily / weekly JSON
- **GigaChat API** (`gigachat`) — LLM calls
- **pypdf** — PDF text extraction
- **json-repair** — tolerant JSON parsing from LLM responses
- **python-dotenv** — `.env` configuration
- **pytest** — tests
- **Markdown / JSON file-based pipeline** — no database

## Current limitations

- нет долгосрочной памяти между неделями;
- нет monthly aggregation;
- нет RAG / vector search;
- нет Telegram / web UI;
- качество зависит от LLM и промптов;
- sensitive diary data остаётся локально, кроме текстов, отправляемых в GigaChat API;
- report quality требует ручной проверки;
- partial week и failed days обрабатываются gracefully, но снижают полноту отчёта.

## Roadmap

### v0.1 — Core local pipeline

Mostly done:

- PDF → report pipeline;
- `run-week` command;
- daily verification pass;
- Pydantic schemas;
- quality checklist;
- eval case templates.

### v0.2 — Quality and evaluation

- больше eval weeks;
- стабильные quality checks;
- prompt tuning по ошибкам;
- regression cases.

### v0.3 — Memory

- month summaries;
- `long_term_patterns.md`;
- accept/reject memory updates.

### v0.4 — Product layer

- удобный интерфейс;
- возможно Telegram / web;
- export formats (PDF / DOCX).

## Development status

Проект находится в **MVP v0.1 / stabilization stage**: core pipeline работает, идёт ручная оценка качества на eval weeks и точечный prompt tuning.

## Tests

```bash
pytest
```

## Git and sensitive data

Personal diary data **не должна попадать в public git**.

- `.gitignore` исключает `.env`, PDF, pipeline outputs, reports;
- `data/eval/**/expected_checks.md` тоже игнорируется (личные заметки);
- структура каталогов сохраняется через `.gitkeep` — см. [data/README.md](data/README.md).

Перед push проверьте:

```bash
git status
```

Не коммитьте `.env` и файлы из `data/raw/`, `data/processed/`, `data/reports/`.
