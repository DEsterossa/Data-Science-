# DiaryLens Architecture

## Overview

DiaryLens — локальный **file-based pipeline** для weekly review личного дневника.

PDF недели проходит через серию детерминированных и LLM-шагов. Каждый шаг сохраняет промежуточный артефакт на диск. Финальный output — Markdown report, но главная ценность MVP — **structured JSON layers** с evidence.

Нет базы данных, нет web server, нет очереди задач. Всё запускается через CLI на машине пользователя.

## Design goals

- **Reproducibility** — один и тот же вход + конфиг дают предсказуемую цепочку артефактов.
- **Evidence-first** — выводы привязаны к цитатам и source fields, а не к «ощущению модели».
- **Intermediate artifacts** — каждый слой можно inspect и перезапустить отдельно.
- **Manual quality control** — качество проверяется человеком по чеклисту и eval cases.
- **No over-analysis** — pipeline не ставит диагнозы и не строит «психологический портрет».
- **Local-first workflow** — данные дневника хранятся локально; наружу уходят только LLM prompts/responses.

## Pipeline architecture

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

Оркестрация: `diarylens run-week` (`src/diarylens/week_runner.py`).

## Layers

### Extraction layer

**Module:** `pdf_extractor.py`

PDF → raw markdown (`data/interim/raw_text/{week_id}_raw.md`).

Использует `pypdf`. Сохраняет page markers для отладки.

### Cleaning layer

**Module:** `text_cleaner.py`

Raw markdown → clean markdown (`data/interim/clean_text/{week_id}_clean.md`).

Нормализует пробелы, переносы, шум PDF extraction.

### Day splitting layer

**Module:** `day_splitter.py`

Clean markdown → daily markdown files + manifest.

Outputs:

- `data/processed/days_md/YYYY-MM-DD.md`
- `data/processed/day_manifests/{week_id}_days.json`

Парсит русские date headers (полные и сокращённые месяцы, например `июня` и `июн.`).

### Daily extraction layer

**Module:** `daily_extractor.py`

Daily markdown → daily JSON (`data/processed/days_json/YYYY-MM-DD.json`).

LLM + Pydantic schema `DailyExtraction`. Prompt: `prompts/daily_extraction.md`.

При ошибке LLM:

- сохраняется `data/processed/days_json_failed/YYYY-MM-DD_error.json`;
- создаётся empty placeholder daily JSON;
- pipeline продолжается для остальных дней.

### Verification layer

**Module:** `daily_extractor.py` (second pass)

Draft daily JSON → verified daily JSON (перезапись того же файла).

LLM сверяет JSON с исходным day text. Prompt: `prompts/daily_verification.md`.

Включено по умолчанию в `run-week`; отключается флагом `--no-verify`.

### Weekly aggregation layer

**Module:** `weekly_aggregator.py`

Daily JSONs → weekly JSON (`data/processed/weeks_json/{week_id}.json`).

LLM строит evidence-backed weekly claims. Prompt: `prompts/weekly_aggregation.md`.

Поддерживает partial week (`missing_days`) с warning.

### Report generation layer

**Module:** `report_generator.py`

Weekly JSON → `data/reports/{week_id}_weekly_report.md`.

LLM генерирует readable Markdown (без JSON parsing на выходе). Prompt: `prompts/weekly_report.md`.

Опционально подмешивает user context:

- `data/context/goals.md`
- `data/context/context_about_me.md`

## Data contracts

### Daily JSON

Один день, schema `DailyExtraction`:

| Field group | Purpose |
| ----------- | ------- |
| `date`, `week_id`, `source_day_md` | Identity and traceability |
| `important_moments`, `wins`, `tensions`, `emotions`, … | Quote-based lists (`ExtractedItem`: `quote` + optional `note`) |
| `short_summary` | Neutral one-day summary |

Principle: **source-backed fragments**, не свободный пересказ.

### Weekly JSON

Одна неделя, schema `WeeklyAggregation`:

| Field group | Purpose |
| ----------- | ------- |
| `week_id`, `start_date`, `end_date` | Week identity |
| `days_included`, `missing_days` | Coverage |
| `week_essence`, `main_events`, `main_tensions`, … | Lists of `WeeklyItem` |
| `short_summary` | Short week summary |

Each `WeeklyItem`:

- `summary` — weekly claim;
- `evidence[]` — fragments with `date`, `source_daily_json`, `source_field`, `quote`, `note`.

Principle: **no claim without evidence**.

### Evidence model

```text
daily quote → daily JSON field → weekly evidence → weekly claim → report prose
```

Scope rule: evidence из одного дня не должно автоматически становиться «темой всей недели».

## Why intermediate artifacts matter

- **Debug by layer** — можно увидеть, на каком шаге испортился результат.
- **Cheaper iteration** — не нужно каждый раз заново парсить PDF и звать LLM для всех шагов (`run-week` переиспользует существующие файлы без `--force`).
- **Evaluation** — daily / weekly JSON можно сравнивать с expected checks независимо от report.
- **Audit trail** — failed days сохраняются в `days_json_failed/`.

## LLM usage

| Step | LLM? | Validated by Pydantic? |
| ---- | ---- | ---------------------- |
| PDF extraction | No | — |
| Text cleaning | No | — |
| Day splitting | No | — |
| Daily extraction | Yes | Yes (`DailyExtraction`) |
| Daily verification | Yes | Yes (`DailyExtraction`) |
| Weekly aggregation | Yes | Yes (`WeeklyAggregation`) |
| Report generation | Yes | No (Markdown output) |

Models configured via `.env`: `GIGACHAT_MODEL_DAILY`, `GIGACHAT_MODEL_WEEKLY`, `GIGACHAT_MODEL_REPORT`.

JSON responses parsed with `json-repair` fallback for minor LLM formatting issues.

## Failure modes

| Failure | Where to look |
| ------- | ------------- |
| PDF extraction noise | `data/interim/raw_text/` |
| Day splitting errors (wrong day count) | `data/interim/clean_text/`, date header format |
| Missing daily JSON | `data/processed/day_manifests/`, `days_json/` |
| LLM timeout | CLI error, increase `GIGACHAT_TIMEOUT` |
| Invalid JSON from LLM | `days_json_failed/`, pytest-normalized weekly fields |
| LLM refusal (sensitive content) | `days_json_failed/` + empty placeholder |
| Hallucinated claims | weekly JSON evidence vs daily JSON |
| Weak / generic report | compare report vs weekly JSON |
| Partial week | `missing_days` in weekly JSON, stderr warnings |

## What is intentionally not included in v0.1

- RAG / retrieval over past weeks;
- vector DB (Qdrant, embeddings);
- long-term memory updates;
- Telegram bot / web UI;
- monthly / quarterly aggregation;
- psychological diagnosis or clinical labels;
- automatic benchmark scoring.

## Related docs

- [pipeline.md](pipeline.md) — commands and workflow
- [evaluation.md](evaluation.md) — manual quality evaluation
- [quality_checklist.md](quality_checklist.md) — detailed checklist
