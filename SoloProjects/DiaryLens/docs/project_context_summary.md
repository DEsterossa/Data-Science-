# DiaryLens Project Context Summary

Compact project memory for ChatGPT Project / continuation of development.  
Last aligned with codebase: **DiaryLens v0.1.0** (`pyproject.toml`).

---

## 1. Project identity

**DiaryLens / Journal Analyzer** — локальная система структурирования и поиска по личному дневнику (local-first searchable diary memory).

- **Цель:** превращать дневниковые PDF в day-level артефакты, строить day index и (в v0.2+) отвечать на вопросы к прошлому через retrieval с evidence.
- **Не ChatGPT-chat:** воспроизводимый file-based pipeline с промежуточными артефактами и Pydantic validation.
- **Не только weekly review:** weekly JSON и `weekly_report.md` — optional snapshot; главная единица памяти — **день**.

**Главный принцип:**

```text
raw diary → day-level structure → retrieval → answer with evidence
```

**Иерархия источников истины:**

```text
days_md          = source of truth (исходный дневной markdown)
days_json        = day index (события, темы, open loops, навигация)
weekly_json      = optional weekly snapshot (не главный memory layer)
weekly_report.md = optional readable snapshot недели
```

**Проект не должен превращаться в:**

- мотивационный self-help;
- психологический диагноз;
- бесконечный самоанализ;
- обычный чат без структуры и evidence.

Дополнительный контекст (vision, modes, memory plans): корневые файлы `00_project_identity.md`, `01_inputs_and_pipeline.md`, `02_report_format_and_rules.md`, `03_modes_and_memory.md`.

---

## 2. Current project stage

```text
MVP v0.1 / stabilization stage
```

**Статус:**

- Core weekly pipeline работает **end-to-end** через `diarylens run-week`.
- Day-level артефакты (`days_md`, `days_json`) — фундамент для будущего retrieval.
- Retrieval / `ask_history` **ещё не в коде** — запланировано v0.2.
- Есть unit tests (~144), docs, quality checklist, eval case folders.
- Код запушен в GitHub (`Data-Science-/SoloProjects/DiaryLens/`).
- Локальные pipeline outputs могут быть очищены; для нового прогона нужны PDF в `data/raw/weekly/`.
- Исторически успешно прогонялись недели **2026-W21, 2026-W22, 2026-W23** (eval cases есть; артефакты не обязаны лежать локально).

**Продуктовый сдвиг:**

```text
weekly_report.md = optional weekly snapshot
ask_history / retrieval over days = main future direction (v0.2+)
```

---

## 3. Current pipeline (v0.1)

```text
PDF
→ raw markdown
→ clean markdown
→ days md                    ← source of truth
→ daily JSON                 ← day index
→ verified daily JSON        (optional second LLM pass; default ON in run-week)
→ weekly JSON                ← optional snapshot
→ weekly_report.md           ← optional snapshot
```

**Главная команда:**

```bash
diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/2026-W22.pdf
```

**Флаги `run-week`:**

- `--no-verify` — без verification pass daily JSON
- `--force` — перезаписать существующие outputs (включая report)

**Итоговый output v0.1:**

```text
data/reports/{week_id}_weekly_report.md
```

**Поведение при ошибках daily LLM:** сохраняется `days_json_failed/{date}_error.json`, создаётся empty placeholder `days_json/{date}.json`, pipeline **продолжается** (warning, не fatal).

**Поведение без `--force`:** существующие intermediate files переиспользуются; если report уже есть — ошибка с hint использовать `--force`.

---

## 4. Existing CLI commands (v0.1, in code)

Реальные команды (`src/diarylens/cli.py`, argparse):

```bash
diarylens --help
diarylens --version

diarylens extract-pdf <pdf_path> --week-id 2026-W22
diarylens clean-text --week-id 2026-W22
diarylens split-days --week-id 2026-W22
diarylens extract-daily --week-id 2026-W22 [--verify]
diarylens aggregate-week --week-id 2026-W22
diarylens generate-report --week-id 2026-W22

diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/week.pdf [--no-verify] [--force]

diarylens test-llm [--model daily|weekly|report]
```

Entry point: `diarylens` → `diarylens.cli:main` (`pip install -e .`).

---

## 5. Planned CLI commands (v0.2+, not in code yet)

Добавлять после notebook/prototype проверки:

```bash
diarylens build-day-index
diarylens build-memory-index
diarylens search-memory "когда у меня появлялась скука"
diarylens ask-history "как менялось моё отношение к DiaryLens?"
```

---

## 6. Main directories and artifacts

| Path | Purpose |
|------|---------|
| `data/raw/weekly/` | Input PDFs |
| `data/raw/archive/` | Archived PDFs (optional; v0.3 backfill) |
| `data/interim/raw_text/` | `{week_id}_raw.md` |
| `data/interim/clean_text/` | `{week_id}_clean.md` |
| `data/processed/days_md/` | `YYYY-MM-DD.md` — **source of truth** |
| `data/processed/day_manifests/` | `{week_id}_days.json` |
| `data/processed/days_json/` | Daily JSON per day — **day index** |
| `data/processed/days_json_failed/` | Failed-day diagnostics |
| `data/processed/weeks_json/` | `{week_id}.json` — optional weekly snapshot |
| `data/reports/` | `{week_id}_weekly_report.md` — optional weekly snapshot |
| `data/context/` | Optional `goals.md`, `context_about_me.md` for report |
| `data/eval/weeks/` | Human `expected_checks.md` per week |
| `data/memory/` | **Planned v0.2** — retrieval index (see below) |
| `prompts/` | LLM prompt templates |
| `docs/` | Architecture, pipeline, evaluation, quality, this file |
| `tests/` | pytest suite |
| `src/diarylens/` | Python package |

**Planned `data/memory/` structure (v0.2, not in code yet):**

```text
data/memory/
  day_index.jsonl      — navigation layer from daily JSON
  chunks.jsonl         — indexable raw day markdown chunks
  embeddings.npy       — local embeddings
  metadata.jsonl       — embedding metadata
  search_results/      — optional debug output
```

**Root planning docs:** `00_project_identity.md`, `01_inputs_and_pipeline.md`, `02_report_format_and_rules.md`, `03_modes_and_memory.md`.

**Not used / removed:** legacy `reports/weekly/`, root `context/` (outputs live under `data/`).

---

## 7. Daily JSON schema

Source of truth: `src/diarylens/schemas.py` → `DailyExtraction`.

**Роль:** day index / navigation layer — не замена исходного `days_md`.

**Top-level fields (Pydantic):**

```text
date
week_id
source_day_md
important_moments
wins
tensions
emotions
body_energy_signals
study_signals
ml_ds_signals
social_signals
decisions
open_questions
key_quotes
short_summary
```

**Смысл полей:**

| Field | Meaning |
|-------|---------|
| `important_moments` | Главный слой покрытия дня |
| `wins` | Явный прогресс / положительный результат |
| `tensions` | Сомнения, скука, страх, сопротивление, friction |
| `emotions` | Эмоциональные сигналы без глубокого анализа |
| `body_energy_signals` | Сон, еда, спорт, усталость, здоровье |
| `study_signals` | Учёба |
| `ml_ds_signals` | ML / DS / код / проекты |
| `social_signals` | Реальные люди (не AI) |
| `decisions` | Явные решения / планы |
| `open_questions` | Незакрытые вопросы / дела |
| `key_quotes` | Сильные фразы автора |
| `short_summary` | Нейтральное резюме дня |

**Item shape (`ExtractedItem`):**

```json
{
  "quote": "...",
  "note": "..."
}
```

**Legacy (не использовать):** `facts`, `problems`, `energy_signals`, `health_signals` as separate daily fields.

**Weekly mapping note:** in weekly JSON evidence, daily `emotions` → `source_field: "emotional_signals"`, daily `open_questions` → `"open_loops"` (see `SOURCE_FIELD_ALIASES` in `weekly_aggregator.py`).

---

## 8. Weekly JSON schema

Source of truth: `WeeklyAggregation` in `schemas.py`.

**Роль:** optional weekly snapshot — не единственный источник для будущих ответов.

**Top-level fields:**

```text
type                    (literal "week")
week_id
start_date
end_date
days_included
missing_days
week_essence
main_events
main_wins
main_tensions
emotional_background
body_energy
study_and_projects
social_context
actual_focus
repeated_topics
important_contradictions
open_loops
risks_next_week
next_week_focus_candidates
what_not_to_do
short_summary
```

**WeeklyItem:**

```json
{
  "summary": "...",
  "evidence": [...]
}
```

**WeeklyEvidence:**

```json
{
  "date": "YYYY-MM-DD",
  "source_day_md": "data/processed/days_md/YYYY-MM-DD.md",
  "source_daily_json": "data/processed/days_json/YYYY-MM-DD.json",
  "source_field": "tensions",
  "quote": "...",
  "note": "..."
}
```

**Rules:**

- Every `WeeklyItem` needs non-empty `evidence`.
- `source_field` must be valid `WeeklySourceField` (see schemas); aliases normalized before validation.
- `source_daily_json` must not be null.
- Claim scope must not exceed evidence (1 day ≠ whole week).

**Normalization:** `normalize_weekly_aggregation_data()` repairs common LLM shape issues; missing `short_summary` filled from `week_essence` or `""`.

---

## 9. Prompts

| File | Role |
|------|------|
| `prompts/daily_extraction.md` | LLM: day md → daily JSON (day index) |
| `prompts/daily_verification.md` | LLM: conservative verify/fix daily JSON vs source text |
| `prompts/weekly_aggregation.md` | LLM: daily JSONs → weekly JSON (optional snapshot) |
| `prompts/weekly_report.md` | LLM: weekly JSON → markdown report (optional snapshot) |

**Report generator principle:** `report_generator.py` reads **weekly JSON** (+ optional `data/context/goals.md`, `data/context/context_about_me.md`). It does **not** re-analyze raw diary PDF/md.

---

## 10. LLM provider and env vars

**Provider:** GigaChat API (`gigachat` Python SDK). **Not OpenAI.**

**Env vars** (`.env` at project root, loaded with `python-dotenv`):

| Variable | Required | Notes |
|----------|----------|-------|
| `GIGACHAT_AUTH_KEY` | Yes | API credentials |
| `GIGACHAT_SCOPE` | No | Default `GIGACHAT_API_PERS` |
| `GIGACHAT_MODEL_DAILY` | Yes | Daily extraction + verification |
| `GIGACHAT_MODEL_WEEKLY` | Yes | Weekly aggregation |
| `GIGACHAT_MODEL_REPORT` | No | Report; fallback → weekly model |
| `GIGACHAT_TIMEOUT` | No | Seconds; default 120 |

Optional: `GIGACHAT_VERIFY_SSL_CERTS`, `GIGACHAT_CA_BUNDLE_FILE`.

**Known ops issue:** model `GigaChat-2` on personal/Lite subscription may return **402 Payment Required**; switch to models included in plan or top up quota. Test with `diarylens test-llm --model daily`.

---

## 11. Quality and evaluation

**Docs:**

- `docs/quality_checklist.md` — manual QA for daily / weekly / report
- `docs/evaluation.md` — eval workflow
- `docs/architecture.md`, `docs/pipeline.md`
- `README.md` — external / portfolio overview

**Eval cases:**

```text
data/eval/weeks/2026-W21/expected_checks.md   (filled, scored)
data/eval/weeks/2026-W22/expected_checks.md   (filled, scored)
data/eval/weeks/2026-W23/expected_checks.md   (template / TODO)
```

**Expected checks =** human criteria, not golden reports:

- must mention / must not say
- important evidence quotes
- scope checks
- report quality checks
- scores (Daily / Weekly / Report) + overall verdict

**Observations from filled eval (factual, from files):**

| Week | MVP pass | Scores (D/W/R) | Notes |
|------|----------|----------------|-------|
| W21 | yes | 6 / 6 / 7 | Needs more event detail; weak next steps & what-not-to-do |
| W22 | yes | 6 / 7 / 8 | Needs prompt tuning; improve or drop next steps / what-not-to-do |
| W23 | — | not scored | Template not filled |

Common weak spots: **next steps**, **what not to do**, **insufficient event depth** in reports.

---

## 12. Current known limitations

- Quality depends on LLM + prompts; manual checklist review required.
- No automatic benchmark scoring.
- **Retrieval / `ask_history` not implemented yet** (planned v0.2).
- No long-term memory between weeks.
- No monthly aggregation.
- No Telegram / web UI.
- Sensitive diary data stored locally in files; **text sent to GigaChat API** on LLM steps.
- Partial weeks and failed days supported but reduce report quality.
- `day_splitter` must handle Russian date headers (full + abbreviated months, e.g. `июн.`).
- GigaChat refusals / 402 / timeouts possible.
- Two CLI leak tests may fail if local manifest missing (test env issue, not production).

---

## 13. Roadmap

### v0.1 — PDF to Day-Level Artifacts + Optional Weekly Snapshot

Статус: core pipeline работает, идёт stabilization.

- `run-week` end-to-end;
- `days_md` + `days_json` как фундамент;
- optional weekly JSON + report;
- eval-driven prompt tuning.

### v0.2 — Day-First Retrieval Prototype

```text
days_md + days_json
→ day_index / chunks
→ local embeddings (embeddings.npy + cosine similarity)
→ search-memory
→ ask-history answer with evidence from raw day markdown
```

- локальный индекс без Qdrant;
- `data/memory/` artifacts;
- planned CLI: `build-day-index`, `build-memory-index`, `search-memory`, `ask-history`.

### v0.3 — Historical Backfill + Searchable Diary Memory

```text
archive PDFs (variable length, not 1 PDF = 1 week)
→ days_md + days_json
→ chunk index + embeddings
→ searchable diary memory over ~2 years
```

### v0.4 — Retrieval Modes and Pattern Analysis

Режимы: `ask_history`, `similar_days`, `similar_periods`, `open_loops`, `trend_check`, `weekly_snapshot`.

---

## 14. What not to suggest next

Unless user explicitly asks, **do not propose:**

- Telegram bot
- Web app / dashboard
- Qdrant / external vector DB as mandatory first step
- Fine-tuning
- Long-term memory updates
- Monthly aggregation
- PDF/DOCX export
- Complicated multi-service architecture
- Another LLM-summary JSON layer on top of daily/weekly JSON

**Planned and OK to discuss (v0.2):**

- simple **local** retrieval prototype (`data/memory/`, embeddings.npy, cosine similarity);
- `search-memory` / `ask-history` CLI commands;
- day-first retrieval over `days_md` + `days_json`.

**Current focus:**

```text
stabilize MVP v0.1 → eval-driven prompt tuning → prototype day-first retrieval (v0.2)
```

---

## 15. Cursor workflow preferences

- User (ChatGPT) formulates **precise tasks**; Cursor executes **small technical steps**.
- Tasks should include: context, goal, files, constraints, done criteria.
- Cursor must **not** rewrite the whole project or add technologies without request.
- Prefer minimal diffs; reuse existing modules (`pdf_extractor`, `daily_extractor`, `week_runner`, etc.).
- Do not duplicate pipeline logic in CLI vs runner.
- Do not log API keys.
- After changes: list modified files + test status (`pytest`).
- User prefers answers in **Russian** for chat; code/docs may be EN/RU mixed.
- Do not commit/push unless explicitly asked.

---

## 16. Code modules (quick map)

| Module | Responsibility |
|--------|----------------|
| `pdf_extractor.py` | PDF → raw md |
| `text_cleaner.py` | raw → clean md |
| `day_splitter.py` | clean → days md + manifest |
| `daily_extractor.py` | days md → daily JSON (+ verify) |
| `weekly_aggregator.py` | daily JSONs → weekly JSON |
| `report_generator.py` | weekly JSON → report md |
| `week_runner.py` | Orchestrate full pipeline |
| `llm_client.py` | GigaChat calls, model env resolution |
| `json_utils.py` | LLM JSON parse + repair |
| `schemas.py` | Pydantic models |
| `config.py` | Paths, `ensure_project_dirs()` |
| `cli.py` | argparse CLI |

**Tests:** `tests/test_*.py` (10 modules), run `pytest`.

**Tech stack:** Python 3.10+, Pydantic v2, pypdf, json-repair, pytest, argparse (not Typer).

---

## Compact handoff summary

DiaryLens is at **MVP v0.1 stabilization** with a **day-first product strategy**. Current code: `diarylens run-week` → `days_md` (source of truth), `days_json` (day index), optional weekly JSON + `data/reports/{week_id}_weekly_report.md`. Future main function: `ask_history` via local retrieval over days (v0.2). Daily JSON fields: `important_moments`, `wins`, `tensions`, `emotions`, `body_energy_signals`, `study_signals`, `ml_ds_signals`, `social_signals`, `decisions`, `open_questions`, `key_quotes`, `short_summary`. Weekly JSON is evidence-based: `WeeklyItem(summary, evidence)` with `WeeklyEvidence(date, source_day_md, source_daily_json, source_field, quote, note)`. GigaChat via `.env` model vars. Quality checklist + eval weeks W21–W23 exist (W23 template incomplete). Next: re-run pipeline, complete eval, tune weekly/report prompts, then prototype v0.2 retrieval (`data/memory/`, planned CLI commands). **Do not** jump to Qdrant, Telegram, or web app without explicit request.
