# DiaryLens Pipeline

Практическое руководство: как запустить pipeline, что он создаёт и как дебажить.

## Full pipeline command

```bash
diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/2026-W22.pdf
```

Flags:

| Flag | Description |
| ---- | ----------- |
| `--week-id` | Week identifier, e.g. `2026-W22` (required) |
| `--pdf` | Path to weekly PDF (required) |
| `--no-verify` | Skip daily JSON verification pass |
| `--force` | Overwrite existing pipeline outputs including final report |

Behavior without `--force`:

- existing intermediate files are reused;
- if final report already exists, command fails with hint to use `--force`.

## Step-by-step commands

Каждый шаг можно запустить отдельно:

```bash
diarylens extract-pdf data/raw/weekly/week.pdf --week-id 2026-W22
diarylens clean-text --week-id 2026-W22
diarylens split-days --week-id 2026-W22
diarylens extract-daily --week-id 2026-W22 --verify
diarylens aggregate-week --week-id 2026-W22
diarylens generate-report --week-id 2026-W22
```

Дополнительно:

```bash
diarylens test-llm --model daily
diarylens test-llm --model weekly
diarylens test-llm --model report
diarylens --version
diarylens --help
```

## Pipeline steps (detailed)

| # | Step | Command | Output |
| - | ---- | ------- | ------ |
| 1 | PDF extraction | `extract-pdf` | `data/interim/raw_text/{week_id}_raw.md` |
| 2 | Text cleaning | `clean-text` | `data/interim/clean_text/{week_id}_clean.md` |
| 3 | Day splitting | `split-days` | `data/processed/days_md/YYYY-MM-DD.md`, `data/processed/day_manifests/{week_id}_days.json` |
| 4 | Daily extraction | `extract-daily` | `data/processed/days_json/YYYY-MM-DD.json` |
| 5 | Daily verification | `extract-daily --verify` | overwrites daily JSON |
| 6 | Weekly aggregation | `aggregate-week` | `data/processed/weeks_json/{week_id}.json` |
| 7 | Report generation | `generate-report` | `data/reports/{week_id}_weekly_report.md` |

On daily extraction failure for a day:

- `data/processed/days_json_failed/YYYY-MM-DD_error.json` — diagnostics;
- `data/processed/days_json/YYYY-MM-DD.json` — empty placeholder;
- other days continue processing.

## Inputs

### Required

- **Weekly PDF** — typically placed in `data/raw/weekly/`.

### Optional context (report generation)

- `data/context/goals.md`
- `data/context/context_about_me.md`

If missing, report is generated without that context block.

### Prompts (bundled)

- `prompts/daily_extraction.md`
- `prompts/daily_verification.md`
- `prompts/weekly_aggregation.md`
- `prompts/weekly_report.md`

## Outputs

| Stage | Output |
| ----- | ------ |
| PDF extraction | `data/interim/raw_text/{week_id}_raw.md` |
| Cleaning | `data/interim/clean_text/{week_id}_clean.md` |
| Day splitting | `data/processed/days_md/YYYY-MM-DD.md`, `data/processed/day_manifests/{week_id}_days.json` |
| Daily extraction | `data/processed/days_json/YYYY-MM-DD.json` |
| Failed daily extraction | `data/processed/days_json_failed/YYYY-MM-DD_error.json` |
| Weekly aggregation | `data/processed/weeks_json/{week_id}.json` |
| Report generation | `data/reports/{week_id}_weekly_report.md` |

## Recommended workflow

1. Put PDF into `data/raw/weekly/`.
2. Configure `.env` (GigaChat credentials and models).
3. Run `diarylens run-week --week-id ... --pdf ...`.
4. Read `data/reports/{week_id}_weekly_report.md`.
5. Inspect `data/processed/weeks_json/{week_id}.json` if report looks wrong.
6. For eval weeks, fill or update `data/eval/weeks/{week_id}/expected_checks.md`.
7. Use [quality_checklist.md](quality_checklist.md) for structured review.

## Debugging

Если report плохой, иди **снизу вверх** по pipeline:

| Symptom | Likely layer | Action |
| ------- | ------------ | ------ |
| Report bad, weekly JSON good | Report prompt | Tune `prompts/weekly_report.md` |
| Weekly JSON bad, daily JSON good | Weekly aggregation | Tune `prompts/weekly_aggregation.md` |
| Daily JSON bad | Daily extraction / verification | Tune `prompts/daily_extraction.md` or `daily_verification.md` |
| Wrong day count / missing days | Day splitter | Check date headers in clean text; see `day_splitter.py` |
| Garbled text / wrong page breaks | PDF extraction / cleaning | Check raw and clean markdown |
| One day failed | LLM refusal / timeout | Check `days_json_failed/` |

Re-run from a specific step:

```bash
diarylens split-days --week-id 2026-W22
diarylens extract-daily --week-id 2026-W22 --verify --force   # if force added manually via re-extraction
diarylens aggregate-week --week-id 2026-W22
```

Or full re-run:

```bash
diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/week.pdf --force
```

## Environment variables

Used by `src/diarylens/llm_client.py`:

| Variable | Required | Description |
| -------- | -------- | ----------- |
| `GIGACHAT_AUTH_KEY` | Yes | GigaChat API credentials |
| `GIGACHAT_SCOPE` | No | Default: `GIGACHAT_API_PERS` |
| `GIGACHAT_MODEL_DAILY` | Yes | Model for daily extraction / verification |
| `GIGACHAT_MODEL_WEEKLY` | Yes | Model for weekly aggregation |
| `GIGACHAT_MODEL_REPORT` | No | Model for report; falls back to `GIGACHAT_MODEL_WEEKLY` |
| `GIGACHAT_TIMEOUT` | No | Request timeout in seconds (default: `120`) |

Optional:

| Variable | Description |
| -------- | ----------- |
| `GIGACHAT_VERIFY_SSL_CERTS` | SSL verification (default: `false`) |
| `GIGACHAT_CA_BUNDLE_FILE` | Custom CA bundle path |

Loaded from project root `.env` via `python-dotenv`.

## Common errors

### Timeout

```text
GigaChat request failed: ...
```

Increase `GIGACHAT_TIMEOUT` in `.env` (e.g. `300`).

### Invalid JSON

```text
LLM JSON failed validation: ...
```

Check LLM raw output indirectly via failed artifacts or re-run the step. Weekly aggregator applies some normalization (e.g. missing `short_summary`).

### Missing daily JSON

Weekly aggregation may report `missing_days` or fail if no valid daily JSON exists at all.

Check:

- `data/processed/day_manifests/{week_id}_days.json`
- `data/processed/days_json/`
- `data/processed/days_json_failed/`

### Missing weekly JSON

Report generation requires `data/processed/weeks_json/{week_id}.json`. Run `aggregate-week` first.

### Report already exists

```text
Error: report already exists: data/reports/2026-W22_weekly_report.md
Use --force to overwrite.
```

Use `--force` on `run-week` or delete the report manually.

### PDF not found

```text
Error: PDF not found: path/to/file.pdf
```

Check path passed to `--pdf`.

## Related docs

- [architecture.md](architecture.md)
- [evaluation.md](evaluation.md)
- [quality_checklist.md](quality_checklist.md)
