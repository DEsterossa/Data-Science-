# DiaryLens Evaluation

## Goal

Evaluation в DiaryLens MVP v0.1 нужна **не для автоматического benchmark score**, а для **ручной проверки качества** LLM pipeline.

Цель — понять:

- можно ли доверять weekly review;
- на каком слое (daily / weekly / report) возникла проблема;
- нужен ли prompt tuning, schema fix или extraction fix.

## Evaluation layers

```text
daily JSON quality
→ weekly JSON quality
→ weekly report quality
```

Не начинай оценку с финального report. Если daily JSON слабый, weekly JSON и report почти наверняка искажены.

## Quality checklist

Основной инструмент: [quality_checklist.md](quality_checklist.md)

Содержит:

- evidence-first principles;
- scope rules;
- red flags;
- debug decision tree;
- scoring guidance.

## Eval cases

Структура:

```text
data/eval/weeks/{week_id}/expected_checks.md
```

Один файл на eval week. Это **human-written expected checks**, а не эталонный отчёт и не ground truth JSON.

## What expected_checks.md is

Файл помогает проверить generated report по заранее заданным ожиданиям пользователя.

Typical sections:

| Section | Purpose |
| ------- | ------- |
| **Must mention** | Темы, которые отчёт обязан отразить |
| **Must not say** | Неверные обобщения, галлюцинации, generic advice |
| **Important evidence to preserve** | Даты + цитаты, которые нельзя потерять |
| **Scope checks** | Нет over-generalization с одного дня на всю неделю |
| **Report quality checks** | Конкретность, tensions, next steps |
| **Score** | Daily / Weekly / Report scores + overall verdict |
| **Notes** | Заметки после проверки |

Template ships with `TODO` placeholders — пользователь заполняет содержание вручную после прочтения дневника и report.

**Git:** файлы `expected_checks.md` в `.gitignore` (могут содержать личные цитаты). В репозитории остаётся `data/eval/weeks/README.md` и шаблон секций в этом документе.

## Current eval weeks

Существуют case folders:

```text
data/eval/weeks/2026-W21/expected_checks.md
data/eval/weeks/2026-W22/expected_checks.md
data/eval/weeks/2026-W23/expected_checks.md
```

Corresponding pipeline outputs (when processed):

```text
data/processed/weeks_json/2026-W21.json
data/processed/weeks_json/2026-W22.json
data/processed/weeks_json/2026-W23.json
data/reports/2026-W21_weekly_report.md
data/reports/2026-W22_weekly_report.md
data/reports/2026-W23_weekly_report.md
```

Scores and pass/fail status vary by week and are filled by the user in each `expected_checks.md`. Documentation does not prescribe final grades.

## Scoring

MVP uses **manual scoring** in `expected_checks.md`:

```text
Daily JSON score: TODO
Weekly JSON score: TODO
Report score: TODO
```

Recommended scales (choose one and stay consistent):

| Scale | Meaning |
| ----- | ------- |
| **A / B / C / D** | Quick pass / acceptable / weak / fail |
| **1–10** | Finer granularity for portfolio notes |

Overall verdict checkboxes:

- MVP pass
- Needs prompt tuning
- Needs schema change
- Needs extraction fix

## How to use evaluation

1. Run week pipeline:

   ```bash
   diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/2026-W22.pdf
   ```

2. Open generated report: `data/reports/{week_id}_weekly_report.md`.

3. Open weekly JSON: `data/processed/weeks_json/{week_id}.json`.

4. Spot-check daily JSON for key days: `data/processed/days_json/YYYY-MM-DD.json`.

5. Compare report with `data/eval/weeks/{week_id}/expected_checks.md`.

6. Fill scores and notes; mark status checkboxes.

7. Decide next action:

   | Verdict | Typical next step |
   | ------- | ----------------- |
   | MVP pass | Add more eval weeks |
   | Needs prompt tuning | Edit relevant prompt in `prompts/` |
   | Needs schema change | Update Pydantic schema + prompts |
   | Needs extraction fix | Fix PDF / splitter / cleaner code |

8. Cross-check with [quality_checklist.md](quality_checklist.md) for consistency.

## Known current issues (MVP v0.1)

Осторожно — не всё из этого полностью решено:

- некоторые important events могут быть недостаточно раскрыты в report;
- `next_week_focus_candidates` / `what_not_to_do` иногда слабые или generic;
- weekly report может быть слишком кратким относительно богатого weekly JSON;
- weekly aggregation иногда требует prompt tuning (missing fields, weak evidence);
- LLM refusal на отдельных днях даёт empty placeholder daily JSON;
- abbreviated date headers в PDF (`июн.`) ранее ломали day split — fix есть, но новые форматы дат возможны;
- report quality still requires manual review — no auto-pass.

## What evaluation is not

- Not automated CI benchmark (yet).
- Not a golden dataset of perfect reports.
- Not a substitute for reading the original diary when in doubt.

## Related docs

- [quality_checklist.md](quality_checklist.md)
- [pipeline.md](pipeline.md)
- [architecture.md](architecture.md)
