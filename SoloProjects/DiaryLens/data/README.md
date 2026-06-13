# Data directory

Локальные данные DiaryLens. **Не коммитьте personal diary outputs в public git.**

## Structure

```text
data/
  raw/weekly/          Input PDFs
  raw/archive/         Archived PDFs (optional)
  interim/raw_text/    Extracted raw markdown
  interim/clean_text/  Cleaned markdown
  processed/days_md/   Daily markdown files
  processed/day_manifests/
  processed/days_json/
  processed/days_json_failed/
  processed/weeks_json/
  reports/             Generated weekly reports
  context/             Optional goals.md, context_about_me.md
  eval/weeks/          Manual evaluation expected_checks.md
```

Pipeline создаёт папки автоматически (`diarylens run-week` или любая CLI-команда).

## Git

См. корневой `.gitignore`:

- PDF, interim, processed outputs и reports **игнорируются**;
- `expected_checks.md` в eval **игнорируется** (может содержать личные заметки);
- `.gitkeep` файлы сохраняют структуру каталогов в репозитории.

Для portfolio можно закоммитить anonymized sample outputs отдельно, если нужно.
