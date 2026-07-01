# DiaryLens — входные данные и пайплайн анализа

## 1. Основные входные данные

### Обычная текущая неделя

В текущем MVP пользователь обычно загружает **один PDF-файл**:

```text
data/raw/weekly/
  2026-W23.pdf
```

Внутри этого PDF обычно находится дневник за одну неделю.

Главная команда текущего кода:

```bash
diarylens run-week --week-id 2026-W23 --pdf data/raw/weekly/2026-W23.pdf
```

Текущий pipeline уже умеет:

```text
weekly PDF
→ raw markdown
→ clean markdown
→ days md
→ daily JSON
→ verified daily JSON
→ weekly JSON
→ weekly_report.md
```

Но новая архитектурная рамка:

```text
days_md = source of truth
days_json = day index / important events / topics
weekly_json = optional weekly snapshot, not main memory layer
```

---

### Исторический архив

Исторический архив будет загружаться **несколькими PDF-файлами**.

Важно: у этих PDF может быть **неопределённая длительность по дням**.

Один архивный PDF может содержать:

- несколько дней;
- одну неделю;
- несколько недель;
- часть месяца;
- другой нерегулярный диапазон.

Пример:

```text
data/raw/archive/
  diary_export_2024_part_01.pdf
  diary_export_2024_part_02.pdf
  diary_export_2025_spring.pdf
  diary_export_2025_2026_mixed.pdf
```

Поэтому historical backfill не должен предполагать, что:

```text
1 PDF = 1 неделя
```

Правильнее считать так:

```text
1 PDF = произвольный контейнер дневниковых записей
```

Главная цель historical backfill — получить day-level артефакты и searchable diary memory.

---

## 2. Постоянный контекст

Опциональные контекстные файлы:

```text
data/context/
  goals.md
  context_about_me.md
```

Они могут использоваться для weekly snapshot или answer generation, но не должны заменять evidence из дневника.

Если файлов нет, pipeline не должен падать.

---

## 3. Рабочие форматы

PDF — raw source.

Markdown / text — внутренний рабочий формат и основной источник истины после extraction.

JSON — структурированное представление для навигации и поиска.

```text
PDF
→ raw markdown
→ clean markdown
→ daily markdown
→ daily JSON as day index
→ optional weekly JSON
→ retrieval / ask_history
```

Не стоит анализировать PDF напрямую на всех этапах.

Лучше один раз извлечь текст, сохранить промежуточные результаты и дальше работать с ними.

Главное правило:

> Для ответов на вопросы и evidence-based reasoning использовать raw / clean day markdown, а не только сжатые JSON.

---

## 4. Актуальная структура файлов v0.1

```text
data/
  raw/
    weekly/
      2026-W23.pdf
    archive/
      diary_export_2024_part_01.pdf

  interim/
    raw_text/
      2026-W23_raw.md
    clean_text/
      2026-W23_clean.md

  processed/
    days_md/
      2026-06-01.md
      2026-06-02.md

    day_manifests/
      2026-W23_days.json

    days_json/
      2026-06-01.json
      2026-06-02.json

    days_json_failed/
      2026-06-03_error.json

    weeks_json/
      2026-W23.json

  reports/
```

Итоговые weekly snapshots сохраняются в:

```text
data/reports/{week_id}_weekly_report.md
```

Legacy-путь вида `reports/weekly/` не является актуальным output path для текущего MVP.

---

## 5. Будущая структура retrieval memory

Для `ask_history` нужен отдельный retrieval layer:

```text
data/
  memory/
    day_index.jsonl
    chunks.jsonl
    embeddings.npy
    metadata.jsonl
    search_results/
```

Важно: это не должен быть новый аналитический пересказ дневника.

`chunks.jsonl` хранит индексируемые куски исходного дневника:

```json
{
  "doc_id": "day_chunk:2026-06-13:003",
  "type": "raw_day_chunk",
  "text": "исходный фрагмент дневника...",
  "metadata": {
    "date": "2026-06-13",
    "week_id": "2026-W24",
    "source_day_md": "data/processed/days_md/2026-06-13.md",
    "chunk_index": 3
  }
}
```

`day_index.jsonl` хранит навигационные признаки дня:

```json
{
  "date": "2026-06-13",
  "week_id": "2026-W24",
  "source_day_md": "data/processed/days_md/2026-06-13.md",
  "source_daily_json": "data/processed/days_json/2026-06-13.json",
  "important_events": [],
  "topics": [],
  "open_questions": [],
  "embedding_text": ""
}
```

---

## 6. Что делает система внутри: v0.1

### Шаг 1. PDF text extraction

Система берёт PDF недели:

```text
data/raw/weekly/2026-W23.pdf
```

И извлекает raw markdown:

```text
data/interim/raw_text/2026-W23_raw.md
```

---

### Шаг 2. Text cleaning

Система чистит raw text:

```text
data/interim/raw_text/2026-W23_raw.md
→ data/interim/clean_text/2026-W23_clean.md
```

Задача cleaning:

- убрать технический мусор;
- сохранить даты;
- сохранить порядок записей;
- не потерять важные фрагменты.

---

### Шаг 3. Split into days

Система разбивает clean markdown на дни:

```text
data/interim/clean_text/2026-W23_clean.md
→ data/processed/days_md/YYYY-MM-DD.md
```

Также создаётся day manifest:

```text
data/processed/day_manifests/2026-W23_days.json
```

Для обычной недели ожидается около 7 дней, но система должна быть готова к пропускам или дополнительным коротким заметкам.

---

### Шаг 4. Daily extraction as day index

Для каждого дня LLM извлекает structured daily JSON.

Daily JSON не должен быть финальной заменой дневника.

Его роль:

```text
найти главные события дня
зафиксировать темы
сохранить open questions
дать короткие quotes / signals для навигации
```

Актуальная daily schema может оставаться прежней:

```json
{
  "date": "2026-06-01",
  "week_id": "2026-W23",
  "source_day_md": "data/processed/days_md/2026-06-01.md",
  "important_moments": [],
  "wins": [],
  "tensions": [],
  "emotions": [],
  "body_energy_signals": [],
  "study_signals": [],
  "ml_ds_signals": [],
  "social_signals": [],
  "decisions": [],
  "open_questions": [],
  "key_quotes": [],
  "short_summary": ""
}
```

Items обычно имеют вид:

```json
{
  "quote": "...",
  "note": "..."
}
```

На этом этапе модель не делает больших выводов за неделю.

Она отвечает:

> Что в этом дне было важно и какие признаки помогут потом найти этот день?

---

### Шаг 5. Daily verification

Verification pass — optional second LLM pass.

В текущем `run-week` verification может быть включён по умолчанию, но стратегически его стоит рассматривать как quality/debug режим.

Цель:

- проверить daily JSON против исходного day markdown;
- добавить явно пропущенные important moments;
- убрать неподтверждённые интерпретации;
- исправить неверную классификацию;
- не переписывать корректный draft без причины.

---

### Шаг 6. Optional weekly aggregation

Weekly aggregation остаётся полезной, но больше не является главным memory layer.

Модель может собирать daily JSONs в evidence-based weekly JSON для weekly snapshot.

Главное ограничение:

> weekly JSON не должен быть единственным источником для будущих ответов, потому что он сжимает и теряет детали.

Weekly JSON может использоваться для:

- краткого weekly snapshot;
- навигации по неделям;
- широкого поиска периода;
- ручной оценки недели.

Но `ask_history` должен при необходимости открывать raw day markdown.

---

### Шаг 7. Optional final weekly report

На вход финальному weekly report могут идти:

```text
weekly_json
+ selected daily JSONs
+ optional selected day markdown fragments
+ optional data/context/goals.md
+ optional data/context/context_about_me.md
```

Если цель — строгий snapshot, можно генерировать report только по weekly JSON.

Если цель — качество и полнота, report должен иметь доступ к selected raw day fragments, чтобы не писать по слишком сжатой версии недели.

На выходе:

```text
data/reports/2026-W23_weekly_report.md
```

---

## 7. Новый retrieval pipeline: v0.2 prototype

### Шаг 1. Build day index

```text
data/processed/days_json/*.json
→ data/memory/day_index.jsonl
```

Day index строится из daily JSON и содержит:

- date;
- week_id;
- source_day_md;
- important_events;
- topics;
- open_questions;
- short_summary;
- embedding_text.

---

### Шаг 2. Chunk raw day markdown

```text
data/processed/days_md/*.md
→ data/memory/chunks.jsonl
```

Chunking должен сохранять:

- date;
- week_id;
- source_day_md;
- chunk_index;
- raw text fragment.

---

### Шаг 3. Build embeddings

```text
day_index / chunks
→ embeddings.npy
→ metadata.jsonl
```

На первом этапе не нужен Qdrant.

Достаточно локальных файлов:

```text
embeddings.npy
metadata.jsonl
```

и cosine similarity.

---

### Шаг 4. Search memory

```text
query
→ query embedding
→ top-k relevant days / chunks
```

Поиск может быть двухступенчатым:

```text
query → relevant days via day_index
then open raw chunks from these days
```

Или прямым:

```text
query → relevant raw chunks
```

---

### Шаг 5. Ask history

```text
question
+ retrieved raw chunks
+ dates / source paths
→ LLM answer with evidence
```

Ответ должен содержать:

- прямой ответ;
- evidence с датами и короткими цитатами;
- limitations, если данных мало.

---

## 8. CLI-команды v0.1

Текущая главная команда:

```bash
diarylens run-week --week-id 2026-W23 --pdf data/raw/weekly/2026-W23.pdf
```

Отдельные команды:

```bash
diarylens extract-pdf <pdf_path> --week-id 2026-W23
diarylens clean-text --week-id 2026-W23
diarylens split-days --week-id 2026-W23
diarylens extract-daily --week-id 2026-W23 --verify
diarylens aggregate-week --week-id 2026-W23
diarylens generate-report --week-id 2026-W23
diarylens test-llm --model daily
```

---

## 9. Будущие CLI-команды для retrieval

Добавлять постепенно, после notebook/prototype проверки:

```bash
diarylens build-day-index
diarylens build-memory-index
diarylens search-memory "когда у меня появлялась скука"
diarylens ask-history "как менялось моё отношение к DiaryLens?"
```

---

## 10. Технический принцип в одну схему

Новая стратегическая схема:

```text
weekly / archive PDF
      ↓
PDF text extraction
      ↓
clean text
      ↓
split into days
      ↓
daily markdown = source of truth
      ↓
daily JSON = important events / topics / open loops
      ↓
day index + raw chunks
      ↓
embeddings / search
      ↓
ask_history answer with evidence from raw text
```

Weekly report остаётся optional output:

```text
daily JSONs / selected raw fragments
      ↓
optional weekly JSON
      ↓
optional weekly_report.md
```

---

## 11. Historical backfill

Historical backfill — отдельный этап, не часть v0.1.

Он нужен, чтобы обработать 2 года дневника.

```text
multiple archive PDFs
      ↓
extract text
      ↓
split into days
      ↓
daily markdown
      ↓
daily JSON day index
      ↓
raw chunk index
      ↓
embeddings
      ↓
searchable diary memory
```

Важно: historical backfill должен уметь работать с PDF неопределённой длительности.

Он не должен предполагать, что архивный PDF всегда равен неделе.

---

## 12. Работа в CursorAI

Разработка ведётся через CursorAI.

Перед генерацией кода в CursorAI лучше сначала в ChatGPT уточнять:

- что именно строим;
- какие входы;
- какие выходы;
- какие файлы должны появиться;
- какие ограничения есть;
- что пока не нужно добавлять.

Затем для CursorAI формируется короткое техническое задание.

Задачи для Cursor должны быть маленькими и проверяемыми.

Для retrieval-этапа сначала предпочтительны notebook/prototype задачи, а не переписывание production pipeline.
